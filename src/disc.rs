pub mod ader;
pub mod basis;
pub mod boundary;
// pub mod flux;
pub mod gauss_points;
pub mod geometric;
pub mod mesh;
// pub mod riemann_solver;
// pub mod advection1d_space_time_quad;
// pub mod advection1d_space_time_tri;
pub mod burgers1d;
pub mod burgers1d_space_time;
// pub mod euler1d;
use faer::{Col, linalg::solvers::DenseSolveCore, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use nalgebra::{DMatrix, DVector, coordinates::X, dvector};
use ndarray::{
    Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3,
    ArrayViewMut4, Axis, Zip, array, concatenate, s,
};
use ndarray_linalg::Inverse;
use ndarray_stats::QuantileExt;

use crate::{
    disc::{
        basis::{Basis, lagrange1d::LobattoBasis, triangle::TriangleBasis},
        boundary::{BoundaryPosition, scalar1d::PolynomialBoundary},
        geometric::Geometric2D,
        mesh::mesh2d::{Mesh2d, Status, TriangleElement},
    },
    io::write_to_vtu::{write_average, write_nodal_solutions},
};
pub trait P0Solver: Geometric2D + SpaceTimeSolver1DScalar {
    fn compute_initial_guess(&self, mesh: &Mesh2d<TriangleElement>) -> Array2<f64> {
        let mut solutions = Array2::zeros((mesh.elem_num, 1));
        self.initialize_solution(solutions.view_mut());
        let nelem = mesh.elem_num;
        let mut residuals = Array2::<f64>::zeros((nelem, 1));
        let max_iter = 5000;
        let tol = 1e-10;

        for i in 0..max_iter {
            self.compute_p0_residuals(mesh, solutions.view(), residuals.view_mut());

            let res_norm = residuals.iter().map(|x| x.powi(2)).sum::<f64>().sqrt() / (nelem as f64);

            if i % 100 == 0 {
                dbg!(&residuals);
                println!("solutions: {:?}", solutions);
                println!("PTC Iter: {}, Res norm: {}", i, res_norm);
            }

            if res_norm < tol {
                println!("PTC converged after {} iterations.", i);
                println!("solutions: {:?}", solutions);
                return solutions;
            }

            let dts = self.compute_time_steps(mesh, solutions.view());
            for ielem in 0..nelem {
                if let Status::Active(elem) = &mesh.elements[ielem] {
                    let x: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[elem.inodes[i]].as_ref().x);
                    let y: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[elem.inodes[i]].as_ref().y);
                    let area = Self::compute_element_area(&x, &y);
                    solutions[[ielem, 0]] -= dts[ielem] / area * residuals[[ielem, 0]];
                }
            }
            println!("iter: {:?}", i);
            // println!("p0_solutions: {:?}", solutions);
        }
        println!("PTC did not converge within {} iterations", max_iter);
        solutions
    }

    fn compute_p0_residuals(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        solutions: ArrayView2<f64>,
        mut residuals: ArrayViewMut2<f64>,
    ) {
        residuals.fill(0.0);

        // Internal edges
        for &iedge in &mesh.interior_edges {
            if let Status::Active(edge) = &mesh.edges[iedge] {
                let ileft = edge.parents[0];
                let iright = edge.parents[1];

                let lelem = mesh.elements[ileft].as_ref();

                let u_left = solutions[(ileft, 0)];
                let u_right = solutions[(iright, 0)];

                let local_ids = &edge.local_ids;
                let n0 = mesh.nodes[lelem.inodes[local_ids[0]]].as_ref();
                let n1 = mesh.nodes[lelem.inodes[(local_ids[0] + 1) % 3]].as_ref();

                let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();

                let flux = self.compute_numerical_flux(u_left, u_right, n0.x, n1.x, n0.y, n1.y);

                residuals[(ileft, 0)] += flux * edge_length;
                residuals[(iright, 0)] -= flux * edge_length;
            }
        }

        // Constant boundaries
        for bnd in &mesh.boundaries.constant {
            let ub = bnd.value;
            for &iedge in &bnd.iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let u = solutions[(ielem, 0)];
                    let local_id = edge.local_ids[0];
                    let n0 = mesh.nodes[elem.inodes[local_id]].as_ref();
                    let n1 = mesh.nodes[elem.inodes[(local_id + 1) % 3]].as_ref();
                    let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();

                    // For boundary edges, the node ordering is assumed to be counter-clockwise
                    // for the parent element, so the normal computed from (n0, n1) is outward-pointing.
                    let flux = self.compute_boundary_flux(u, ub, n0.x, n1.x, n0.y, n1.y);

                    residuals[(ielem, 0)] += flux * edge_length;
                }
            }
        }
        // Function boundaries
        for bnd in &mesh.boundaries.function {
            let iedges = &bnd.iedges;
            let func = bnd.func;

            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let u = solutions[(ielem, 0)];
                    let local_id = edge.local_ids[0];
                    let n0 = mesh.nodes[elem.inodes[local_id]].as_ref();
                    let n1 = mesh.nodes[elem.inodes[(local_id + 1) % 3]].as_ref();
                    let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();
                    let (mid_x, mid_y) = (0.5 * (n0.x + n1.x), 0.5 * (n0.y + n1.y));
                    let bnd_value = func(mid_x, mid_y);
                    let flux = self.compute_boundary_flux(u, bnd_value, n0.x, n1.x, n0.y, n1.y);

                    residuals[(ielem, 0)] += flux * edge_length;
                }
            }
        }
        // Polynomial boundaries
        /*
        for bnd in &mesh.boundaries.polynomial {
            let iedges = &bnd.iedges;
            let nodal_coeffs = &bnd.nodal_coeffs;
            let n0_bnd = mesh.nodes[bnd.inodes[0]].as_ref();
            let n1_bnd = mesh.nodes[bnd.inodes[1]].as_ref();
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let lelem = mesh.elements[ielem].as_ref();
                    let u = solutions[(ielem, 0)];
                    let local_id = edge.local_ids[0];
                    let n0 = mesh.nodes[lelem.inodes[local_id]].as_ref();
                    let n1 = mesh.nodes[lelem.inodes[(local_id + 1) % 3]].as_ref();
                    let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();
                    let xi = array![0.5];
                    let bnd_value = Self::compute_boundary_value_by_interpolation(
                        &nodal_coeffs,
                        self.basis().basis1d.n,
                        &xi,
                        &self.basis().basis1d.inv_vandermonde,
                        n0.x,
                        n1.x,
                        n0.y,
                        n1.y,
                        n0_bnd.x,
                        n1_bnd.x,
                        n0_bnd.y,
                        n1_bnd.y,
                    );

                    let flux = self.compute_boundary_flux(u, bnd_value[0], n0.x, n1.x, n0.y, n1.y);
                    residuals[(ielem, 0)] += flux * edge_length;
                }
            }
        }
        */
        // Open boundaries
        for bnd in &mesh.boundaries.open {
            let iedges = &bnd.iedges;
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let lelem = mesh.elements[ielem].as_ref();
                    let u = solutions[(ielem, 0)];
                    let local_id = edge.local_ids[0];
                    let n0 = mesh.nodes[lelem.inodes[local_id]].as_ref();
                    let n1 = mesh.nodes[lelem.inodes[(local_id + 1) % 3]].as_ref();
                    let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();
                    let flux = self.compute_open_boundary_flux(u, n0.x, n1.x, n0.y, n1.y);
                    residuals[(ielem, 0)] += flux * edge_length;
                }
            }
        }
    }
    fn compute_time_steps(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        solutions: ArrayView2<f64>,
    ) -> Array1<f64>;
}

pub trait SpaceTimeSolver1DScalar: Geometric2D {
    fn basis(&self) -> &TriangleBasis;
    fn enriched_basis(&self) -> &TriangleBasis;
    fn interp_node_to_cubature(&self) -> &Array2<f64>;
    fn interp_node_to_enriched_cubature(&self) -> &Array2<f64>;
    fn interp_node_to_enriched_quadrature(&self) -> &Array2<f64>;
    // fn mesh(&self) -> &Mesh2d<TriangleElement>;
    // fn mesh_mut(&mut self) -> &mut Mesh2d<TriangleElement>;

    fn compute_interp_matrix_1d(
        n: usize,
        inv_vandermonde: ArrayView2<f64>,
        r: ArrayView1<f64>,
    ) -> Array2<f64> {
        let v = TriangleBasis::vandermonde1d(n, r);
        let interp_matrix = v.dot(&inv_vandermonde);
        interp_matrix
    }
    fn compute_interp_matrix_2d(
        n: usize,
        inv_vandermonde: ArrayView2<f64>,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
    ) -> Array2<f64> {
        let v = TriangleBasis::vandermonde2d(n, r, s);
        let interp_matrix = v.dot(&inv_vandermonde);
        interp_matrix
    }
    fn compute_residuals(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        solutions: ArrayView2<f64>,
        mut residuals: ArrayViewMut2<f64>,
        is_enriched: bool,
    ) {
        let basis = {
            if is_enriched {
                &self.enriched_basis()
            } else {
                &self.basis()
            }
        };
        let ncell_basis = basis.r.len();
        let nedge_basis = basis.quad_p.len();
        let edge_weights = &basis.quad_w;
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                let interp_sol = if is_enriched {
                    self.interp_node_to_enriched_cubature()
                        .dot(&solutions.slice(s![ielem, ..]))
                } else {
                    self.interp_node_to_cubature()
                        .dot(&solutions.slice(s![ielem, ..]))
                };
                for itest_func in 0..ncell_basis {
                    let res = self.volume_integral(
                        basis,
                        itest_func,
                        &interp_sol.as_slice().unwrap(),
                        &x_slice,
                        &y_slice,
                    );
                    residuals[(ielem, itest_func)] += res;
                }
            }
        }
        for &iedge in mesh.interior_edges.iter() {
            if let Status::Active(edge) = &mesh.edges[iedge] {
                let ilelem = edge.parents[0];
                let irelem = edge.parents[1];
                let left_elem = mesh.elements[ilelem].as_ref();
                let right_elem = mesh.elements[irelem].as_ref();
                let left_inodes = &left_elem.inodes;
                let right_inodes = &right_elem.inodes;
                let left_x_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.nodes[left_inodes[i]].as_ref().x);
                let left_y_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.nodes[left_inodes[i]].as_ref().y);
                let right_x_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.nodes[right_inodes[i]].as_ref().x);
                let right_y_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.nodes[right_inodes[i]].as_ref().y);
                let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_ids = &edge.local_ids;
                let left_ref_normal = Self::compute_ref_normal(local_ids[0]);
                let right_ref_normal = Self::compute_ref_normal(local_ids[1]);
                let left_edge_length = Self::compute_ref_edge_length(local_ids[0]);
                let right_edge_length = Self::compute_ref_edge_length(local_ids[1]);
                let (left_sol_slice, right_sol_slice) = {
                    let left_sol_slice = solutions.slice(s![ilelem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let right_sol_slice = solutions.slice(s![irelem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_ids[1], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    if is_enriched {
                        (
                            self.interp_node_to_enriched_quadrature()
                                .dot(&left_sol_slice),
                            self.interp_node_to_enriched_quadrature()
                                .dot(&right_sol_slice),
                        )
                    } else {
                        (left_sol_slice, right_sol_slice)
                    }
                };
                let left_xi_slice = basis.r.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
                let left_eta_slice = basis.s.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
                let right_xi_slice = basis.r.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[1], ..])
                        .as_slice()
                        .unwrap(),
                );
                let right_eta_slice = basis.s.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[1], ..])
                        .as_slice()
                        .unwrap(),
                );
                for i in 0..nedge_basis {
                    let left_value = left_sol_slice[i];
                    let right_value = right_sol_slice[nedge_basis - 1 - i];
                    let num_flux = self.compute_numerical_flux(
                        left_value,
                        right_value,
                        left_x_slice[local_ids[0]],
                        left_x_slice[(local_ids[0] + 1) % 3],
                        left_y_slice[local_ids[0]],
                        left_y_slice[(local_ids[0] + 1) % 3],
                    );
                    let left_scaling = self.compute_flux_scaling(
                        left_xi_slice[i],
                        left_eta_slice[i],
                        left_ref_normal,
                        &left_x_slice,
                        &left_y_slice,
                    );
                    let right_scaling = self.compute_flux_scaling(
                        right_xi_slice[nedge_basis - 1 - i],
                        right_eta_slice[nedge_basis - 1 - i],
                        right_ref_normal,
                        &right_x_slice,
                        &right_y_slice,
                    );
                    let left_transformed_flux = num_flux * left_scaling;
                    let right_transformed_flux = -num_flux * right_scaling;
                    let left_itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                    let right_itest_func =
                        basis.nodes_along_edges[(local_ids[1], nedge_basis - 1 - i)];
                    residuals[(ilelem, left_itest_func)] +=
                        left_edge_length * edge_weights[i] * left_transformed_flux;
                    residuals[(irelem, right_itest_func)] +=
                        right_edge_length * edge_weights[i] * right_transformed_flux;
                }
            }
        }
        // Constant boundaries
        for bnd in &mesh.boundaries.constant {
            let iedges = &bnd.iedges;
            let bnd_value = bnd.value;
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let inodes = &elem.inodes;
                    let x_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                    let y_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_ids = &edge.local_ids;
                    let ref_normal = Self::compute_ref_normal(local_ids[0]);
                    let edge_length = Self::compute_ref_edge_length(local_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.r.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.s.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];
                        let boundary_flux = self.compute_boundary_flux(
                            sol_slice[i],
                            bnd_value,
                            x_slice[local_ids[0]],
                            x_slice[(local_ids[0] + 1) % 3],
                            y_slice[local_ids[0]],
                            y_slice[(local_ids[0] + 1) % 3],
                        );
                        let scaling =
                            self.compute_flux_scaling(xi, eta, ref_normal, &x_slice, &y_slice);
                        let transformed_flux = boundary_flux * scaling;
                        let itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                        residuals[(ielem, itest_func)] +=
                            edge_length * edge_weights[i] * transformed_flux;
                    }
                }
            }
        }
        // Function boundaries
        for bnd in &mesh.boundaries.function {
            let iedges = &bnd.iedges;
            let func = &bnd.func;
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let inodes = &elem.inodes;
                    let x_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                    let y_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_ids = &edge.local_ids;
                    let ref_normal = Self::compute_ref_normal(local_ids[0]);
                    let edge_length = Self::compute_ref_edge_length(local_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.r.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.s.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );

                    // Compute the physical coordinates of the quadrature points on the edge
                    let n1 = 1.0 - &xi_slice - &eta_slice;
                    let n2 = xi_slice.to_owned();
                    let n3 = eta_slice.to_owned();

                    let x_phys = &n1 * x_slice[0] + &n2 * x_slice[1] + &n3 * x_slice[2];
                    let y_phys = &n1 * y_slice[0] + &n2 * y_slice[1] + &n3 * y_slice[2];

                    let bnd_values = Zip::from(&x_phys)
                        .and(&y_phys)
                        .map_collect(|&x, &y| func(x, y));

                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];

                        let boundary_flux = self.compute_boundary_flux(
                            sol_slice[i],
                            bnd_values[i],
                            x_slice[local_ids[0]],
                            x_slice[(local_ids[0] + 1) % 3],
                            y_slice[local_ids[0]],
                            y_slice[(local_ids[0] + 1) % 3],
                        );
                        let scaling =
                            self.compute_flux_scaling(xi, eta, ref_normal, &x_slice, &y_slice);
                        let transformed_flux = boundary_flux * scaling;
                        let itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                        residuals[(ielem, itest_func)] +=
                            edge_length * edge_weights[i] * transformed_flux;
                    }
                }
            }
        }
        // Polynomial boundaries
        /*
        for bnd in &mesh.boundaries.polynomial {
            let iedges = &bnd.iedges;
            let nodal_coeffs = &bnd.nodal_coeffs;
            let n0_bnd = mesh.nodes[bnd.inodes[0]].as_ref();
            let n1_bnd = mesh.nodes[bnd.inodes[1]].as_ref();
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let inodes = &elem.inodes;
                    let x_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                    let y_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_ids = &edge.local_ids;
                    let ref_normal = Self::compute_ref_normal(local_ids[0]);
                    let edge_length = Self::compute_ref_edge_length(local_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.r.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.s.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let n0 = mesh.nodes[elem.inodes[local_ids[0]]].as_ref();
                    let n1 = mesh.nodes[elem.inodes[(local_ids[0] + 1) % 3]].as_ref();
                    let bnd_values = Self::compute_boundary_value_by_interpolation(
                        &nodal_coeffs,
                        self.basis().basis1d.n,
                        &basis.basis1d.xi,
                        &self.basis().basis1d.inv_vandermonde,
                        n0.x,
                        n1.x,
                        n0.y,
                        n1.y,
                        n0_bnd.x,
                        n1_bnd.x,
                        n0_bnd.y,
                        n1_bnd.y,
                    );
                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];
                        let boundary_flux = self.compute_boundary_flux(
                            sol_slice[i],
                            bnd_values[i],
                            x_slice[local_ids[0]],
                            x_slice[(local_ids[0] + 1) % 3],
                            y_slice[local_ids[0]],
                            y_slice[(local_ids[0] + 1) % 3],
                        );
                        let scaling =
                            self.compute_flux_scaling(xi, eta, ref_normal, &x_slice, &y_slice);
                        let transformed_flux = boundary_flux * scaling;
                        let itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                        residuals[(ielem, itest_func)] +=
                            edge_length * edge_weights[i] * transformed_flux;
                    }
                }
            }
        }
        */
        // Open boundaries
        for bnd in &mesh.boundaries.open {
            let iedges = &bnd.iedges;
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let inodes = &elem.inodes;
                    let x_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                    let y_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_ids = &edge.local_ids;
                    let ref_normal = Self::compute_ref_normal(local_ids[0]);
                    let edge_length = Self::compute_ref_edge_length(local_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.r.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.s.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    for i in 0..nedge_basis {
                        let u = sol_slice[i];
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];
                        let boundary_flux = self.compute_open_boundary_flux(
                            u,
                            x_slice[local_ids[0]],
                            x_slice[(local_ids[0] + 1) % 3],
                            y_slice[local_ids[0]],
                            y_slice[(local_ids[0] + 1) % 3],
                        );
                        let scaling =
                            self.compute_flux_scaling(xi, eta, ref_normal, &x_slice, &y_slice);
                        let transformed_flux = boundary_flux * scaling;
                        let itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                        residuals[(ielem, itest_func)] +=
                            edge_length * edge_weights[i] * transformed_flux;
                    }
                }
            }
        }
    }
    fn compute_residuals_and_derivatives(
        &self,
        mesh: &mut Mesh2d<TriangleElement>,
        solutions: ArrayView2<f64>,
        mut residuals: ArrayViewMut2<f64>,
        mut dsol: ArrayViewMut4<f64>,
        mut dx: ArrayViewMut3<f64>,
        mut dy: ArrayViewMut3<f64>,
        is_enriched: bool,
        iter: usize,
    ) {
        let basis = {
            if is_enriched {
                &self.enriched_basis()
            } else {
                &self.basis()
            }
        };
        let unenriched_ncell_basis = self.basis().r.len();
        let ncell_basis = basis.r.len();
        let nedge_basis = basis.quad_p.len();
        let edge_weights = &basis.quad_w;
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                let interp_matrix = if is_enriched {
                    &self.interp_node_to_enriched_cubature()
                } else {
                    &self.interp_node_to_cubature()
                };
                let interp_sol = interp_matrix.dot(&solutions.slice(s![ielem, ..]));
                for itest_func in 0..ncell_basis {
                    let mut dvol_sol: Array1<f64> = Array1::zeros(basis.cub_r.len());
                    let mut dvol_x: Array1<f64> = Array1::zeros(3);
                    let mut dvol_y: Array1<f64> = Array1::zeros(3);
                    let res = self.dvolume(
                        basis,
                        itest_func,
                        &interp_sol.as_slice().unwrap(),
                        dvol_sol.as_slice_mut().unwrap(),
                        &x_slice,
                        dvol_x.as_slice_mut().unwrap(),
                        &y_slice,
                        dvol_y.as_slice_mut().unwrap(),
                        1.0,
                    );
                    residuals[(ielem, itest_func)] += res;
                    let dres_dsol_dofs = interp_matrix.t().dot(&dvol_sol);

                    dsol.slice_mut(s![ielem, itest_func, ielem, ..])
                        .scaled_add(1.0, &dres_dsol_dofs);
                    for i in 0..3 {
                        dx[(ielem, itest_func, inodes[i])] += dvol_x[i];
                        dy[(ielem, itest_func, inodes[i])] += dvol_y[i];
                    }
                }
            }
        }
        for &iedge in mesh.interior_edges.iter() {
            if let Status::Active(edge) = &mesh.edges[iedge] {
                let ilelem = edge.parents[0];
                let irelem = edge.parents[1];
                let left_elem = mesh.elements[ilelem].as_ref();
                let right_elem = mesh.elements[irelem].as_ref();
                let left_inodes = &left_elem.inodes;
                let right_inodes = &right_elem.inodes;
                let left_x_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.nodes[left_inodes[i]].as_ref().x);
                let left_y_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.nodes[left_inodes[i]].as_ref().y);
                let right_x_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.nodes[right_inodes[i]].as_ref().x);
                let right_y_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.nodes[right_inodes[i]].as_ref().y);
                let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_ids = &edge.local_ids;
                let left_ref_normal = Self::compute_ref_normal(local_ids[0]);
                let right_ref_normal = Self::compute_ref_normal(local_ids[1]);
                let left_edge_length = Self::compute_ref_edge_length(local_ids[0]);
                let right_edge_length = Self::compute_ref_edge_length(local_ids[1]);
                let (left_sol_slice, right_sol_slice) = {
                    let left_sol_slice = solutions.slice(s![ilelem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let right_sol_slice = solutions.slice(s![irelem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_ids[1], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    if is_enriched {
                        (
                            self.interp_node_to_enriched_quadrature()
                                .dot(&left_sol_slice),
                            self.interp_node_to_enriched_quadrature()
                                .dot(&right_sol_slice),
                        )
                    } else {
                        (left_sol_slice, right_sol_slice)
                    }
                };
                let left_xi_slice = basis.r.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
                let left_eta_slice = basis.s.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
                let right_xi_slice = basis.r.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[1], ..])
                        .as_slice()
                        .unwrap(),
                );
                let right_eta_slice = basis.s.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[1], ..])
                        .as_slice()
                        .unwrap(),
                );
                for i in 0..nedge_basis {
                    let left_value = left_sol_slice[i];
                    let right_value = right_sol_slice[nedge_basis - 1 - i];
                    let mut dflux_dleft_x = [0.0; 3];
                    let mut dflux_dleft_y = [0.0; 3];
                    let mut dflux_dright_x = [0.0; 3];
                    let mut dflux_dright_y = [0.0; 3];
                    let (
                        num_flux,
                        dflux_dul,
                        dflux_dur,
                        dflux_dx0,
                        dflux_dx1,
                        dflux_dy0,
                        dflux_dy1,
                    ): (f64, f64, f64, f64, f64, f64, f64) = self.dnum_flux(
                        left_value,
                        right_value,
                        left_x_slice[local_ids[0]],
                        left_x_slice[(local_ids[0] + 1) % 3],
                        left_y_slice[local_ids[0]],
                        left_y_slice[(local_ids[0] + 1) % 3],
                        1.0,
                    );
                    dflux_dleft_x[local_ids[0]] = dflux_dx0;
                    dflux_dleft_x[(local_ids[0] + 1) % 3] = dflux_dx1;
                    dflux_dleft_y[local_ids[0]] = dflux_dy0;
                    dflux_dleft_y[(local_ids[0] + 1) % 3] = dflux_dy1;
                    dflux_dright_x[local_ids[1]] = dflux_dx1;
                    dflux_dright_x[(local_ids[1] + 1) % 3] = dflux_dx0;
                    dflux_dright_y[local_ids[1]] = dflux_dy1;
                    dflux_dright_y[(local_ids[1] + 1) % 3] = dflux_dy0;

                    let mut dleft_scaling_dx = [0.0; 3];
                    let mut dleft_scaling_dy = [0.0; 3];
                    let mut dright_scaling_dx = [0.0; 3];
                    let mut dright_scaling_dy = [0.0; 3];
                    let left_scaling: f64 = self.dscaling(
                        left_xi_slice[i],
                        left_eta_slice[i],
                        left_ref_normal,
                        left_x_slice.as_slice(),
                        dleft_scaling_dx.as_mut_slice(),
                        left_y_slice.as_slice(),
                        dleft_scaling_dy.as_mut_slice(),
                        1.0,
                    );
                    let right_scaling: f64 = self.dscaling(
                        right_xi_slice[nedge_basis - 1 - i],
                        right_eta_slice[nedge_basis - 1 - i],
                        right_ref_normal,
                        right_x_slice.as_slice(),
                        dright_scaling_dx.as_mut_slice(),
                        right_y_slice.as_slice(),
                        dright_scaling_dy.as_mut_slice(),
                        1.0,
                    );

                    let left_transformed_flux = num_flux * left_scaling;
                    let right_transformed_flux = -num_flux * right_scaling;

                    let dleft_transformed_flux_dul = left_scaling * dflux_dul;
                    let dleft_transformed_flux_dur = left_scaling * dflux_dur;

                    let dleft_transformed_flux_dx = &ArrayView1::from(&dleft_scaling_dx) * num_flux
                        + &ArrayView1::from(&dflux_dleft_x) * left_scaling;
                    let dleft_transformed_flux_dy = &ArrayView1::from(&dleft_scaling_dy) * num_flux
                        + &ArrayView1::from(&dflux_dleft_y) * left_scaling;

                    let dright_transformed_flux_dul = -right_scaling * dflux_dul;
                    let dright_transformed_flux_dur = -right_scaling * dflux_dur;

                    let dright_transformed_flux_dx = -(&ArrayView1::from(&dright_scaling_dx)
                        * num_flux
                        + &ArrayView1::from(&dflux_dright_x) * right_scaling);
                    let dright_transformed_flux_dy = -(&ArrayView1::from(&dright_scaling_dy)
                        * num_flux
                        + &ArrayView1::from(&dflux_dright_y) * right_scaling);

                    let left_itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                    let right_itest_func =
                        basis.nodes_along_edges[(local_ids[1], nedge_basis - 1 - i)];

                    residuals[(ilelem, left_itest_func)] +=
                        left_edge_length * edge_weights[i] * left_transformed_flux;
                    residuals[(irelem, right_itest_func)] +=
                        right_edge_length * edge_weights[i] * right_transformed_flux;

                    if is_enriched {
                        // derivatives w.r.t. left value
                        for (j, &isol_node) in sol_nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .indexed_iter()
                        {
                            dsol[(ilelem, left_itest_func, ilelem, isol_node)] += 0.5
                                * left_edge_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()[(i, j)]
                                * dleft_transformed_flux_dul;
                            dsol[(irelem, right_itest_func, irelem, isol_node)] += 0.5
                                * right_edge_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()[(i, j)]
                                * dright_transformed_flux_dul;
                        }
                        // derivatives w.r.t. right value
                        for (j, &isol_node) in sol_nodes_along_edges
                            .slice(s![local_ids[1], ..])
                            .indexed_iter()
                        {
                            dsol[(ilelem, left_itest_func, ilelem, isol_node)] += 0.5
                                * left_edge_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()
                                    [(nedge_basis - 1 - i, j)]
                                * dleft_transformed_flux_dur;
                            dsol[(irelem, right_itest_func, irelem, isol_node)] += 0.5
                                * right_edge_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()
                                    [(nedge_basis - 1 - i, j)]
                                * dright_transformed_flux_dur;
                        }
                    } else {
                        let left_sol_nodes = sol_nodes_along_edges.slice(s![local_ids[0], ..]);
                        let right_sol_nodes = sol_nodes_along_edges.slice(s![local_ids[1], ..]);
                        // derivatives w.r.t. left value
                        dsol[(ilelem, left_itest_func, ilelem, left_itest_func)] +=
                            left_edge_length * edge_weights[i] * dleft_transformed_flux_dul;
                        dsol[(irelem, right_itest_func, ilelem, left_itest_func)] +=
                            right_edge_length * edge_weights[i] * dright_transformed_flux_dul;
                        // derivatives w.r.t. right value
                        dsol[(ilelem, left_itest_func, irelem, right_itest_func)] +=
                            left_edge_length * edge_weights[i] * dleft_transformed_flux_dur;
                        dsol[(irelem, right_itest_func, irelem, right_itest_func)] +=
                            right_edge_length * edge_weights[i] * dright_transformed_flux_dur;
                    }
                    for j in 0..3 {
                        dx[(ilelem, left_itest_func, left_elem.inodes[j])] +=
                            left_edge_length * edge_weights[i] * dleft_transformed_flux_dx[j];
                        dy[(ilelem, left_itest_func, left_elem.inodes[j])] +=
                            left_edge_length * edge_weights[i] * dleft_transformed_flux_dy[j];
                        dx[(irelem, right_itest_func, right_elem.inodes[j])] +=
                            right_edge_length * edge_weights[i] * dright_transformed_flux_dx[j];
                        dy[(irelem, right_itest_func, right_elem.inodes[j])] +=
                            right_edge_length * edge_weights[i] * dright_transformed_flux_dy[j];
                    }
                }
            }
        }
        // Constant boundaries
        for bnd in &mesh.boundaries.constant {
            let iedges = &bnd.iedges;
            let bnd_value = bnd.value;
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let inodes = &elem.inodes;
                    let x_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                    let y_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_ids = &edge.local_ids;
                    let ref_normal = Self::compute_ref_normal(local_ids[0]);
                    let edge_length = Self::compute_ref_edge_length(local_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.r.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.s.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];
                        let (boundary_flux, dflux_du, dflux_dx0, dflux_dx1, dflux_dy0, dflux_dy1): (
                            f64,
                            f64,
                            f64,
                            f64,
                            f64,
                            f64,
                        ) = self.dbnd_flux(
                            sol_slice[i],
                            bnd_value,
                            x_slice[local_ids[0]],
                            x_slice[(local_ids[0] + 1) % 3],
                            y_slice[local_ids[0]],
                            y_slice[(local_ids[0] + 1) % 3],
                            1.0,
                        );

                        let mut dflux_dx = [0.0; 3];
                        let mut dflux_dy = [0.0; 3];
                        dflux_dx[local_ids[0]] = dflux_dx0;
                        dflux_dx[(local_ids[0] + 1) % 3] = dflux_dx1;
                        dflux_dy[local_ids[0]] = dflux_dy0;
                        dflux_dy[(local_ids[0] + 1) % 3] = dflux_dy1;

                        let mut dscaling_dx = [0.0; 3];
                        let mut dscaling_dy = [0.0; 3];
                        let scaling: f64 = self.dscaling(
                            xi,
                            eta,
                            ref_normal,
                            &x_slice,
                            dscaling_dx.as_mut_slice(),
                            &y_slice,
                            dscaling_dy.as_mut_slice(),
                            1.0,
                        );
                        let transformed_flux = boundary_flux * scaling;

                        let dtransformed_flux_du = dflux_du * scaling;
                        let dtransformed_flux_dx = &(&ArrayView1::from(&dflux_dx) * scaling)
                            + &(&ArrayView1::from(&dscaling_dx) * boundary_flux);
                        let dtransformed_flux_dy = &(&ArrayView1::from(&dflux_dy) * scaling)
                            + &(&ArrayView1::from(&dscaling_dy) * boundary_flux);

                        let itest_func = basis.nodes_along_edges[(local_ids[0], i)];

                        residuals[(ielem, itest_func)] +=
                            edge_length * edge_weights[i] * transformed_flux;

                        for j in 0..3 {
                            dx[(ielem, itest_func, inodes[j])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_dx[j];
                            dy[(ielem, itest_func, inodes[j])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_dy[j];
                        }

                        if is_enriched {
                            for (j, &isol_node) in sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .indexed_iter()
                            {
                                dsol[(ielem, itest_func, ielem, isol_node)] += 0.5
                                    * edge_length
                                    * edge_weights[i]
                                    * self.interp_node_to_enriched_quadrature()[(i, j)]
                                    * dtransformed_flux_du;
                            }
                        } else {
                            let sol_nodes = sol_nodes_along_edges.slice(s![local_ids[0], ..]);
                            dsol[(ielem, itest_func, ielem, sol_nodes[i])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_du;
                        }
                    }
                }
            }
        }
        // Function boundaries
        for bnd in &mesh.boundaries.function {
            let iedges = &bnd.iedges;
            let func = &bnd.func;
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let inodes = &elem.inodes;
                    let x_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                    let y_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_ids = &edge.local_ids;
                    let ref_normal = Self::compute_ref_normal(local_ids[0]);
                    let edge_length = Self::compute_ref_edge_length(local_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.r.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.s.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );

                    // Compute the physical coordinates of the quadrature points on the edge
                    let shape_func = [
                        1.0 - &xi_slice - &eta_slice,
                        xi_slice.to_owned(),
                        eta_slice.to_owned(),
                    ];

                    let x_phys = &shape_func[0] * x_slice[0]
                        + &shape_func[1] * x_slice[1]
                        + &shape_func[2] * x_slice[2];
                    let y_phys = &shape_func[0] * y_slice[0]
                        + &shape_func[1] * y_slice[1]
                        + &shape_func[2] * y_slice[2];

                    let bnd_values = Zip::from(&x_phys)
                        .and(&y_phys)
                        .map_collect(|&x, &y| func(x, y));

                    // Compute the derivatives of boundary values with respect to coordinates changes of the edge nodes
                    let bnd_values_x0_perturbed = {
                        let mut x_slice_perturbed = x_slice;
                        x_slice_perturbed[local_ids[0]] += 1e-6;
                        let x_phys = &shape_func[0] * x_slice_perturbed[0]
                            + &shape_func[1] * x_slice_perturbed[1]
                            + &shape_func[2] * x_slice_perturbed[2];
                        let y_phys = &shape_func[0] * y_slice[0]
                            + &shape_func[1] * y_slice[1]
                            + &shape_func[2] * y_slice[2];
                        Zip::from(&x_phys)
                            .and(&y_phys)
                            .map_collect(|&x, &y| func(x, y))
                    };
                    let bnd_values_x1_perturbed = {
                        let mut x_slice_perturbed = x_slice;
                        x_slice_perturbed[(local_ids[0] + 1) % 3] += 1e-6;
                        let x_phys = &shape_func[0] * x_slice_perturbed[0]
                            + &shape_func[1] * x_slice_perturbed[1]
                            + &shape_func[2] * x_slice_perturbed[2];
                        let y_phys = &shape_func[0] * y_slice[0]
                            + &shape_func[1] * y_slice[1]
                            + &shape_func[2] * y_slice[2];
                        Zip::from(&x_phys)
                            .and(&y_phys)
                            .map_collect(|&x, &y| func(x, y))
                    };
                    let bnd_values_y0_perturbed = {
                        let mut y_slice_perturbed = y_slice;
                        y_slice_perturbed[local_ids[0]] += 1e-6;
                        let x_phys = &shape_func[0] * x_slice[0]
                            + &shape_func[1] * x_slice[1]
                            + &shape_func[2] * x_slice[2];
                        let y_phys = &shape_func[0] * y_slice_perturbed[0]
                            + &shape_func[1] * y_slice_perturbed[1]
                            + &shape_func[2] * y_slice_perturbed[2];
                        Zip::from(&x_phys)
                            .and(&y_phys)
                            .map_collect(|&x, &y| func(x, y))
                    };
                    let bnd_values_y1_perturbed = {
                        let mut y_slice_perturbed = y_slice;
                        y_slice_perturbed[(local_ids[0] + 1) % 3] += 1e-6;
                        let x_phys = &shape_func[0] * x_slice[0]
                            + &shape_func[1] * x_slice[1]
                            + &shape_func[2] * x_slice[2];
                        let y_phys = &shape_func[0] * y_slice_perturbed[0]
                            + &shape_func[1] * y_slice_perturbed[1]
                            + &shape_func[2] * y_slice_perturbed[2];
                        Zip::from(&x_phys)
                            .and(&y_phys)
                            .map_collect(|&x, &y| func(x, y))
                    };

                    let dbnd_values_dx0 = (&bnd_values_x0_perturbed - &bnd_values) / 1e-6;
                    let dbnd_values_dx1 = (&bnd_values_x1_perturbed - &bnd_values) / 1e-6;
                    let dbnd_values_dy0 = (&bnd_values_y0_perturbed - &bnd_values) / 1e-6;
                    let dbnd_values_dy1 = (&bnd_values_y1_perturbed - &bnd_values) / 1e-6;

                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];

                        let (
                            boundary_flux,
                            dflux_du,
                            dflux_dbnd_value,
                            dflux_dx0,
                            dflux_dx1,
                            dflux_dy0,
                            dflux_dy1,
                        ): (f64, f64, f64, f64, f64, f64, f64) = self.dnum_flux(
                            sol_slice[i],
                            bnd_values[i],
                            x_slice[local_ids[0]],
                            x_slice[(local_ids[0] + 1) % 3],
                            y_slice[local_ids[0]],
                            y_slice[(local_ids[0] + 1) % 3],
                            1.0,
                        );
                        let mut dflux_dx = [0.0; 3];
                        let mut dflux_dy = [0.0; 3];
                        dflux_dx[local_ids[0]] = dflux_dx0 + dflux_dbnd_value * dbnd_values_dx0[i];
                        dflux_dx[(local_ids[0] + 1) % 3] =
                            dflux_dx1 + dflux_dbnd_value * dbnd_values_dx1[i];
                        dflux_dy[local_ids[0]] = dflux_dy0 + dflux_dbnd_value * dbnd_values_dy0[i];
                        dflux_dy[(local_ids[0] + 1) % 3] =
                            dflux_dy1 + dflux_dbnd_value * dbnd_values_dy1[i];

                        let mut dscaling_dx = [0.0; 3];
                        let mut dscaling_dy = [0.0; 3];
                        let scaling: f64 = self.dscaling(
                            xi,
                            eta,
                            ref_normal,
                            &x_slice,
                            dscaling_dx.as_mut_slice(),
                            &y_slice,
                            dscaling_dy.as_mut_slice(),
                            1.0,
                        );
                        let transformed_flux = boundary_flux * scaling;

                        let dtransformed_flux_du = dflux_du * scaling;
                        let dtransformed_flux_dx = &(&ArrayView1::from(&dflux_dx) * scaling)
                            + &(&ArrayView1::from(&dscaling_dx) * boundary_flux);
                        let dtransformed_flux_dy = &(&ArrayView1::from(&dflux_dy) * scaling)
                            + &(&ArrayView1::from(&dscaling_dy) * boundary_flux);

                        let itest_func = basis.nodes_along_edges[(local_ids[0], i)];

                        residuals[(ielem, itest_func)] +=
                            edge_length * edge_weights[i] * transformed_flux;

                        for j in 0..3 {
                            dx[(ielem, itest_func, inodes[j])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_dx[j];
                            dy[(ielem, itest_func, inodes[j])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_dy[j];
                        }

                        if is_enriched {
                            for (j, &isol_node) in sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .indexed_iter()
                            {
                                dsol[(ielem, itest_func, ielem, isol_node)] += edge_length
                                    * edge_weights[i]
                                    * self.interp_node_to_enriched_quadrature()[(i, j)]
                                    * dtransformed_flux_du;
                            }
                        } else {
                            let sol_nodes = sol_nodes_along_edges.slice(s![local_ids[0], ..]);
                            dsol[(ielem, itest_func, ielem, sol_nodes[i])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_du;
                        }
                    }
                }
            }
        }
        // Polynomial boundaries
        /*
        for bnd in &mesh.boundaries.polynomial {
            let iedges = &bnd.iedges;
            let nodal_coeffs = &bnd.nodal_coeffs;
            let n0_bnd = mesh.nodes[bnd.inodes[0]].as_ref();
            let n1_bnd = mesh.nodes[bnd.inodes[1]].as_ref();
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let inodes = &elem.inodes;
                    let x_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                    let y_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_ids = &edge.local_ids;
                    let ref_normal = Self::compute_ref_normal(local_ids[0]);
                    let edge_length = Self::compute_ref_edge_length(local_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.r.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.s.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let n0 = mesh.nodes[elem.inodes[local_ids[0]]].as_ref();
                    let n1 = mesh.nodes[elem.inodes[(local_ids[0] + 1) % 3]].as_ref();

                    let bnd_values = Self::compute_boundary_value_by_interpolation(
                        &nodal_coeffs,
                        self.basis().basis1d.n,
                        &basis.basis1d.xi,
                        &self.basis().basis1d.inv_vandermonde,
                        n0.x,
                        n1.x,
                        n0.y,
                        n1.y,
                        n0_bnd.x,
                        n1_bnd.x,
                        n0_bnd.y,
                        n1_bnd.y,
                    );

                    let bnd_values_x0_perturbed = Self::compute_boundary_value_by_interpolation(
                        &nodal_coeffs,
                        self.basis().basis1d.n,
                        &basis.basis1d.xi,
                        &self.basis().basis1d.inv_vandermonde,
                        n0.x + 1e-6,
                        n1.x,
                        n0.y,
                        n1.y,
                        n0_bnd.x,
                        n1_bnd.x,
                        n0_bnd.y,
                        n1_bnd.y,
                    );
                    let bnd_values_x1_perturbed = Self::compute_boundary_value_by_interpolation(
                        &nodal_coeffs,
                        self.basis().basis1d.n,
                        &basis.basis1d.xi,
                        &self.basis().basis1d.inv_vandermonde,
                        n0.x,
                        n1.x + 1e-6,
                        n0.y,
                        n1.y,
                        n0_bnd.x,
                        n1_bnd.x,
                        n0_bnd.y,
                        n1_bnd.y,
                    );
                    let bnd_values_y0_perturbed = Self::compute_boundary_value_by_interpolation(
                        &nodal_coeffs,
                        self.basis().basis1d.n,
                        &basis.basis1d.xi,
                        &self.basis().basis1d.inv_vandermonde,
                        n0.x,
                        n1.x,
                        n0.y + 1e-6,
                        n1.y,
                        n0_bnd.x,
                        n1_bnd.x,
                        n0_bnd.y,
                        n1_bnd.y,
                    );
                    let bnd_values_y1_perturbed = Self::compute_boundary_value_by_interpolation(
                        &nodal_coeffs,
                        self.basis().basis1d.n,
                        &basis.basis1d.xi,
                        &self.basis().basis1d.inv_vandermonde,
                        n0.x,
                        n1.x,
                        n0.y,
                        n1.y + 1e-6,
                        n0_bnd.x,
                        n1_bnd.x,
                        n0_bnd.y,
                        n1_bnd.y,
                    );
                    let dbnd_values_dx0 = (&bnd_values_x0_perturbed - &bnd_values) / 1e-6;
                    let dbnd_values_dx1 = (&bnd_values_x1_perturbed - &bnd_values) / 1e-6;
                    let dbnd_values_dy0 = (&bnd_values_y0_perturbed - &bnd_values) / 1e-6;
                    let dbnd_values_dy1 = (&bnd_values_y1_perturbed - &bnd_values) / 1e-6;

                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];
                        let (
                            boundary_flux,
                            dflux_du,
                            dflux_dbnd_value,
                            dflux_dx0,
                            dflux_dx1,
                            dflux_dy0,
                            dflux_dy1,
                        ): (f64, f64, f64, f64, f64, f64, f64) = self.dnum_flux(
                            sol_slice[i],
                            bnd_values[i],
                            x_slice[local_ids[0]],
                            x_slice[(local_ids[0] + 1) % 3],
                            y_slice[local_ids[0]],
                            y_slice[(local_ids[0] + 1) % 3],
                            1.0,
                        );
                        let mut dflux_dx = [0.0; 3];
                        let mut dflux_dy = [0.0; 3];
                        dflux_dx[local_ids[0]] = dflux_dx0 + dflux_dbnd_value * dbnd_values_dx0[i];
                        dflux_dx[(local_ids[0] + 1) % 3] =
                            dflux_dx1 + dflux_dbnd_value * dbnd_values_dx1[i];
                        dflux_dy[local_ids[0]] = dflux_dy0 + dflux_dbnd_value * dbnd_values_dy0[i];
                        dflux_dy[(local_ids[0] + 1) % 3] =
                            dflux_dy1 + dflux_dbnd_value * dbnd_values_dy1[i];

                        let mut dscaling_dx = [0.0; 3];
                        let mut dscaling_dy = [0.0; 3];
                        let scaling: f64 = self.dscaling(
                            xi,
                            eta,
                            ref_normal,
                            &x_slice,
                            dscaling_dx.as_mut_slice(),
                            &y_slice,
                            dscaling_dy.as_mut_slice(),
                            1.0,
                        );
                        let transformed_flux = boundary_flux * scaling;

                        let dtransformed_flux_du = dflux_du * scaling;
                        let dtransformed_flux_dx = &(&ArrayView1::from(&dflux_dx) * scaling)
                            + &(&ArrayView1::from(&dscaling_dx) * boundary_flux);
                        let dtransformed_flux_dy = &(&ArrayView1::from(&dflux_dy) * scaling)
                            + &(&ArrayView1::from(&dscaling_dy) * boundary_flux);

                        let itest_func = basis.nodes_along_edges[(local_ids[0], i)];

                        residuals[(ielem, itest_func)] +=
                            edge_length * edge_weights[i] * transformed_flux;

                        let row_idx = ielem * ncell_basis + itest_func;
                        for j in 0..3 {
                            dx[(ielem, itest_func, inodes[j])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_dx[j];
                            dy[(ielem, itest_func, inodes[j])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_dy[j];
                        }

                        if is_enriched {
                            for (j, &isol_node) in sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .indexed_iter()
                            {
                                let col_idx = ielem * unenriched_ncell_basis + isol_node;
                                dsol[(ielem, itest_func, ielem, isol_node)] += 0.5
                                    * edge_length
                                    * edge_weights[i]
                                    * self.interp_node_to_enriched_quadrature()[(i, j)]
                                    * dtransformed_flux_du;
                            }
                        } else {
                            let sol_nodes = sol_nodes_along_edges.slice(s![local_ids[0], ..]);
                            let col_idx = ielem * ncell_basis + sol_nodes[i];
                            dsol[(ielem, itest_func, ielem, sol_nodes[i])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_du;
                        }
                    }
                }
            }
        }
        */
        // Open boundaries
        for bnd in &mesh.boundaries.open {
            let iedges = &bnd.iedges;
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let inodes = &elem.inodes;
                    let x_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                    let y_slice: [f64; 3] =
                        std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_ids = &edge.local_ids;
                    let ref_normal = Self::compute_ref_normal(local_ids[0]);
                    let edge_length = Self::compute_ref_edge_length(local_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.r.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.s.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    for i in 0..nedge_basis {
                        let u = sol_slice[i];
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];

                        let (boundary_flux, dflux_du, dflux_dx0, dflux_dx1, dflux_dy0, dflux_dy1): (
                            f64,
                            f64,
                            f64,
                            f64,
                            f64,
                            f64,
                        ) = self.dopen_bnd_flux(
                            u,
                            x_slice[local_ids[0]],
                            x_slice[(local_ids[0] + 1) % 3],
                            y_slice[local_ids[0]],
                            y_slice[(local_ids[0] + 1) % 3],
                            1.0,
                        );

                        let mut dflux_dx = [0.0; 3];
                        let mut dflux_dy = [0.0; 3];
                        dflux_dx[local_ids[0]] = dflux_dx0;
                        dflux_dx[(local_ids[0] + 1) % 3] = dflux_dx1;
                        dflux_dy[local_ids[0]] = dflux_dy0;
                        dflux_dy[(local_ids[0] + 1) % 3] = dflux_dy1;

                        let mut dscaling_dx = [0.0; 3];
                        let mut dscaling_dy = [0.0; 3];
                        let scaling: f64 = self.dscaling(
                            xi,
                            eta,
                            ref_normal,
                            &x_slice,
                            dscaling_dx.as_mut_slice(),
                            &y_slice,
                            dscaling_dy.as_mut_slice(),
                            1.0,
                        );
                        let transformed_flux = boundary_flux * scaling;

                        let dtransformed_flux_du = dflux_du * scaling;
                        let dtransformed_flux_dx = &(&ArrayView1::from(&dflux_dx) * scaling)
                            + &(&ArrayView1::from(&dscaling_dx) * boundary_flux);
                        let dtransformed_flux_dy = &(&ArrayView1::from(&dflux_dy) * scaling)
                            + &(&ArrayView1::from(&dscaling_dy) * boundary_flux);

                        let itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                        if ielem == 1 {
                            println!("boundary_flux: {:?}", boundary_flux);
                            println!("transformed_flux: {:?}", transformed_flux);
                        }
                        residuals[(ielem, itest_func)] +=
                            edge_length * edge_weights[i] * transformed_flux;

                        for j in 0..3 {
                            dx[(ielem, itest_func, inodes[j])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_dx[j];
                            dy[(ielem, itest_func, inodes[j])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_dy[j];
                        }

                        if is_enriched {
                            for (j, &isol_node) in sol_nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .indexed_iter()
                            {
                                dsol[(ielem, itest_func, ielem, isol_node)] += edge_length
                                    * edge_weights[i]
                                    * self.interp_node_to_enriched_quadrature()[(i, j)]
                                    * dtransformed_flux_du;
                            }
                        } else {
                            let sol_nodes = sol_nodes_along_edges.slice(s![local_ids[0], ..]);
                            dsol[(ielem, itest_func, ielem, sol_nodes[i])] +=
                                edge_length * edge_weights[i] * dtransformed_flux_du;
                        }
                    }
                }
            }
        }
    }
    fn compute_mesh_distortion_deviation(
        &self,
        res_mesh: &mut Array1<f64>,
        dres_mesh_dx: &mut Array2<f64>,
        dres_mesh_dy: &mut Array2<f64>,
        mesh: &Mesh2d<TriangleElement>,
    ) {
        let ref_x = [0.0, 1.0, 0.0];
        let ref_y = [0.0, 0.0, 1.0];
        let ref_distortion = Self::compute_distortion(&ref_x, &ref_y, &self.basis());
        for (elem_idx, element) in mesh.elements.iter().enumerate() {
            if let Status::Active(element) = element {
                let inodes = &element.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].as_ref().y);
                let distortion = Self::compute_distortion(&x_slice, &y_slice, &self.basis());
                res_mesh[elem_idx] = distortion - ref_distortion;

                let mut distortion_x_perturbed = [0.0; 3];
                let mut distortion_y_perturbed = [0.0; 3];
                distortion_x_perturbed[0] = {
                    let mut x_slice_perturbed = x_slice;
                    x_slice_perturbed[0] += 1e-6;
                    Self::compute_distortion(&x_slice_perturbed, &y_slice, &self.basis())
                };
                distortion_x_perturbed[1] = {
                    let mut x_slice_perturbed = x_slice;
                    x_slice_perturbed[1] += 1e-6;
                    Self::compute_distortion(&x_slice_perturbed, &y_slice, &self.basis())
                };
                distortion_x_perturbed[2] = {
                    let mut x_slice_perturbed = x_slice;
                    x_slice_perturbed[2] += 1e-6;
                    Self::compute_distortion(&x_slice_perturbed, &y_slice, &self.basis())
                };
                distortion_y_perturbed[0] = {
                    let mut y_slice_perturbed = y_slice;
                    y_slice_perturbed[0] += 1e-6;
                    Self::compute_distortion(&x_slice, &y_slice_perturbed, &self.basis())
                };
                distortion_y_perturbed[1] = {
                    let mut y_slice_perturbed = y_slice;
                    y_slice_perturbed[1] += 1e-6;
                    Self::compute_distortion(&x_slice, &y_slice_perturbed, &self.basis())
                };
                distortion_y_perturbed[2] = {
                    let mut y_slice_perturbed = y_slice;
                    y_slice_perturbed[2] += 1e-6;
                    Self::compute_distortion(&x_slice, &y_slice_perturbed, &self.basis())
                };
                for j in 0..3 {
                    dres_mesh_dx[(elem_idx, inodes[j])] =
                        (distortion_x_perturbed[j] - distortion) / 1e-6;
                    dres_mesh_dy[(elem_idx, inodes[j])] =
                        (distortion_y_perturbed[j] - distortion) / 1e-6;
                }
            }
        }
    }
    fn volume_integral(
        &self,
        basis: &TriangleBasis,
        itest_func: usize,
        sol: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> f64;
    fn dvolume(
        &self,
        basis: &TriangleBasis,
        itest_func: usize,
        sol: &[f64],
        d_sol: &mut [f64],
        x: &[f64],
        d_x: &mut [f64],
        y: &[f64],
        d_y: &mut [f64],
        d_retval: f64,
    ) -> f64;
    fn compute_boundary_flux(&self, u: f64, ub: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64;
    fn dbnd_flux(
        &self,
        u: f64,
        ub: f64,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        d_retval: f64,
    ) -> (f64, f64, f64, f64, f64, f64);
    fn compute_numerical_flux(&self, ul: f64, ur: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64;
    fn dnum_flux(
        &self,
        ul: f64,
        ur: f64,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        d_retval: f64,
    ) -> (f64, f64, f64, f64, f64, f64, f64);
    fn compute_flux_scaling(
        &self,
        xi: f64,
        eta: f64,
        ref_normal: [f64; 2],
        x: &[f64],
        y: &[f64],
    ) -> f64;
    fn dscaling(
        &self,
        xi: f64,
        eta: f64,
        ref_normal: [f64; 2],
        x: &[f64],
        d_x: &mut [f64],
        y: &[f64],
        d_y: &mut [f64],
        d_retval: f64,
    ) -> f64;
    fn compute_open_boundary_flux(&self, u: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64;
    fn dopen_bnd_flux(
        &self,
        u: f64,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        d_retval: f64,
    ) -> (f64, f64, f64, f64, f64, f64);
    fn compute_interior_flux(
        &self,
        xi: f64,
        eta: f64,
        ref_normal: [f64; 2],
        u: f64,
        x: &[f64],
        y: &[f64],
    ) -> f64;
    fn dinterior_flux(
        &self,
        xi: f64,
        eta: f64,
        ref_normal: [f64; 2],
        u: f64,
        x: &[f64],
        d_x: &mut [f64],
        y: &[f64],
        d_y: &mut [f64],
        d_retval: f64,
    ) -> (f64, f64);
    fn physical_flux(&self, u: f64) -> [f64; 2];
    fn initialize_solution(&self, solutions: ArrayViewMut2<f64>);
    fn set_initial_solution(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        initial_guess: ArrayView2<f64>,
    ) -> Array2<f64> {
        let ncell_basis = self.basis().r.len();
        let nelem = mesh.elem_num;
        let mut solutions = Array2::zeros((nelem, ncell_basis));

        for ielem in 0..nelem {
            let p0_val = initial_guess[[ielem, 0]];
            // For a nodal basis, the P0 solution is represented by setting all nodal
            // values within the element to the same constant value.
            solutions.slice_mut(s![ielem, ..]).fill(p0_val);
        }
        solutions
    }
    fn get_solutions_along_boundary(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        iedges: &Vec<usize>,
        solutions: &Array2<f64>,
    ) -> Array2<f64> {
        let sol_nodes_along_edges = &self.basis().nodes_along_edges;
        let n_bnd_edges = iedges.len();
        let nedge_basis = sol_nodes_along_edges.shape()[1];

        let mut bnd_solutions = Array2::zeros((n_bnd_edges, nedge_basis));

        for (i, &iedge) in iedges.iter().enumerate() {
            let edge = mesh.edges[iedge].as_ref();
            let ielem = edge.parents[0];
            let local_edge_id = edge.local_ids[0];
            let element_sols = solutions.slice(s![ielem, ..]);
            let dof_indices_on_edge = sol_nodes_along_edges.slice(s![local_edge_id, ..]);

            let sols_on_edge =
                element_sols.select(Axis(0), dof_indices_on_edge.as_slice().unwrap());

            bnd_solutions.row_mut(i).assign(&sols_on_edge);
        }
        bnd_solutions
    }
    fn compute_boundary_value_by_interpolation(
        source_solution: &Array1<f64>,
        n: usize,                      // order of the source basis
        xi: &Array1<f64>,              // must be mapped to [0, 1] before passing in
        inv_vandermonde: &Array2<f64>, // inverse of the vandermonde matrix of the source basis
        x0_edge: f64,
        x1_edge: f64,
        y0_edge: f64,
        y1_edge: f64,
        x0_bnd: f64,
        x1_bnd: f64,
        y0_bnd: f64,
        y1_bnd: f64,
    ) -> Array1<f64> {
        // Compute reference coordinates of the edge's ends on the boundary
        let l_bnd_sq = (x1_bnd - x0_bnd).powi(2) + (y1_bnd - y0_bnd).powi(2);

        let l_bnd = l_bnd_sq.sqrt();

        // Parametric coordinate of the edge's start point on the boundary [0, 1]
        let t_bnd_start = ((x0_edge - x0_bnd) * (x1_bnd - x0_bnd)
            + (y0_edge - y0_bnd) * (y1_bnd - y0_bnd))
            / l_bnd_sq;

        let l_edge = ((x1_edge - x0_edge).powi(2) + (y1_edge - y0_edge).powi(2)).sqrt();

        // Determine relative orientation of the edge wrt the boundary
        let dot_product =
            (x1_edge - x0_edge) * (x1_bnd - x0_bnd) + (y1_edge - y0_edge) * (y1_bnd - y0_bnd);
        let orientation_sign = dot_product.signum();

        let mapped_coords = xi.mapv(|t_edge| {
            // t_edge is the parametric coordinate on the edge, in [0, 1]
            // Map it to the parametric coordinate on the boundary
            let t_bnd = t_bnd_start + t_edge * (l_edge / l_bnd) * orientation_sign;
            // Map from [0, 1] on boundary to [-1, 1] on reference element
            2.0 * t_bnd - 1.0
        });

        let v = LobattoBasis::vandermonde1d(n, mapped_coords.view());
        let interp_matrix = v.dot(inv_vandermonde);
        interp_matrix.dot(source_solution)
    }
}
pub trait SQP: P0Solver + SpaceTimeSolver1DScalar {
    fn compute_node_constraints(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        new_to_old: &Vec<usize>,
    ) -> Array2<f64> {
        let n_nodes = mesh.node_num;
        let total_dofs = 2 * n_nodes;

        let n_free_dofs = new_to_old.len();

        let mut constraint_matrix = Array2::zeros((total_dofs, n_free_dofs));

        for (new_idx, &old_idx) in new_to_old.iter().enumerate() {
            constraint_matrix[[old_idx, new_idx]] = 1.0;
        }

        constraint_matrix
    }
    #[allow(non_snake_case)]
    fn solve_linear_subproblem(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        node_constraints: ArrayView2<f64>,
        res: ArrayView2<f64>,
        hessian_uu: ArrayView2<f64>,
        hessian_ux: ArrayView2<f64>,
        hessian_xx: ArrayView2<f64>,
        dsol: ArrayView2<f64>,
        dcoord: ArrayView2<f64>,
        dobj_dsol: ArrayView1<f64>,
        dobj_dcoord: ArrayView1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let free_coords = &mesh.free_bnd_x.len() + &mesh.free_bnd_y.len();
        let interior_nnodes = mesh.interior_nodes.len();
        let num_u = dsol.shape()[1];
        let num_x: usize = free_coords + 2 * interior_nnodes;
        let num_lambda = num_u;
        let n_total = num_u + num_x + num_lambda;

        let mut A_ndarray = Array2::<f64>::zeros((n_total, n_total));
        let mut b_ndarray = Array1::<f64>::zeros(n_total);

        A_ndarray
            .slice_mut(s![0..num_u, 0..num_u])
            .assign(&hessian_uu);
        A_ndarray
            .slice_mut(s![0..num_u, num_u..num_u + num_x])
            .assign(&hessian_ux);
        A_ndarray
            .slice_mut(s![0..num_u, num_u + num_x..n_total])
            .assign(&dsol.t());

        A_ndarray
            .slice_mut(s![num_u..num_u + num_x, 0..num_u])
            .assign(&hessian_ux.t());
        A_ndarray
            .slice_mut(s![num_u..num_u + num_x, num_u..num_u + num_x])
            .assign(&hessian_xx);
        A_ndarray
            .slice_mut(s![num_u..num_u + num_x, num_u + num_x..n_total])
            .assign(&dcoord.dot(&node_constraints).t());

        A_ndarray
            .slice_mut(s![num_u + num_x..n_total, 0..num_u])
            .assign(&dsol);
        A_ndarray
            .slice_mut(s![num_u + num_x..n_total, num_u..num_u + num_x])
            .assign(&dcoord.dot(&node_constraints));

        b_ndarray
            .slice_mut(s![0..num_u])
            .assign(&(&dobj_dsol * -1.0));
        b_ndarray
            .slice_mut(s![num_u..num_u + num_x])
            .assign(&(&dobj_dcoord * -1.0));
        b_ndarray
            .slice_mut(s![num_u + num_x..n_total])
            .assign(&(res.flatten() * -1.0));

        let A = A_ndarray.view().into_faer();
        let b = Col::<f64>::from_iter(b_ndarray.view().iter().copied());
        let flu = A.partial_piv_lu();
        let u_x_lambda = flu.solve(&b);

        let delta_u = u_x_lambda.subrows(0, num_u);
        let delta_x = u_x_lambda.subrows(num_u, num_x);

        let delta_u_ndarray = Array1::from_iter(delta_u.iter().copied());
        let delta_x_ndarray = Array1::from_iter(delta_x.iter().copied());
        (delta_u_ndarray, delta_x_ndarray)
    }
    fn update_solutions(
        &self,
        solutions: &mut Array2<f64>,
        new_to_old: &Vec<usize>,
        alpha: f64,
        delta_u: &Array1<f64>,
    ) {
        let delta_u_reshaped = delta_u
            .to_shape((new_to_old.len(), self.basis().r.len()))
            .unwrap();
        for (i, &new_idx) in new_to_old.iter().enumerate() {
            let delta_u_slice = delta_u_reshaped.row(i);
            solutions.row_mut(new_idx).scaled_add(alpha, &delta_u_slice);
        }
    }
    fn solve(&self, mesh: &mut Mesh2d<TriangleElement>, solutions: &mut Array2<f64>) {
        let initial_guess = self.compute_initial_guess(mesh);
        let initial_solutions = self.set_initial_solution(mesh, initial_guess.view());
        solutions.assign(&initial_solutions);
        let nelem = mesh.elem_num;
        let nnode = mesh.node_num;
        let mut new_to_old_elem: Vec<usize> = (0..nelem).collect();
        let ncell_basis = self.basis().r.len();
        let enriched_ncell_basis = self.enriched_basis().r.len();
        let epsilon1 = 1e-10;
        let epsilon2 = 1e-12;
        let mut gamma_k = 1e-2; // regularization parameter for hessian_xx
        let gamma_min = 1e-8;
        let k1 = 1e-2;
        let k2 = 1e-1;
        let sigma: f64 = 0.5;
        let max_line_search_iter = 20;
        let max_sqp_iter = 30;
        // let free_coords = &self.mesh.free_coords;
        // println!("free_coords: {:?}", free_coords);

        let mut residuals: Array2<f64> = Array2::zeros((nelem, ncell_basis));
        let mut dsol: Array4<f64> = Array4::zeros((nelem, ncell_basis, nelem, ncell_basis));
        let mut dx: Array3<f64> = Array3::zeros((nelem, ncell_basis, nnode));
        let mut dy: Array3<f64> = Array3::zeros((nelem, ncell_basis, nnode));
        let mut enriched_residuals: Array2<f64> = Array2::zeros((nelem, enriched_ncell_basis));
        let mut enriched_dsol: Array4<f64> =
            Array4::zeros((nelem, enriched_ncell_basis, nelem, ncell_basis));
        let mut enriched_dx: Array3<f64> = Array3::zeros((nelem, enriched_ncell_basis, nnode));
        let mut enriched_dy: Array3<f64> = Array3::zeros((nelem, enriched_ncell_basis, nnode));

        let mut iter: usize = 0;
        while iter < max_sqp_iter {
            println!("iter: {:?}", iter);
            // reset residuals, dsol, dx, enriched_residuals, enriched_dsol, enriched_dx
            residuals.fill(0.0);
            dsol.fill(0.0);
            dx.fill(0.0);
            dy.fill(0.0);
            enriched_residuals.fill(0.0);
            enriched_dsol.fill(0.0);
            enriched_dx.fill(0.0);
            enriched_dy.fill(0.0);

            // mesh.print_free_node_coords();
            write_average(solutions, mesh, &self.basis(), iter);
            write_nodal_solutions(&solutions, &mesh, &self.basis(), iter);
            mesh.collapse_small_elements(0.2, &mut new_to_old_elem);

            dbg!(&mesh.nodes[2].as_ref().parents);

            let new_to_old_node = mesh.rearrange_node_dofs();
            let node_constraints = self.compute_node_constraints(mesh, &new_to_old_node);

            self.compute_residuals_and_derivatives(
                mesh,
                solutions.view(),
                residuals.view_mut(),
                dsol.view_mut(),
                dx.view_mut(),
                dy.view_mut(),
                false,
                iter,
            );

            let res_active = residuals.select(Axis(0), &new_to_old_elem);
            let dsol_active = dsol
                .select(Axis(0), &new_to_old_elem)
                .select(Axis(2), &new_to_old_elem)
                .to_shape((
                    new_to_old_elem.len() * ncell_basis,
                    new_to_old_elem.len() * ncell_basis,
                ))
                .unwrap()
                .to_owned();
            let dx_active = dx
                .select(Axis(0), &new_to_old_elem)
                .to_shape((new_to_old_elem.len() * ncell_basis, nnode))
                .unwrap()
                .to_owned();
            let dy_active = dy
                .select(Axis(0), &new_to_old_elem)
                .to_shape((new_to_old_elem.len() * ncell_basis, nnode))
                .unwrap()
                .to_owned();

            self.compute_residuals_and_derivatives(
                mesh,
                solutions.view(),
                enriched_residuals.view_mut(),
                enriched_dsol.view_mut(),
                enriched_dx.view_mut(),
                enriched_dy.view_mut(),
                true,
                iter,
            );
            let enr_res_active = enriched_residuals.select(Axis(0), &new_to_old_elem);
            let enr_dsol_active = enriched_dsol
                .select(Axis(0), &new_to_old_elem)
                .select(Axis(2), &new_to_old_elem)
                .to_shape((
                    new_to_old_elem.len() * enriched_ncell_basis,
                    new_to_old_elem.len() * ncell_basis,
                ))
                .unwrap()
                .to_owned();
            let enr_dx_active = enriched_dx
                .select(Axis(0), &new_to_old_elem)
                .to_shape((new_to_old_elem.len() * enriched_ncell_basis, nnode))
                .unwrap()
                .to_owned();
            let enr_dy_active = enriched_dy
                .select(Axis(0), &new_to_old_elem)
                .to_shape((new_to_old_elem.len() * enriched_ncell_basis, nnode))
                .unwrap()
                .to_owned();
            /*
            {
                let mut perturbed_solutions = solutions.to_owned();
                perturbed_solutions[(0, 1)] += 1e-6;
                let mut perturbed_residuals = Array2::<f64>::zeros((nelem, enriched_ncell_basis));
                self.compute_residuals(
                    mesh,
                    perturbed_solutions.view(),
                    perturbed_residuals.view_mut(),
                    true,
                );
                let dres_dsol = (&perturbed_residuals - &enriched_residuals) / 1e-6;
                println!("dres_dsol by FD: {:?}", dres_dsol);
                println!("dsol: {:?}", enriched_dsol.slice(s![.., 1]));
            }
            {
                let mut perturbed_mesh = mesh.clone();
                perturbed_mesh.nodes[1].x += 1e-6;
                let mut perturbed_residuals = Array2::<f64>::zeros((nelem, enriched_ncell_basis));
                self.compute_residuals(
                    &perturbed_mesh,
                    solutions.view(),
                    perturbed_residuals.view_mut(),
                    true,
                );
                let dres_dx = (&perturbed_residuals - &enriched_residuals) / 1e-6;
                println!("dres_dx by FD: {:?}", dres_dx);
                println!("enriched_dx: {:?}", enriched_dx.slice(s![.., 1]));
            }
            */
            let dcoord = concatenate(Axis(1), &[dx_active.view(), dy_active.view()]).unwrap();
            let dobj_dsol = enr_dsol_active.t().dot(&enr_res_active.flatten());
            let enriched_dcoord =
                concatenate(Axis(1), &[enr_dx_active.view(), enr_dy_active.view()]).unwrap();
            let dobj_dcoord = enriched_dcoord
                .t()
                .dot(&enr_res_active.flatten())
                .dot(&node_constraints);
            let dsol_inv = dsol_active.inv().unwrap();
            let dsol_inv_t = dsol_inv.t();
            let dobj_dsol_t = dobj_dsol.t();
            let lambda_hat = dsol_inv_t.dot(&dobj_dsol_t);
            let mu = lambda_hat.mapv(f64::abs).max().copied().unwrap() * 2.0;
            // termination criteria
            let optimality = &dobj_dcoord.t()
                - &dcoord
                    .dot(&node_constraints)
                    .t()
                    .dot(&dsol_inv_t)
                    .dot(&dobj_dsol.t());
            let optimality_norm = optimality.mapv(|x| x.powi(2)).sum().sqrt();
            let feasibility_norm = res_active.mapv(|x| x.powi(2)).sum().sqrt();
            println!("optimality: {:?}", optimality_norm);
            println!("feasibility: {:?}", feasibility_norm);
            if optimality_norm < epsilon1 && feasibility_norm < epsilon2 {
                println!("Terminating SQP at iter: {:?}", iter);
                break;
            }
            let hessian_uu = enr_dsol_active.t().dot(&enr_dsol_active);
            let hessian_ux = enr_dsol_active
                .t()
                .dot(&enriched_dcoord.dot(&node_constraints));
            let mut hessian_xx = enriched_dcoord
                .dot(&node_constraints)
                .t()
                .dot(&enriched_dcoord.dot(&node_constraints));
            println!("gamma_k: {:?}", gamma_k);
            hessian_xx += &(gamma_k * &Array2::eye(hessian_xx.shape()[0]));

            let (delta_u, delta_x) = self.solve_linear_subproblem(
                mesh,
                node_constraints.view(),
                res_active.view(),
                hessian_uu.view(),
                hessian_ux.view(),
                hessian_xx.view(),
                dsol_active.view(),
                dcoord.view(),
                dobj_dsol.view(),
                dobj_dcoord.view(),
            );
            println!("linear subproblem solved");
            // backtracking line search
            let merit_func = |alpha: f64| -> f64 {
                let mut tmp_mesh = mesh.clone();
                let mut u = solutions.clone();
                self.update_solutions(&mut u, &new_to_old_elem, alpha, &delta_u);
                /*
                let delta_u_ndarray = Array::from_iter(delta_u.iter().copied());
                let u_flat = &solutions.flatten() + alpha * &delta_u_ndarray;
                let u = u_flat.to_shape((nelem, ncell_basis)).unwrap();
                */
                tmp_mesh.update_node_coords(&new_to_old_node, alpha, delta_x.view());
                let mut tmp_res = Array2::zeros((nelem, ncell_basis));
                self.compute_residuals(&tmp_mesh, u.view(), tmp_res.view_mut(), false);
                let mut tmp_enr_res = Array2::zeros((nelem, enriched_ncell_basis));
                self.compute_residuals(&tmp_mesh, u.view(), tmp_enr_res.view_mut(), true);
                let f = 0.5 * &tmp_enr_res.flatten().dot(&tmp_enr_res.flatten());
                let l1_norm = tmp_res.mapv(f64::abs).sum();

                f + mu * l1_norm
            };
            let merit_func_0 = merit_func(0.0);

            let dir_deriv = dobj_dsol.dot(&delta_u) + dobj_dcoord.dot(&delta_x)
                - mu * residuals.mapv(f64::abs).sum();
            let c: f64 = 1e-4;
            let tau: f64 = 0.5;
            let mut n: i32 = 1;
            let mut alpha: f64 = tau.powi(n - 1);
            let mut line_search_iter: usize = 0;
            while line_search_iter < max_line_search_iter {
                if merit_func(alpha) <= merit_func_0 + c * alpha * dir_deriv {
                    break;
                }
                alpha *= tau;
                n += 1;
                line_search_iter += 1;
            }
            if line_search_iter == max_line_search_iter {
                panic!(
                    "Warning: Line search did not converge within {} iterations.",
                    max_line_search_iter
                );
            }
            println!("line search done");

            // update gamma_k
            let delta_x_norm = delta_x.mapv(|x| x.powi(2)).sum().sqrt();
            let gamma_k_bar = {
                if delta_x_norm < k1 {
                    sigma * gamma_k
                } else if delta_x_norm > k2 {
                    1.0 / sigma * gamma_k
                } else {
                    gamma_k
                }
            };
            gamma_k = gamma_k_bar.max(gamma_min);

            //solutions.scaled_add(alpha, &delta_u.to_shape(solutions.shape()).unwrap());
            self.update_solutions(solutions, &new_to_old_elem, alpha, &delta_u);
            mesh.update_node_coords(&new_to_old_node, alpha, delta_x.view());

            iter += 1;
        }
        write_average(solutions, mesh, &self.basis(), iter);
        write_nodal_solutions(&solutions, &mesh, &self.basis(), iter);
        /*
        println!("enriched_dsol[.., 6]: {:?}", enriched_dsol.slice(s![.., 6]));
        {
            println!("=== Computing finite difference for enriched_dsol[.., 6] ===");
            let epsilon = 1e-7;
            let isol_dof_to_check = 6;
            let ncell_basis = self.basis.r.len();
            let ielem_to_check = isol_dof_to_check / ncell_basis;
            let idof_to_check = isol_dof_to_check % ncell_basis;

            let base_residuals = enriched_residuals.clone();

            let mut perturbed_solutions = solutions.to_owned();
            perturbed_solutions[(ielem_to_check, idof_to_check)] += epsilon;

            let mut perturbed_residuals: Array2<f64> = Array2::zeros((nelem, enriched_ncell_basis));
            self.compute_residuals(
                &self.mesh,
                perturbed_solutions.view(),
                perturbed_residuals.view_mut(),
                true,
            );

            let fd_dsol = (perturbed_residuals
                .into_shape(nelem * enriched_ncell_basis)
                .unwrap()
                - base_residuals
                    .into_shape(nelem * enriched_ncell_basis)
                    .unwrap())
                / epsilon;
            println!("FD enriched_dsol[.., 6]: {:?}", fd_dsol);
        }
        println!("enriched_dx[.., 4]: {:?}", enriched_dx.slice(s![.., 4]));
        {
            println!("=== Computing finite difference for enriched_dx[.., 4] ===");
            let epsilon = 1e-7;
            let inode_to_check = 4;
            let base_residuals = enriched_residuals.clone();
            let original_x = self.mesh.nodes[inode_to_check].x;
            self.mesh.nodes[inode_to_check].x += epsilon;
            let mut perturbed_residuals: Array2<f64> = Array2::zeros((nelem, enriched_ncell_basis));
            self.compute_residuals(
                &self.mesh,
                solutions.view(),
                perturbed_residuals.view_mut(),
                true,
            );
            self.mesh.nodes[inode_to_check].x = original_x;
            let fd_dx = (perturbed_residuals
                .into_shape(nelem * enriched_ncell_basis)
                .unwrap()
                - base_residuals
                    .into_shape(nelem * enriched_ncell_basis)
                    .unwrap())
                / epsilon;
            println!("FD enriched_dx[.., 4]: {:?}", fd_dx);
        }
        */
    }
}

pub fn compute_source_value_nalgebra(
    source_solution: &DVector<f64>,
    n: usize,
    xi: f64, // must be mapped to [0, 1] before passing in
    inv_vandermonde: &DMatrix<f64>,
    x0: f64,
    x1: f64,
) -> f64 {
    let coord = xi * (x1 - x0) + x0;
    let coord_mapped = dvector![coord * 2.0 - 1.0];
    let v = LobattoBasis::vandermonde1d_nalgebra(n, &coord_mapped);
    let interp_matrix = v * inv_vandermonde;
    let values = interp_matrix * source_solution;
    values[0]
}
pub fn print_solutions(solutions: &Array2<f64>, new_to_old_elem: &Array1<usize>) {
    println!("Solutions:");
    for &i in new_to_old_elem {
        let row_str: Vec<String> = solutions
            .row(i)
            .iter()
            .map(|&val| format!("{:.4}", val))
            .collect();
        println!("[{}]", row_str.join(", "));
    }
}
