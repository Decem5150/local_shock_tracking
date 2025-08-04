use ndarray::{
    Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3, ArrayViewMut4, Axis, Zip,
    s,
};

use crate::disc::{
    dg_basis::{Basis1D, Basis2D, lagrange1d::LobattoBasis, triangle::TriangleBasis},
    finite_difference::{
        FiniteDifference, compute_flux_scaling_derivatives, compute_numerical_flux_derivatives,
        compute_volume_derivatives,
    },
    geometric::Geometric2D,
    mesh::mesh2d::{Mesh2d, Status, TriangleElement},
};

pub trait SpaceTime1DScalar: Geometric2D {
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
        v.dot(&inv_vandermonde)
    }
    fn compute_interp_matrix_2d(
        n: usize,
        inv_vandermonde: ArrayView2<f64>,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
    ) -> Array2<f64> {
        let v = TriangleBasis::vandermonde2d(n, r, s);
        v.dot(&inv_vandermonde)
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
        let ncell_basis = basis.xi.len();
        let nedge_basis = basis.quad_p.len();
        let edge_weights = &basis.quad_w;
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let inodes = &elem.inodes;
                let phys_x: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                let phys_y: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
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
                        interp_sol.as_slice().unwrap(),
                        &phys_x,
                        &phys_y,
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

                let left_phys_x: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[left_inodes[i]].as_ref().x);
                let left_phys_y: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[left_inodes[i]].as_ref().y);
                let right_phys_x: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[right_inodes[i]].as_ref().x);
                let right_phys_y: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[right_inodes[i]].as_ref().y);

                let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_edge_ids = &edge.local_ids;
                let local_node_ids = [local_edge_ids[0], (local_edge_ids[0] + 1) % 3];
                let left_cano_normal = Self::compute_canonical_normal(local_edge_ids[0]);
                let right_cano_normal = Self::compute_canonical_normal(local_edge_ids[1]);
                let left_cano_length = Self::compute_canonical_edge_length(local_edge_ids[0]);
                let right_cano_length = Self::compute_canonical_edge_length(local_edge_ids[1]);
                let (left_sol_slice, right_sol_slice) = {
                    let left_sol_slice = solutions.slice(s![ilelem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let right_sol_slice = solutions.slice(s![irelem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_edge_ids[1], ..])
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
                let left_xi_slice = basis.xi.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_edge_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
                let left_eta_slice = basis.eta.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_edge_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
                let right_xi_slice = basis.xi.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_edge_ids[1], ..])
                        .as_slice()
                        .unwrap(),
                );
                let right_eta_slice = basis.eta.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_edge_ids[1], ..])
                        .as_slice()
                        .unwrap(),
                );
                for i in 0..nedge_basis {
                    let left_value = left_sol_slice[i];
                    let right_value = right_sol_slice[nedge_basis - 1 - i];
                    let num_flux = self.compute_smoothed_numerical_flux(
                        left_value,
                        right_value,
                        left_phys_x[local_node_ids[0]],
                        left_phys_x[local_node_ids[1]],
                        left_phys_y[local_node_ids[0]],
                        left_phys_y[local_node_ids[1]],
                    );
                    let left_scaling = self.compute_flux_scaling(
                        left_xi_slice[i],
                        left_eta_slice[i],
                        left_cano_normal,
                        &left_phys_x,
                        &left_phys_y,
                    );
                    let right_scaling = self.compute_flux_scaling(
                        right_xi_slice[nedge_basis - 1 - i],
                        right_eta_slice[nedge_basis - 1 - i],
                        right_cano_normal,
                        &right_phys_x,
                        &right_phys_y,
                    );
                    let left_transformed_flux = num_flux * left_scaling;
                    let right_transformed_flux = -num_flux * right_scaling;
                    let left_itest_func = basis.nodes_along_edges[(local_edge_ids[0], i)];
                    let right_itest_func =
                        basis.nodes_along_edges[(local_edge_ids[1], nedge_basis - 1 - i)];
                    residuals[(ilelem, left_itest_func)] +=
                        left_cano_length * edge_weights[i] * left_transformed_flux;
                    residuals[(irelem, right_itest_func)] +=
                        right_cano_length * edge_weights[i] * right_transformed_flux;
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
                    let phys_x: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                    let phys_y: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_edge_ids = &edge.local_ids;
                    let local_node_ids = [local_edge_ids[0], (local_edge_ids[0] + 1) % 3];
                    let cano_normal = Self::compute_canonical_normal(local_edge_ids[0]);
                    let cano_length = Self::compute_canonical_edge_length(local_edge_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.xi.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.eta.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];
                        let boundary_flux = self.compute_smoothed_boundary_flux(
                            sol_slice[i],
                            bnd_value,
                            phys_x[local_node_ids[0]],
                            phys_x[local_node_ids[1]],
                            phys_y[local_node_ids[0]],
                            phys_y[local_node_ids[1]],
                        );
                        let scaling =
                            self.compute_flux_scaling(xi, eta, cano_normal, &phys_x, &phys_y);
                        let transformed_flux = boundary_flux * scaling;
                        let itest_func = basis.nodes_along_edges[(local_edge_ids[0], i)];
                        residuals[(ielem, itest_func)] +=
                            cano_length * edge_weights[i] * transformed_flux;
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
                    let phys_x: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                    let phys_y: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_edge_ids = &edge.local_ids;
                    let local_node_ids = [local_edge_ids[0], (local_edge_ids[0] + 1) % 3];
                    let cano_normal = Self::compute_canonical_normal(local_edge_ids[0]);
                    let cano_length = Self::compute_canonical_edge_length(local_edge_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.xi.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.eta.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );

                    // Compute the physical coordinates of the quadrature points on the edge
                    let n1 = 1.0 - &xi_slice - &eta_slice;
                    let n2 = xi_slice.to_owned();
                    let n3 = eta_slice.to_owned();

                    let x_phys = &n1 * phys_x[0] + &n2 * phys_x[1] + &n3 * phys_x[2];
                    let y_phys = &n1 * phys_y[0] + &n2 * phys_y[1] + &n3 * phys_y[2];

                    let bnd_values = Zip::from(&x_phys)
                        .and(&y_phys)
                        .map_collect(|&x, &y| func(x, y));

                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];

                        let boundary_flux = self.compute_smoothed_boundary_flux(
                            sol_slice[i],
                            bnd_values[i],
                            phys_x[local_edge_ids[0]],
                            phys_x[local_node_ids[1]],
                            phys_y[local_node_ids[0]],
                            phys_y[local_node_ids[1]],
                        );
                        let scaling =
                            self.compute_flux_scaling(xi, eta, cano_normal, &phys_x, &phys_y);
                        let transformed_flux = boundary_flux * scaling;
                        let itest_func = basis.nodes_along_edges[(local_edge_ids[0], i)];
                        residuals[(ielem, itest_func)] +=
                            cano_length * edge_weights[i] * transformed_flux;
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
                    let phys_x: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                    let phys_y: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_edge_ids = &edge.local_ids;
                    let local_node_ids = [local_edge_ids[0], (local_edge_ids[0] + 1) % 3];
                    let cano_normal = Self::compute_canonical_normal(local_edge_ids[0]);
                    let cano_length = Self::compute_canonical_edge_length(local_edge_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.xi.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.eta.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    for i in 0..nedge_basis {
                        let u = sol_slice[i];
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];
                        let boundary_flux = self.compute_open_boundary_flux(
                            u,
                            phys_x[local_node_ids[0]],
                            phys_x[local_node_ids[1]],
                            phys_y[local_node_ids[0]],
                            phys_y[local_node_ids[1]],
                        );
                        let scaling =
                            self.compute_flux_scaling(xi, eta, cano_normal, &phys_x, &phys_y);
                        let transformed_flux = boundary_flux * scaling;
                        let itest_func = basis.nodes_along_edges[(local_edge_ids[0], i)];
                        residuals[(ielem, itest_func)] +=
                            cano_length * edge_weights[i] * transformed_flux;
                    }
                }
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
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
        // Create finite difference calculator
        let fd = FiniteDifference::new();
        let basis = {
            if is_enriched {
                &self.enriched_basis()
            } else {
                &self.basis()
            }
        };
        let ncell_basis = basis.xi.len();
        let nedge_basis = basis.quad_p.len();
        let edge_weights = &basis.quad_w;
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let inodes = &elem.inodes;
                let phys_x: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                let phys_y: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                let interp_matrix = if is_enriched {
                    &self.interp_node_to_enriched_cubature()
                } else {
                    &self.interp_node_to_cubature()
                };
                let interp_sol = interp_matrix.dot(&solutions.slice(s![ielem, ..]));
                for itest_func in 0..ncell_basis {
                    let mut dvol_sol: Array1<f64> = Array1::zeros(basis.cub_xi.len());
                    let mut dvol_x: Array1<f64> = Array1::zeros(3);
                    let mut dvol_y: Array1<f64> = Array1::zeros(3);
                    // let res = self.dvolume(
                    //     basis,
                    //     itest_func,
                    //     &interp_sol.as_slice().unwrap(),
                    //     dvol_sol.as_slice_mut().unwrap(),
                    //     &phys_x,
                    //     dvol_x.as_slice_mut().unwrap(),
                    //     &phys_y,
                    //     dvol_y.as_slice_mut().unwrap(),
                    //     1.0,
                    // );

                    // Compute volume integral and its derivatives using finite difference
                    let fd = FiniteDifference::new();
                    let res = compute_volume_derivatives(
                        &fd,
                        |sol, x, y| self.volume_integral(basis, itest_func, sol, x, y),
                        interp_sol.as_slice().unwrap(),
                        &phys_x,
                        &phys_y,
                        dvol_sol.as_slice_mut().unwrap(),
                        dvol_x.as_slice_mut().unwrap(),
                        dvol_y.as_slice_mut().unwrap(),
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
                let left_phys_x: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[left_inodes[i]].as_ref().x);
                let left_phys_y: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[left_inodes[i]].as_ref().y);
                let right_phys_x: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[right_inodes[i]].as_ref().x);
                let right_phys_y: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[right_inodes[i]].as_ref().y);
                let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_edge_ids = &edge.local_ids;
                let left_local_node_ids = [local_edge_ids[0], (local_edge_ids[0] + 1) % 3];
                let right_local_node_ids = [local_edge_ids[1], (local_edge_ids[1] + 1) % 3];
                let left_cano_normal = Self::compute_canonical_normal(local_edge_ids[0]);
                let right_cano_normal = Self::compute_canonical_normal(local_edge_ids[1]);
                let left_cano_length = Self::compute_canonical_edge_length(local_edge_ids[0]);
                let right_cano_length = Self::compute_canonical_edge_length(local_edge_ids[1]);
                let (left_sol_slice, right_sol_slice) = {
                    let left_sol_slice = solutions.slice(s![ilelem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let right_sol_slice = solutions.slice(s![irelem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_edge_ids[1], ..])
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
                let left_xi_slice = basis.xi.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_edge_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
                let left_eta_slice = basis.eta.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_edge_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
                let right_xi_slice = basis.xi.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_edge_ids[1], ..])
                        .as_slice()
                        .unwrap(),
                );
                let right_eta_slice = basis.eta.select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_edge_ids[1], ..])
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
                        // ): (f64, f64, f64, f64, f64, f64, f64) = self.dnum_flux(
                        //     left_value,
                        //     right_value,
                        //     left_phys_x[left_local_node_ids[0]],
                        //     left_phys_x[left_local_node_ids[1]],
                        //     left_phys_y[left_local_node_ids[0]],
                        //     left_phys_y[left_local_node_ids[1]],
                        //     1.0,
                        // );
                    ): (f64, f64, f64, f64, f64, f64, f64) = {
                        let fd = FiniteDifference::new();
                        compute_numerical_flux_derivatives(
                            &fd,
                            |ul, ur, x0, x1, y0, y1| {
                                self.compute_smoothed_numerical_flux(ul, ur, x0, x1, y0, y1)
                            },
                            left_value,
                            right_value,
                            left_phys_x[left_local_node_ids[0]],
                            left_phys_x[left_local_node_ids[1]],
                            left_phys_y[left_local_node_ids[0]],
                            left_phys_y[left_local_node_ids[1]],
                        )
                    };
                    dflux_dleft_x[left_local_node_ids[0]] = dflux_dx0;
                    dflux_dleft_x[left_local_node_ids[1]] = dflux_dx1;
                    dflux_dleft_y[left_local_node_ids[0]] = dflux_dy0;
                    dflux_dleft_y[left_local_node_ids[1]] = dflux_dy1;
                    dflux_dright_x[right_local_node_ids[0]] = dflux_dx1;
                    dflux_dright_x[right_local_node_ids[1]] = dflux_dx0;
                    dflux_dright_y[right_local_node_ids[0]] = dflux_dy1;
                    dflux_dright_y[right_local_node_ids[1]] = dflux_dy0;

                    let mut dleft_scaling_dx = [0.0; 3];
                    let mut dleft_scaling_dy = [0.0; 3];
                    let mut dright_scaling_dx = [0.0; 3];
                    let mut dright_scaling_dy = [0.0; 3];
                    // let left_scaling: f64 = self.dscaling(
                    //     left_xi_slice[i],
                    //     left_eta_slice[i],
                    //     left_cano_normal,
                    //     left_phys_x.as_slice(),
                    //     dleft_scaling_dx.as_mut_slice(),
                    //     left_phys_y.as_slice(),
                    //     dleft_scaling_dy.as_mut_slice(),
                    //     1.0,
                    // );
                    let left_scaling: f64 = self.compute_flux_scaling(
                        left_xi_slice[i],
                        left_eta_slice[i],
                        left_cano_normal,
                        &left_phys_x,
                        &left_phys_y,
                    ); // TODO: Compute derivatives with finite difference
                    // let right_scaling: f64 = self.dscaling(
                    //     right_xi_slice[nedge_basis - 1 - i],
                    //     right_eta_slice[nedge_basis - 1 - i],
                    //     right_cano_normal,
                    //     right_phys_x.as_slice(),
                    //     dright_scaling_dx.as_mut_slice(),
                    //     right_phys_y.as_slice(),
                    //     dright_scaling_dy.as_mut_slice(),
                    //     1.0,
                    // );
                    let right_scaling: f64 = self.compute_flux_scaling(
                        right_xi_slice[nedge_basis - 1 - i],
                        right_eta_slice[nedge_basis - 1 - i],
                        right_cano_normal,
                        &right_phys_x,
                        &right_phys_y,
                    ); // TODO: Compute derivatives with finite difference

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

                    let left_itest_func = basis.nodes_along_edges[(local_edge_ids[0], i)];
                    let right_itest_func =
                        basis.nodes_along_edges[(local_edge_ids[1], nedge_basis - 1 - i)];

                    residuals[(ilelem, left_itest_func)] +=
                        left_cano_length * edge_weights[i] * left_transformed_flux;
                    residuals[(irelem, right_itest_func)] +=
                        right_cano_length * edge_weights[i] * right_transformed_flux;

                    if is_enriched {
                        // derivatives w.r.t. left value
                        for (j, &isol_node) in sol_nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .indexed_iter()
                        {
                            dsol[(ilelem, left_itest_func, ilelem, isol_node)] += left_cano_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()[(i, j)]
                                * dleft_transformed_flux_dul;
                            dsol[(irelem, right_itest_func, irelem, isol_node)] += right_cano_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()[(i, j)]
                                * dright_transformed_flux_dul;
                        }
                        // derivatives w.r.t. right value
                        for (j, &isol_node) in sol_nodes_along_edges
                            .slice(s![local_edge_ids[1], ..])
                            .indexed_iter()
                        {
                            dsol[(ilelem, left_itest_func, ilelem, isol_node)] += left_cano_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()
                                    [(nedge_basis - 1 - i, j)]
                                * dleft_transformed_flux_dur;
                            dsol[(irelem, right_itest_func, irelem, isol_node)] += right_cano_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()
                                    [(nedge_basis - 1 - i, j)]
                                * dright_transformed_flux_dur;
                        }
                    } else {
                        // derivatives w.r.t. left value
                        dsol[(ilelem, left_itest_func, ilelem, left_itest_func)] +=
                            left_cano_length * edge_weights[i] * dleft_transformed_flux_dul;
                        dsol[(irelem, right_itest_func, ilelem, left_itest_func)] +=
                            right_cano_length * edge_weights[i] * dright_transformed_flux_dul;
                        // derivatives w.r.t. right value
                        dsol[(ilelem, left_itest_func, irelem, right_itest_func)] +=
                            left_cano_length * edge_weights[i] * dleft_transformed_flux_dur;
                        dsol[(irelem, right_itest_func, irelem, right_itest_func)] +=
                            right_cano_length * edge_weights[i] * dright_transformed_flux_dur;
                    }
                    for j in 0..3 {
                        dx[(ilelem, left_itest_func, left_elem.inodes[j])] +=
                            left_cano_length * edge_weights[i] * dleft_transformed_flux_dx[j];
                        dy[(ilelem, left_itest_func, left_elem.inodes[j])] +=
                            left_cano_length * edge_weights[i] * dleft_transformed_flux_dy[j];
                        dx[(irelem, right_itest_func, right_elem.inodes[j])] +=
                            right_cano_length * edge_weights[i] * dright_transformed_flux_dx[j];
                        dy[(irelem, right_itest_func, right_elem.inodes[j])] +=
                            right_cano_length * edge_weights[i] * dright_transformed_flux_dy[j];
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
                    let phys_x: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                    let phys_y: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_edge_ids = &edge.local_ids;
                    let local_node_ids = [local_edge_ids[0], (local_edge_ids[0] + 1) % 3];
                    let cano_normal = Self::compute_canonical_normal(local_edge_ids[0]);
                    let cano_length = Self::compute_canonical_edge_length(local_edge_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.xi.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.eta.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    for i in 0..nedge_basis {
                        let xi = xi_slice[i];
                        let eta = eta_slice[i];
                        // let (boundary_flux, dflux_du, dflux_dx0, dflux_dx1, dflux_dy0, dflux_dy1): (
                        //    f64,
                        //    f64,
                        //    f64,
                        //    f64,
                        //    f64,
                        //    f64,
                        // ) = self.dbnd_flux(
                        //     sol_slice[i],
                        //     bnd_value,
                        //     phys_x[local_node_ids[0]],
                        //     phys_x[local_node_ids[1]],
                        //     phys_y[local_node_ids[0]],
                        //     phys_y[local_node_ids[1]],
                        //     1.0,
                        // );
                        // ) = {
                        //    let boundary_flux = self.compute_smoothed_boundary_flux(
                        //        sol_slice[i],
                        //        bnd_value,
                        //        phys_x[local_node_ids[0]],
                        //        phys_x[local_node_ids[1]],
                        //        phys_y[local_node_ids[0]],
                        //        phys_y[local_node_ids[1]],
                        //        1.0,
                        //    );
                        //    (boundary_flux, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) // TODO: Replace with finite difference
                        // };
                        // Compute boundary flux and its derivatives using finite difference
                        let fd = FiniteDifference::new();
                        let flux_fn = |u: f64, x0: f64, x1: f64, y0: f64, y1: f64| {
                            self.compute_smoothed_boundary_flux(u, bnd_value, x0, x1, y0, y1)
                        };

                        let boundary_flux = flux_fn(
                            sol_slice[i],
                            phys_x[local_node_ids[0]],
                            phys_x[local_node_ids[1]],
                            phys_y[local_node_ids[0]],
                            phys_y[local_node_ids[1]],
                        );

                        // Compute derivatives using finite differences
                        let dflux_du = fd.scalar_derivative(
                            |u| {
                                flux_fn(
                                    u,
                                    phys_x[local_node_ids[0]],
                                    phys_x[local_node_ids[1]],
                                    phys_y[local_node_ids[0]],
                                    phys_y[local_node_ids[1]],
                                )
                            },
                            sol_slice[i],
                        );
                        let dflux_dx0 = fd.scalar_derivative(
                            |x0| {
                                flux_fn(
                                    sol_slice[i],
                                    x0,
                                    phys_x[local_node_ids[1]],
                                    phys_y[local_node_ids[0]],
                                    phys_y[local_node_ids[1]],
                                )
                            },
                            phys_x[local_node_ids[0]],
                        );
                        let dflux_dx1 = fd.scalar_derivative(
                            |x1| {
                                flux_fn(
                                    sol_slice[i],
                                    phys_x[local_node_ids[0]],
                                    x1,
                                    phys_y[local_node_ids[0]],
                                    phys_y[local_node_ids[1]],
                                )
                            },
                            phys_x[local_node_ids[1]],
                        );
                        let dflux_dy0 = fd.scalar_derivative(
                            |y0| {
                                flux_fn(
                                    sol_slice[i],
                                    phys_x[local_node_ids[0]],
                                    phys_x[local_node_ids[1]],
                                    y0,
                                    phys_y[local_node_ids[1]],
                                )
                            },
                            phys_y[local_node_ids[0]],
                        );
                        let dflux_dy1 = fd.scalar_derivative(
                            |y1| {
                                flux_fn(
                                    sol_slice[i],
                                    phys_x[local_node_ids[0]],
                                    phys_x[local_node_ids[1]],
                                    phys_y[local_node_ids[0]],
                                    y1,
                                )
                            },
                            phys_y[local_node_ids[1]],
                        );

                        let mut dflux_dx = [0.0; 3];
                        let mut dflux_dy = [0.0; 3];
                        dflux_dx[local_node_ids[0]] = dflux_dx0;
                        dflux_dx[local_node_ids[1]] = dflux_dx1;
                        dflux_dy[local_node_ids[0]] = dflux_dy0;
                        dflux_dy[local_node_ids[1]] = dflux_dy1;

                        let mut dscaling_dx = [0.0; 3];
                        let mut dscaling_dy = [0.0; 3];
                        // let scaling: f64 = self.dscaling(
                        //     xi,
                        //     eta,
                        //     cano_normal,
                        //     &phys_x,
                        //     dscaling_dx.as_mut_slice(),
                        //     &phys_y,
                        //     dscaling_dy.as_mut_slice(),
                        //     1.0,
                        // );
                        let scaling: f64 = compute_flux_scaling_derivatives(
                            &fd,
                            |xi, eta, ref_normal, x, y| {
                                self.compute_flux_scaling(xi, eta, ref_normal, x, y)
                            },
                            xi,
                            eta,
                            cano_normal,
                            &phys_x,
                            &phys_y,
                            &mut dscaling_dx,
                            &mut dscaling_dy,
                        );
                        let transformed_flux = boundary_flux * scaling;

                        let dtransformed_flux_du = dflux_du * scaling;
                        let dtransformed_flux_dx = &(&ArrayView1::from(&dflux_dx) * scaling)
                            + &(&ArrayView1::from(&dscaling_dx) * boundary_flux);
                        let dtransformed_flux_dy = &(&ArrayView1::from(&dflux_dy) * scaling)
                            + &(&ArrayView1::from(&dscaling_dy) * boundary_flux);

                        let itest_func = basis.nodes_along_edges[(local_edge_ids[0], i)];

                        residuals[(ielem, itest_func)] +=
                            cano_length * edge_weights[i] * transformed_flux;

                        for j in 0..3 {
                            dx[(ielem, itest_func, inodes[j])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_dx[j];
                            dy[(ielem, itest_func, inodes[j])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_dy[j];
                        }

                        if is_enriched {
                            for (j, &isol_node) in sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .indexed_iter()
                            {
                                dsol[(ielem, itest_func, ielem, isol_node)] += cano_length
                                    * edge_weights[i]
                                    * self.interp_node_to_enriched_quadrature()[(i, j)]
                                    * dtransformed_flux_du;
                            }
                        } else {
                            let sol_nodes = sol_nodes_along_edges.slice(s![local_edge_ids[0], ..]);
                            dsol[(ielem, itest_func, ielem, sol_nodes[i])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_du;
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
                    let phys_x: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                    let phys_y: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_edge_ids = &edge.local_ids;
                    let local_node_ids = [local_edge_ids[0], (local_edge_ids[0] + 1) % 3];
                    let cano_normal = Self::compute_canonical_normal(local_edge_ids[0]);
                    let cano_length = Self::compute_canonical_edge_length(local_edge_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.xi.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.eta.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );

                    // Compute the physical coordinates of the quadrature points on the edge
                    let shape_func = [
                        1.0 - &xi_slice - &eta_slice,
                        xi_slice.to_owned(),
                        eta_slice.to_owned(),
                    ];

                    let x_phys = &shape_func[0] * phys_x[0]
                        + &shape_func[1] * phys_x[1]
                        + &shape_func[2] * phys_x[2];
                    let y_phys = &shape_func[0] * phys_y[0]
                        + &shape_func[1] * phys_y[1]
                        + &shape_func[2] * phys_y[2];

                    let bnd_values = Zip::from(&x_phys)
                        .and(&y_phys)
                        .map_collect(|&x, &y| func(x, y));

                    // Compute the derivatives of boundary values with respect to coordinates changes of the edge nodes
                    let bnd_values_x0_perturbed = {
                        let mut x_slice_perturbed = phys_x;
                        x_slice_perturbed[local_node_ids[0]] += 1e-6;
                        let x_phys = &shape_func[0] * x_slice_perturbed[0]
                            + &shape_func[1] * x_slice_perturbed[1]
                            + &shape_func[2] * x_slice_perturbed[2];
                        let y_phys = &shape_func[0] * phys_y[0]
                            + &shape_func[1] * phys_y[1]
                            + &shape_func[2] * phys_y[2];
                        Zip::from(&x_phys)
                            .and(&y_phys)
                            .map_collect(|&x, &y| func(x, y))
                    };
                    let bnd_values_x1_perturbed = {
                        let mut x_slice_perturbed = phys_x;
                        x_slice_perturbed[local_node_ids[1]] += 1e-6;
                        let x_phys = &shape_func[0] * x_slice_perturbed[0]
                            + &shape_func[1] * x_slice_perturbed[1]
                            + &shape_func[2] * x_slice_perturbed[2];
                        let y_phys = &shape_func[0] * phys_y[0]
                            + &shape_func[1] * phys_y[1]
                            + &shape_func[2] * phys_y[2];
                        Zip::from(&x_phys)
                            .and(&y_phys)
                            .map_collect(|&x, &y| func(x, y))
                    };
                    let bnd_values_y0_perturbed = {
                        let mut y_slice_perturbed = phys_y;
                        y_slice_perturbed[local_node_ids[0]] += 1e-6;
                        let x_phys = &shape_func[0] * phys_x[0]
                            + &shape_func[1] * phys_x[1]
                            + &shape_func[2] * phys_x[2];
                        let y_phys = &shape_func[0] * y_slice_perturbed[0]
                            + &shape_func[1] * y_slice_perturbed[1]
                            + &shape_func[2] * y_slice_perturbed[2];
                        Zip::from(&x_phys)
                            .and(&y_phys)
                            .map_collect(|&x, &y| func(x, y))
                    };
                    let bnd_values_y1_perturbed = {
                        let mut y_slice_perturbed = phys_y;
                        y_slice_perturbed[local_node_ids[1]] += 1e-6;
                        let x_phys = &shape_func[0] * phys_x[0]
                            + &shape_func[1] * phys_x[1]
                            + &shape_func[2] * phys_x[2];
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
                            // ): (f64, f64, f64, f64, f64, f64, f64) = self.dnum_flux(
                            //     sol_slice[i],
                            //     bnd_values[i],
                            //     phys_x[local_node_ids[0]],
                            //     phys_x[local_node_ids[1]],
                            //     phys_y[local_node_ids[0]],
                            //     phys_y[local_node_ids[1]],
                            //     1.0,
                            // );
                        ): (f64, f64, f64, f64, f64, f64, f64) = {
                            let boundary_flux = self.compute_smoothed_numerical_flux(
                                sol_slice[i],
                                bnd_values[i],
                                phys_x[local_node_ids[0]],
                                phys_x[local_node_ids[1]],
                                phys_y[local_node_ids[0]],
                                phys_y[local_node_ids[1]],
                            );
                            (boundary_flux, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) // TODO: Replace with finite difference
                        };
                        let mut dflux_dx = [0.0; 3];
                        let mut dflux_dy = [0.0; 3];
                        dflux_dx[local_node_ids[0]] =
                            dflux_dx0 + dflux_dbnd_value * dbnd_values_dx0[i];
                        dflux_dx[local_node_ids[1]] =
                            dflux_dx1 + dflux_dbnd_value * dbnd_values_dx1[i];
                        dflux_dy[local_node_ids[0]] =
                            dflux_dy0 + dflux_dbnd_value * dbnd_values_dy0[i];
                        dflux_dy[local_node_ids[1]] =
                            dflux_dy1 + dflux_dbnd_value * dbnd_values_dy1[i];

                        let mut dscaling_dx = [0.0; 3];
                        let mut dscaling_dy = [0.0; 3];
                        // let scaling: f64 = self.dscaling(
                        //     xi,
                        //     eta,
                        //     cano_normal,
                        //     &phys_x,
                        //     dscaling_dx.as_mut_slice(),
                        //     &phys_y,
                        //     dscaling_dy.as_mut_slice(),
                        //     1.0,
                        // );
                        let scaling: f64 = compute_flux_scaling_derivatives(
                            &fd,
                            |xi, eta, ref_normal, x, y| {
                                self.compute_flux_scaling(xi, eta, ref_normal, x, y)
                            },
                            xi,
                            eta,
                            cano_normal,
                            &phys_x,
                            &phys_y,
                            &mut dscaling_dx,
                            &mut dscaling_dy,
                        );
                        let transformed_flux = boundary_flux * scaling;

                        let dtransformed_flux_du = dflux_du * scaling;
                        let dtransformed_flux_dx = &(&ArrayView1::from(&dflux_dx) * scaling)
                            + &(&ArrayView1::from(&dscaling_dx) * boundary_flux);
                        let dtransformed_flux_dy = &(&ArrayView1::from(&dflux_dy) * scaling)
                            + &(&ArrayView1::from(&dscaling_dy) * boundary_flux);

                        let itest_func = basis.nodes_along_edges[(local_edge_ids[0], i)];

                        residuals[(ielem, itest_func)] +=
                            cano_length * edge_weights[i] * transformed_flux;

                        for j in 0..3 {
                            dx[(ielem, itest_func, inodes[j])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_dx[j];
                            dy[(ielem, itest_func, inodes[j])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_dy[j];
                        }

                        if is_enriched {
                            for (j, &isol_node) in sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .indexed_iter()
                            {
                                dsol[(ielem, itest_func, ielem, isol_node)] += cano_length
                                    * edge_weights[i]
                                    * self.interp_node_to_enriched_quadrature()[(i, j)]
                                    * dtransformed_flux_du;
                            }
                        } else {
                            let sol_nodes = sol_nodes_along_edges.slice(s![local_edge_ids[0], ..]);
                            dsol[(ielem, itest_func, ielem, sol_nodes[i])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_du;
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
                                dsol[(ielem, itest_func, ielem, isol_node)] += edge_length
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
                    let phys_x: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                    let phys_y: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                    let sol_nodes_along_edges = &self.basis().nodes_along_edges;
                    let nodes_along_edges = &basis.nodes_along_edges;
                    let local_edge_ids = &edge.local_ids;
                    let local_node_ids = [local_edge_ids[0], (local_edge_ids[0] + 1) % 3];
                    let cano_normal = Self::compute_canonical_normal(local_edge_ids[0]);
                    let cano_length = Self::compute_canonical_edge_length(local_edge_ids[0]);
                    let sol_slice = {
                        let sol_slice = solutions.slice(s![ielem, ..]).select(
                            Axis(0),
                            sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        if is_enriched {
                            self.interp_node_to_enriched_quadrature().dot(&sol_slice)
                        } else {
                            sol_slice
                        }
                    };
                    let xi_slice = basis.xi.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    let eta_slice = basis.eta.select(
                        Axis(0),
                        nodes_along_edges
                            .slice(s![local_edge_ids[0], ..])
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
                        // ) = self.dopen_bnd_flux(
                        //     u,
                        //     phys_x[local_node_ids[0]],
                        //     phys_x[local_node_ids[1]],
                        //     phys_y[local_node_ids[0]],
                        //     phys_y[local_node_ids[1]],
                        //     1.0,
                        // );
                        ) = {
                            let boundary_flux = self.compute_open_boundary_flux(
                                u,
                                phys_x[local_node_ids[0]],
                                phys_x[local_node_ids[1]],
                                phys_y[local_node_ids[0]],
                                phys_y[local_node_ids[1]],
                            );
                            (boundary_flux, 0.0, 0.0, 0.0, 0.0, 0.0) // TODO: Replace with finite difference
                        };

                        let mut dflux_dx = [0.0; 3];
                        let mut dflux_dy = [0.0; 3];
                        dflux_dx[local_edge_ids[0]] = dflux_dx0;
                        dflux_dx[local_node_ids[1]] = dflux_dx1;
                        dflux_dy[local_node_ids[0]] = dflux_dy0;
                        dflux_dy[local_node_ids[1]] = dflux_dy1;

                        let mut dscaling_dx = [0.0; 3];
                        let mut dscaling_dy = [0.0; 3];
                        // let scaling: f64 = self.dscaling(
                        //     xi,
                        //     eta,
                        //     cano_normal,
                        //     &phys_x,
                        //     dscaling_dx.as_mut_slice(),
                        //     &phys_y,
                        //     dscaling_dy.as_mut_slice(),
                        //     1.0,
                        // );
                        let scaling: f64 = compute_flux_scaling_derivatives(
                            &fd,
                            |xi, eta, ref_normal, x, y| {
                                self.compute_flux_scaling(xi, eta, ref_normal, x, y)
                            },
                            xi,
                            eta,
                            cano_normal,
                            &phys_x,
                            &phys_y,
                            &mut dscaling_dx,
                            &mut dscaling_dy,
                        );
                        let transformed_flux = boundary_flux * scaling;

                        let dtransformed_flux_du = dflux_du * scaling;
                        let dtransformed_flux_dx = &(&ArrayView1::from(&dflux_dx) * scaling)
                            + &(&ArrayView1::from(&dscaling_dx) * boundary_flux);
                        let dtransformed_flux_dy = &(&ArrayView1::from(&dflux_dy) * scaling)
                            + &(&ArrayView1::from(&dscaling_dy) * boundary_flux);

                        let itest_func = basis.nodes_along_edges[(local_edge_ids[0], i)];
                        if ielem == 1 {
                            println!("boundary_flux: {:?}", boundary_flux);
                            println!("transformed_flux: {:?}", transformed_flux);
                        }
                        residuals[(ielem, itest_func)] +=
                            cano_length * edge_weights[i] * transformed_flux;

                        for j in 0..3 {
                            dx[(ielem, itest_func, inodes[j])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_dx[j];
                            dy[(ielem, itest_func, inodes[j])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_dy[j];
                        }

                        if is_enriched {
                            for (j, &isol_node) in sol_nodes_along_edges
                                .slice(s![local_edge_ids[0], ..])
                                .indexed_iter()
                            {
                                dsol[(ielem, itest_func, ielem, isol_node)] += cano_length
                                    * edge_weights[i]
                                    * self.interp_node_to_enriched_quadrature()[(i, j)]
                                    * dtransformed_flux_du;
                            }
                        } else {
                            let sol_nodes = sol_nodes_along_edges.slice(s![local_edge_ids[0], ..]);
                            dsol[(ielem, itest_func, ielem, sol_nodes[i])] +=
                                cano_length * edge_weights[i] * dtransformed_flux_du;
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
        let ref_distortion = Self::compute_distortion(&ref_x, &ref_y, self.basis());
        for (elem_idx, element) in mesh.elements.iter().enumerate() {
            if let Status::Active(element) = element {
                let inodes = &element.inodes;
                let x_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                let y_slice: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                let distortion = Self::compute_distortion(&x_slice, &y_slice, self.basis());
                res_mesh[elem_idx] = distortion - ref_distortion;

                let mut distortion_x_perturbed = [0.0; 3];
                let mut distortion_y_perturbed = [0.0; 3];
                distortion_x_perturbed[0] = {
                    let mut x_slice_perturbed = x_slice;
                    x_slice_perturbed[0] += 1e-6;
                    Self::compute_distortion(&x_slice_perturbed, &y_slice, self.basis())
                };
                distortion_x_perturbed[1] = {
                    let mut x_slice_perturbed = x_slice;
                    x_slice_perturbed[1] += 1e-6;
                    Self::compute_distortion(&x_slice_perturbed, &y_slice, self.basis())
                };
                distortion_x_perturbed[2] = {
                    let mut x_slice_perturbed = x_slice;
                    x_slice_perturbed[2] += 1e-6;
                    Self::compute_distortion(&x_slice_perturbed, &y_slice, self.basis())
                };
                distortion_y_perturbed[0] = {
                    let mut y_slice_perturbed = y_slice;
                    y_slice_perturbed[0] += 1e-6;
                    Self::compute_distortion(&x_slice, &y_slice_perturbed, self.basis())
                };
                distortion_y_perturbed[1] = {
                    let mut y_slice_perturbed = y_slice;
                    y_slice_perturbed[1] += 1e-6;
                    Self::compute_distortion(&x_slice, &y_slice_perturbed, self.basis())
                };
                distortion_y_perturbed[2] = {
                    let mut y_slice_perturbed = y_slice;
                    y_slice_perturbed[2] += 1e-6;
                    Self::compute_distortion(&x_slice, &y_slice_perturbed, self.basis())
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
    // fn dvolume(
    //     &self,
    //     basis: &TriangleBasis,
    //     itest_func: usize,
    //     sol: &[f64],
    //     d_sol: &mut [f64],
    //     x: &[f64],
    //     d_x: &mut [f64],
    //     y: &[f64],
    //     d_y: &mut [f64],
    //     d_retval: f64,
    // ) -> f64;
    fn compute_boundary_flux(&self, u: f64, ub: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64;
    fn compute_smoothed_boundary_flux(
        &self,
        u: f64,
        ub: f64,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    ) -> f64;
    // fn dbnd_flux(
    //     &self,
    //     u: f64,
    //     ub: f64,
    //     x0: f64,
    //     x1: f64,
    //     y0: f64,
    //     y1: f64,
    //     d_retval: f64,
    // ) -> (f64, f64, f64, f64, f64, f64);
    fn compute_numerical_flux(&self, ul: f64, ur: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64;
    fn compute_smoothed_numerical_flux(
        &self,
        ul: f64,
        ur: f64,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    ) -> f64;
    // fn dnum_flux(
    //     &self,
    //     ul: f64,
    //     ur: f64,
    //     x0: f64,
    //     x1: f64,
    //     y0: f64,
    //     y1: f64,
    //     d_retval: f64,
    // ) -> (f64, f64, f64, f64, f64, f64, f64);
    fn compute_flux_scaling(
        &self,
        xi: f64,
        eta: f64,
        ref_normal: [f64; 2],
        x: &[f64],
        y: &[f64],
    ) -> f64;
    // fn dscaling(
    //     &self,
    //     xi: f64,
    //     eta: f64,
    //     ref_normal: [f64; 2],
    //     x: &[f64],
    //     d_x: &mut [f64],
    //     y: &[f64],
    //     d_y: &mut [f64],
    //     d_retval: f64,
    // ) -> f64;
    fn compute_open_boundary_flux(&self, u: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64;
    // fn dopen_bnd_flux(
    //     &self,
    //     u: f64,
    //     x0: f64,
    //     x1: f64,
    //     y0: f64,
    //     y1: f64,
    //     d_retval: f64,
    // ) -> (f64, f64, f64, f64, f64, f64);
    fn compute_interior_flux(
        &self,
        xi: f64,
        eta: f64,
        ref_normal: [f64; 2],
        u: f64,
        x: &[f64],
        y: &[f64],
    ) -> f64;
    // fn dinterior_flux(
    //     &self,
    //     xi: f64,
    //     eta: f64,
    //     ref_normal: [f64; 2],
    //     u: f64,
    //     x: &[f64],
    //     d_x: &mut [f64],
    //     y: &[f64],
    //     d_y: &mut [f64],
    //     d_retval: f64,
    // ) -> (f64, f64);
    fn physical_flux(&self, u: f64) -> [f64; 2];
    fn initialize_solution(&self, solutions: ArrayViewMut2<f64>);
    fn set_initial_solution(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        initial_guess: ArrayView2<f64>,
    ) -> Array2<f64> {
        let ncell_basis = self.basis().xi.len();
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
