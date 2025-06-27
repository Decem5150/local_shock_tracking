pub mod ader;
pub mod basis;
// pub mod boundary_conditions;
// pub mod flux;
pub mod gauss_points;
pub mod geometric;
pub mod mesh;
// pub mod riemann_solver;
// pub mod advection1d_space_time_quad;
pub mod advection1d_space_time_tri;
pub mod burgers1d;
pub mod burgers1d_space_time;
// pub mod euler1d;
use faer::{Col, linalg::solvers::DenseSolveCore, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, concatenate, s};
use ndarray_stats::QuantileExt;

use crate::disc::{
    basis::{Basis, quadrilateral::QuadrilateralBasis, triangle::TriangleBasis},
    geometric::Geometric2D,
    mesh::mesh2d::{Mesh2d, TriangleElement},
};

pub trait P0Solver: Geometric2D + SpaceTimeSolver1DScalar {
    fn compute_initial_guess(&self) -> Array2<f64> {
        let mut solutions = Array2::zeros((self.mesh().elem_num, 1));
        self.initialize_solution(solutions.view_mut());
        let nelem = self.mesh().elem_num;
        let mut residuals = Array2::<f64>::zeros((nelem, 1));
        let max_iter = 5000;
        let tol = 1e-12;

        for i in 0..max_iter {
            self.compute_p0_residuals(solutions.view(), residuals.view_mut());

            let res_norm = residuals.iter().map(|x| x.powi(2)).sum::<f64>().sqrt() / (nelem as f64);

            if i % 100 == 0 {
                println!("solutions: {:?}", solutions);
                println!("PTC Iter: {}, Res norm: {}", i, res_norm);
            }

            if res_norm < tol {
                println!("PTC converged after {} iterations.", i);
                println!("solutions: {:?}", solutions);
                return solutions;
            }

            let dts = self.compute_time_steps(solutions.view());

            for ielem in 0..nelem {
                let elem = &self.mesh().elements[ielem];
                let x: [f64; 3] = std::array::from_fn(|i| self.mesh().nodes[elem.inodes[i]].x);
                let y: [f64; 3] = std::array::from_fn(|i| self.mesh().nodes[elem.inodes[i]].y);
                let area = Self::compute_element_area(&x, &y);
                solutions[[ielem, 0]] -= dts[ielem] / area * residuals[[ielem, 0]];
            }
        }
        println!("PTC did not converge within {} iterations.", max_iter);
        solutions
    }

    fn compute_p0_residuals(&self, solutions: ArrayView2<f64>, mut residuals: ArrayViewMut2<f64>) {
        residuals.fill(0.0);

        // Internal edges
        for &iedge in &self.mesh().internal_edges {
            let edge = &self.mesh().edges[iedge];
            let ileft = edge.parents[0];
            let iright = edge.parents[1];

            let u_left = solutions[(ileft, 0)];
            let u_right = solutions[(iright, 0)];

            let n0 = &self.mesh().nodes[edge.inodes[0]];
            let n1 = &self.mesh().nodes[edge.inodes[1]];

            let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();

            // Per the user's convention, edge.parents[0] is the left element.
            // The node ordering in edge.inodes is assumed to be counter-clockwise for the left element,
            // so the normal computed from (n0, n1) will point from left to right.
            let flux = self.compute_numerical_flux(u_left, u_right, n0.x, n1.x, n0.y, n1.y);
            println!("flux: {:?}", flux);
            residuals[(ileft, 0)] += flux * edge_length;
            residuals[(iright, 0)] -= flux * edge_length;
        }

        // Inflow boundaries
        for bnd in &self.mesh().flow_in_bnds {
            let u_bnd = bnd.value;
            for &iedge in &bnd.iedges {
                let edge = &self.mesh().edges[iedge];
                let ielem = edge.parents[0];
                let n0 = &self.mesh().nodes[edge.inodes[0]];
                let n1 = &self.mesh().nodes[edge.inodes[1]];

                let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();

                // For boundary edges, the node ordering is assumed to be counter-clockwise
                // for the parent element, so the normal computed from (n0, n1) is outward-pointing.
                let flux = self.compute_boundary_flux(u_bnd, n0.x, n1.x, n0.y, n1.y);

                residuals[(ielem, 0)] += flux * edge_length;
            }
        }

        // Outflow boundaries
        for bnd in &self.mesh().flow_out_bnds {
            for &iedge in &bnd.iedges {
                let edge = &self.mesh().edges[iedge];
                let ielem = edge.parents[0];
                let u_in = solutions[(ielem, 0)];
                let n0 = &self.mesh().nodes[edge.inodes[0]];
                let n1 = &self.mesh().nodes[edge.inodes[1]];

                let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();

                // For boundary edges, the node ordering is assumed to be counter-clockwise
                // for the parent element, so the normal computed from (n0, n1) is outward-pointing.
                let flux = self.compute_boundary_flux(u_in, n0.x, n1.x, n0.y, n1.y);

                residuals[(ielem, 0)] += flux * edge_length;
            }
        }
    }
    fn compute_time_steps(&self, _solutions: ArrayView2<f64>) -> Array1<f64>;
}

pub trait SpaceTimeSolver1DScalar: Geometric2D {
    fn basis(&self) -> &TriangleBasis;
    fn enriched_basis(&self) -> &TriangleBasis;
    fn interp_node_to_cubature(&self) -> &Array2<f64>;
    fn interp_node_to_enriched_cubature(&self) -> &Array2<f64>;
    fn interp_node_to_enriched_quadrature(&self) -> &Array2<f64>;
    fn mesh(&self) -> &Mesh2d<TriangleElement>;
    fn mesh_mut(&mut self) -> &mut Mesh2d<TriangleElement>;

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
            let inodes = &elem.inodes;
            let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].x);
            let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].y);
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
        for &iedge in mesh.internal_edges.iter() {
            let edge = &mesh.edges[iedge];
            let ilelem = edge.parents[0];
            let irelem = edge.parents[1];
            let left_elem = &mesh.elements[ilelem];
            let right_elem = &mesh.elements[irelem];
            let left_inodes = &left_elem.inodes;
            let right_inodes = &right_elem.inodes;
            let left_x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[left_inodes[i]].x);
            let left_y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[left_inodes[i]].y);
            let right_x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[right_inodes[i]].x);
            let right_y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[right_inodes[i]].y);
            let common_edge = [edge.local_ids[0], edge.local_ids[1]];
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
                    left_x_slice[common_edge[0]],
                    left_x_slice[(common_edge[0] + 1) % 3],
                    left_y_slice[common_edge[0]],
                    left_y_slice[(common_edge[0] + 1) % 3],
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
                let right_itest_func = basis.nodes_along_edges[(local_ids[1], nedge_basis - 1 - i)];
                residuals[(ilelem, left_itest_func)] +=
                    0.5 * left_edge_length * edge_weights[i] * left_transformed_flux;
                residuals[(irelem, right_itest_func)] +=
                    0.5 * right_edge_length * edge_weights[i] * right_transformed_flux;
            }
        }
        // flow in boundary
        for ibnd in mesh.flow_in_bnds.iter() {
            let iedges = &ibnd.iedges;
            let value = ibnd.value;
            for &iedge in iedges.iter() {
                let edge = &mesh.edges[iedge];
                let ielem = edge.parents[0];
                let elem = &mesh.elements[ielem];
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].y);
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_ids = &edge.local_ids;
                let ref_normal = Self::compute_ref_normal(local_ids[0]);
                let edge_length = Self::compute_ref_edge_length(local_ids[0]);
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
                        value,
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
                        0.5 * edge_length * edge_weights[i] * transformed_flux;
                }
            }
        }
        // flow out boundary
        for ibnd in mesh.flow_out_bnds.iter() {
            let iedges = &ibnd.iedges;
            for &iedge in iedges.iter() {
                let edge = &mesh.edges[iedge];
                let ielem = edge.parents[0];
                let elem = &mesh.elements[ielem];
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].y);
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
                        0.5 * edge_length * edge_weights[i] * transformed_flux;
                }
            }
        }
    }
    fn compute_residuals_and_derivatives(
        &self,
        solutions: ArrayView2<f64>,
        mut residuals: ArrayViewMut2<f64>,
        mut dsol: ArrayViewMut2<f64>,
        mut dx: ArrayViewMut2<f64>,
        mut dy: ArrayViewMut2<f64>,
        is_enriched: bool,
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
        for (ielem, elem) in self.mesh().elements.iter().enumerate() {
            let inodes = &elem.inodes;
            let x_slice: [f64; 3] = std::array::from_fn(|i| self.mesh().nodes[inodes[i]].x);
            let y_slice: [f64; 3] = std::array::from_fn(|i| self.mesh().nodes[inodes[i]].y);
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

                let res_row_idx = ielem * ncell_basis + itest_func;
                let sol_col_range =
                    ielem * unenriched_ncell_basis..(ielem + 1) * unenriched_ncell_basis;
                dsol.slice_mut(s![res_row_idx, sol_col_range])
                    .scaled_add(1.0, &dres_dsol_dofs);
                for i in 0..3 {
                    dx[(res_row_idx, inodes[i])] += dvol_x[i];
                    dy[(res_row_idx, inodes[i])] += dvol_y[i];
                }
            }
        }
        for &iedge in self.mesh().internal_edges.iter() {
            let edge = &self.mesh().edges[iedge];
            let ilelem = edge.parents[0];
            let irelem = edge.parents[1];
            let left_elem = &self.mesh().elements[ilelem];
            let right_elem = &self.mesh().elements[irelem];
            let left_inodes = &left_elem.inodes;
            let right_inodes = &right_elem.inodes;
            let left_x_slice: [f64; 3] =
                std::array::from_fn(|i| self.mesh().nodes[left_inodes[i]].x);
            let left_y_slice: [f64; 3] =
                std::array::from_fn(|i| self.mesh().nodes[left_inodes[i]].y);
            let right_x_slice: [f64; 3] =
                std::array::from_fn(|i| self.mesh().nodes[right_inodes[i]].x);
            let right_y_slice: [f64; 3] =
                std::array::from_fn(|i| self.mesh().nodes[right_inodes[i]].y);
            let common_edge = [edge.local_ids[0], edge.local_ids[1]];
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
                let (num_flux, dflux_dul, dflux_dur, dflux_dx0, dflux_dx1, dflux_dy0, dflux_dy1): (
                    f64,
                    f64,
                    f64,
                    f64,
                    f64,
                    f64,
                    f64,
                ) = self.dnum_flux(
                    left_value,
                    right_value,
                    left_x_slice[common_edge[0]],
                    left_x_slice[(common_edge[0] + 1) % 3],
                    left_y_slice[common_edge[0]],
                    left_y_slice[(common_edge[0] + 1) % 3],
                    1.0,
                );
                dflux_dleft_x[common_edge[0]] = dflux_dx0;
                dflux_dleft_x[(common_edge[0] + 1) % 3] = dflux_dx1;
                dflux_dleft_y[common_edge[0]] = dflux_dy0;
                dflux_dleft_y[(common_edge[0] + 1) % 3] = dflux_dy1;
                dflux_dright_x[common_edge[1]] = dflux_dx1;
                dflux_dright_x[(common_edge[1] + 1) % 3] = dflux_dx0;
                dflux_dright_y[common_edge[1]] = dflux_dy1;
                dflux_dright_y[(common_edge[1] + 1) % 3] = dflux_dy0;

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
                let right_itest_func = basis.nodes_along_edges[(local_ids[1], nedge_basis - 1 - i)];
                residuals[(ilelem, left_itest_func)] +=
                    0.5 * left_edge_length * edge_weights[i] * left_transformed_flux;
                residuals[(irelem, right_itest_func)] +=
                    0.5 * right_edge_length * edge_weights[i] * right_transformed_flux;

                let row_idx_left = ilelem * ncell_basis + left_itest_func;
                let row_idx_right = irelem * ncell_basis + right_itest_func;
                if is_enriched {
                    // derivatives w.r.t. left value
                    for (j, &isol_node) in sol_nodes_along_edges
                        .slice(s![local_ids[0], ..])
                        .indexed_iter()
                    {
                        let col_idx = ilelem * unenriched_ncell_basis + isol_node;
                        dsol[(row_idx_left, col_idx)] += 0.5
                            * left_edge_length
                            * edge_weights[i]
                            * self.interp_node_to_enriched_quadrature()[(i, j)]
                            * dleft_transformed_flux_dul;
                        dsol[(row_idx_right, col_idx)] += 0.5
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
                        let col_idx = irelem * unenriched_ncell_basis + isol_node;
                        dsol[(row_idx_left, col_idx)] += 0.5
                            * left_edge_length
                            * edge_weights[i]
                            * self.interp_node_to_enriched_quadrature()[(nedge_basis - 1 - i, j)]
                            * dleft_transformed_flux_dur;
                        dsol[(row_idx_right, col_idx)] += 0.5
                            * right_edge_length
                            * edge_weights[i]
                            * self.interp_node_to_enriched_quadrature()[(nedge_basis - 1 - i, j)]
                            * dright_transformed_flux_dur;
                    }
                } else {
                    let left_sol_nodes = sol_nodes_along_edges.slice(s![local_ids[0], ..]);
                    let right_sol_nodes = sol_nodes_along_edges.slice(s![local_ids[1], ..]);
                    let col_idx_left = ilelem * ncell_basis + left_sol_nodes[i];
                    let col_idx_right = irelem * ncell_basis + right_sol_nodes[nedge_basis - 1 - i];
                    // derivatives w.r.t. left value
                    dsol[(row_idx_left, col_idx_left)] +=
                        0.5 * left_edge_length * edge_weights[i] * dleft_transformed_flux_dul;
                    dsol[(row_idx_right, col_idx_left)] +=
                        0.5 * right_edge_length * edge_weights[i] * dright_transformed_flux_dul;
                    // derivatives w.r.t. right value
                    dsol[(row_idx_left, col_idx_right)] +=
                        0.5 * left_edge_length * edge_weights[i] * dleft_transformed_flux_dur;
                    dsol[(row_idx_right, col_idx_right)] +=
                        0.5 * right_edge_length * edge_weights[i] * dright_transformed_flux_dur;
                }
                for j in 0..3 {
                    dx[(row_idx_left, left_elem.inodes[j])] +=
                        0.5 * left_edge_length * edge_weights[i] * dleft_transformed_flux_dx[j];
                    dy[(row_idx_left, left_elem.inodes[j])] +=
                        0.5 * left_edge_length * edge_weights[i] * dleft_transformed_flux_dy[j];
                    dx[(row_idx_right, right_elem.inodes[j])] +=
                        0.5 * right_edge_length * edge_weights[i] * dright_transformed_flux_dx[j];
                    dy[(row_idx_right, right_elem.inodes[j])] +=
                        0.5 * right_edge_length * edge_weights[i] * dright_transformed_flux_dy[j];
                }
            }
        }
        // flow in boundary
        for ibnd in self.mesh().flow_in_bnds.iter() {
            let iedges = &ibnd.iedges;
            let value = ibnd.value;
            for &iedge in iedges.iter() {
                let edge = &self.mesh().edges[iedge];
                let ielem = edge.parents[0];
                let elem = &self.mesh().elements[ielem];
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| self.mesh().nodes[inodes[i]].x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| self.mesh().nodes[inodes[i]].y);
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_ids = &edge.local_ids;
                let ref_normal = Self::compute_ref_normal(local_ids[0]);
                let edge_length = Self::compute_ref_edge_length(local_ids[0]);
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
                    let (boundary_flux, _dflux_dbnd, dflux_dx0, dflux_dx1, dflux_dy0, dflux_dy1): (
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                        f64,
                    ) = self.dbnd_flux(
                        value,
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

                    let dtransformed_flux_dx = &(&ArrayView1::from(&dflux_dx) * scaling)
                        + &(&ArrayView1::from(&dscaling_dx) * boundary_flux);
                    let dtransformed_flux_dy = &(&ArrayView1::from(&dflux_dy) * scaling)
                        + &(&ArrayView1::from(&dscaling_dy) * boundary_flux);

                    let itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                    residuals[(ielem, itest_func)] +=
                        0.5 * edge_length * edge_weights[i] * transformed_flux;

                    let row_idx = ielem * ncell_basis + itest_func;
                    for j in 0..3 {
                        dx[(row_idx, inodes[j])] +=
                            0.5 * edge_length * edge_weights[i] * dtransformed_flux_dx[j];
                        dy[(row_idx, inodes[j])] +=
                            0.5 * edge_length * edge_weights[i] * dtransformed_flux_dy[j];
                    }
                }
            }
        }
        // flow out boundary
        for ibnd in self.mesh().flow_out_bnds.iter() {
            let iedges = &ibnd.iedges;
            for &iedge in iedges.iter() {
                let edge = &self.mesh().edges[iedge];
                let ielem = edge.parents[0];
                let elem = &self.mesh().elements[ielem];
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| self.mesh().nodes[inodes[i]].x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| self.mesh().nodes[inodes[i]].y);
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
                        0.5 * edge_length * edge_weights[i] * transformed_flux;

                    let row_idx = ielem * ncell_basis + itest_func;
                    for j in 0..3 {
                        dx[(row_idx, inodes[j])] +=
                            0.5 * edge_length * edge_weights[i] * dtransformed_flux_dx[j];
                        dy[(row_idx, inodes[j])] +=
                            0.5 * edge_length * edge_weights[i] * dtransformed_flux_dy[j];
                    }

                    if is_enriched {
                        for (j, &isol_node) in sol_nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .indexed_iter()
                        {
                            let col_idx = ielem * unenriched_ncell_basis + isol_node;
                            dsol[(row_idx, col_idx)] += 0.5
                                * edge_length
                                * edge_weights[i]
                                * self.interp_node_to_enriched_quadrature()[(i, j)]
                                * dtransformed_flux_du;
                        }
                    } else {
                        let sol_nodes = sol_nodes_along_edges.slice(s![local_ids[0], ..]);
                        let col_idx = ielem * ncell_basis + sol_nodes[i];
                        dsol[(row_idx, col_idx)] +=
                            0.5 * edge_length * edge_weights[i] * dtransformed_flux_du;
                    }
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
    fn compute_boundary_flux(&self, u: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64;
    fn dbnd_flux(
        &self,
        u: f64,
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
    fn physical_flux(&self, u: f64) -> [f64; 2];
    fn initialize_solution(&self, solutions: ArrayViewMut2<f64>);
    fn set_initial_solution(&self, initial_guess: ArrayView2<f64>) -> Array2<f64> {
        let ncell_basis = self.basis().r.len();
        let nelem = self.mesh().elem_num;
        let mut solutions = Array2::zeros((nelem, ncell_basis));

        for ielem in 0..nelem {
            let p0_val = initial_guess[[ielem, 0]];
            // For a nodal basis, the P0 solution is represented by setting all nodal
            // values within the element to the same constant value.
            solutions.slice_mut(s![ielem, ..]).fill(p0_val);
        }
        solutions
    }
}
pub trait SQP: P0Solver + SpaceTimeSolver1DScalar {
    fn compute_node_constraints(&self, new_to_old: &Vec<usize>) -> Array2<f64> {
        let n_nodes = self.mesh().node_num;
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
        node_constraints: ArrayView2<f64>,
        res: ArrayView2<f64>,
        hessian_uu: ArrayView2<f64>,
        hessian_ux: ArrayView2<f64>,
        hessian_xx: ArrayView2<f64>,
        dsol: ArrayView2<f64>,
        dcoord: ArrayView2<f64>,
        obj_dsol: ArrayView1<f64>,
        obj_dcoord: ArrayView1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let nelem = self.mesh().elem_num;
        let ncell_basis = self.basis().r.len();
        let free_coords = &self.mesh().free_bnd_x.len() + &self.mesh().free_bnd_y.len();
        let interior_nnodes = self.mesh().interior_nodes.len();
        let num_u = nelem * ncell_basis;
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
            .assign(&(&obj_dsol * -1.0));
        b_ndarray
            .slice_mut(s![num_u..num_u + num_x])
            .assign(&(&obj_dcoord * -1.0));
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
    fn solve(&mut self, mut solutions: ArrayViewMut2<f64>) {
        let initial_guess = self.compute_initial_guess();
        let initial_solutions = self.set_initial_solution(initial_guess.view());
        solutions.assign(&initial_solutions);
        let nelem = self.mesh().elem_num;
        let nnode = self.mesh().node_num;
        let ncell_basis = self.basis().r.len();
        let enriched_ncell_basis = self.enriched_basis().r.len();
        let epsilon1 = 1e-10;
        let epsilon2 = 1e-12;
        let max_line_search_iter = 20;
        let max_sqp_iter = 30;
        // let free_coords = &self.mesh.free_coords;
        // println!("free_coords: {:?}", free_coords);
        let (_old_to_new, new_to_old) = self.mesh().rearrange_node_dofs();
        let node_constraints = self.compute_node_constraints(&new_to_old);

        let mut residuals: Array2<f64> = Array2::zeros((nelem, ncell_basis));
        let mut dsol: Array2<f64> = Array2::zeros((nelem * ncell_basis, nelem * ncell_basis));
        let mut dx: Array2<f64> = Array2::zeros((nelem * ncell_basis, nnode));
        let mut dy: Array2<f64> = Array2::zeros((nelem * ncell_basis, nnode));
        let mut enriched_residuals: Array2<f64> = Array2::zeros((nelem, enriched_ncell_basis));
        let mut enriched_dsol: Array2<f64> =
            Array2::zeros((nelem * enriched_ncell_basis, nelem * ncell_basis));
        let mut enriched_dx: Array2<f64> = Array2::zeros((nelem * enriched_ncell_basis, nnode));
        let mut enriched_dy: Array2<f64> = Array2::zeros((nelem * enriched_ncell_basis, nnode));

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
            println!("Solutions:");
            for row in solutions.rows() {
                let row_str: Vec<String> = row.iter().map(|&val| format!("{:.4}", val)).collect();
                println!("[{}]", row_str.join(", "));
            }
            self.mesh().print_free_node_coords();
            self.compute_residuals_and_derivatives(
                solutions.view(),
                residuals.view_mut(),
                dsol.view_mut(),
                dx.view_mut(),
                dy.view_mut(),
                false,
            );
            self.compute_residuals_and_derivatives(
                solutions.view(),
                enriched_residuals.view_mut(),
                enriched_dsol.view_mut(),
                enriched_dx.view_mut(),
                enriched_dy.view_mut(),
                true,
            );
            let dcoord = concatenate(Axis(1), &[dx.view(), dy.view()]).unwrap();
            let dobj_dsol = enriched_dsol.t().dot(&enriched_residuals.flatten());
            let enriched_dcoord =
                concatenate(Axis(1), &[enriched_dx.view(), enriched_dy.view()]).unwrap();
            let dobj_dcoord = enriched_dcoord
                .t()
                .dot(&enriched_residuals.flatten())
                .dot(&node_constraints);
            let dsol_faer = dsol.view().into_faer();
            let dsol_inv = dsol_faer.partial_piv_lu().inverse();
            let dsol_inv_t = dsol_inv.transpose().into_ndarray();
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
            let feasibility_norm = residuals.mapv(|x| x.powi(2)).sum().sqrt();
            println!("optimality: {:?}", optimality_norm);
            println!("feasibility: {:?}", feasibility_norm);
            if optimality_norm < epsilon1 && feasibility_norm < epsilon2 {
                println!("Terminating SQP at iter: {:?}", iter);
                break;
            }
            let hessian_uu = enriched_dsol.t().dot(&enriched_dsol);
            let hessian_ux = enriched_dsol
                .t()
                .dot(&enriched_dcoord.dot(&node_constraints));
            let mut hessian_xx = enriched_dcoord
                .dot(&node_constraints)
                .t()
                .dot(&enriched_dcoord.dot(&node_constraints));
            hessian_xx += &(1e-5 * &Array2::eye(hessian_xx.shape()[0]));

            let (delta_u, delta_x) = self.solve_linear_subproblem(
                node_constraints.view(),
                residuals.view(),
                hessian_uu.view(),
                hessian_ux.view(),
                hessian_xx.view(),
                dsol.view(),
                dcoord.view(),
                dobj_dsol.view(),
                dobj_dcoord.view(),
            );
            // backtracking line search
            let merit_func = |alpha: f64| -> f64 {
                let mut tmp_mesh = self.mesh().clone();
                let delta_u_ndarray = Array::from_iter(delta_u.iter().copied());
                let u_flat = &solutions.flatten() + alpha * &delta_u_ndarray;
                let u = u_flat.to_shape((nelem, ncell_basis)).unwrap();
                tmp_mesh.update_node_coords(&new_to_old, alpha, delta_x.view());
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
            solutions.scaled_add(alpha, &delta_u.to_shape(solutions.shape()).unwrap());
            self.mesh_mut()
                .update_node_coords(&new_to_old, alpha, delta_x.view());
            iter += 1;
        }
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
