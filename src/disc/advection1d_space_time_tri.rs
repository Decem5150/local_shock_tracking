mod flux;

use faer::{Col, Mat, linalg::solvers::DenseSolveCore, mat, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use flux::space_time_flux1d;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, array, s};
use ndarray_linalg::Inverse;
use ndarray_stats::QuantileExt;
use std::autodiff::autodiff_reverse;

use super::{
    basis::triangle::TriangleBasis,
    mesh::mesh2d::{Mesh2d, TriangleElement},
};
use crate::solver::SolverParameters;
fn compute_normal(x0: f64, y0: f64, x1: f64, y1: f64) -> [f64; 2] {
    [y1 - y0, x0 - x1]
}

pub struct Disc1dAdvectionSpaceTimeTri<'a> {
    basis: TriangleBasis,
    enriched_basis: TriangleBasis,
    interp_node_to_cubature: Array2<f64>,
    interp_node_to_enriched_cubature: Array2<f64>,
    interp_enriched_node_to_cubature: Array2<f64>,
    interp_node_to_enriched_quadrature: Array2<f64>,
    basis_at_cubature: Array2<f64>,
    dbasis_dr_at_cubature: Array2<f64>,
    dbasis_ds_at_cubature: Array2<f64>,
    enriched_basis_at_cubature: Array2<f64>,
    denriched_basis_dr_at_cubature: Array2<f64>,
    denriched_basis_ds_at_cubature: Array2<f64>,
    pub mesh: &'a mut Mesh2d<TriangleElement>,
    solver_param: &'a SolverParameters,
    advection_speed: f64,
}
fn evaluate_jacob(_xi: f64, _eta: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]) {
    // For triangular elements with area coordinates (xi, eta)
    // The third coordinate is zeta = 1 - xi - eta
    // Shape functions for linear triangle:
    // N1 = 1 - xi - eta (node 0)
    // N2 = xi            (node 1)
    // N3 = eta           (node 2)

    let dn_dxi = [
        -1.0, // dN1/dξ
        1.0,  // dN2/dξ
        0.0,  // dN3/dξ
    ];
    let dn_deta = [
        -1.0, // dN1/dη
        0.0,  // dN2/dη
        1.0,  // dN3/dη
    ];

    let mut dx_dxi = 0.0;
    let mut dx_deta = 0.0;
    let mut dy_dxi = 0.0;
    let mut dy_deta = 0.0;

    for k in 0..3 {
        dx_dxi += dn_dxi[k] * x[k];
        dx_deta += dn_deta[k] * x[k];
        dy_dxi += dn_dxi[k] * y[k];
        dy_deta += dn_deta[k] * y[k];
    }

    let jacob_det = dx_dxi * dy_deta - dx_deta * dy_dxi;
    let jacob_inv_t = [
        dy_deta / jacob_det,
        -dy_dxi / jacob_det,
        -dx_deta / jacob_det,
        dx_dxi / jacob_det,
    ];

    (jacob_det, jacob_inv_t)
}
impl<'a> Disc1dAdvectionSpaceTimeTri<'a> {
    pub fn new(
        basis: TriangleBasis,
        enriched_basis: TriangleBasis,
        mesh: &'a mut Mesh2d<TriangleElement>,
        solver_param: &'a SolverParameters,
    ) -> Disc1dAdvectionSpaceTimeTri<'a> {
        let interp_node_to_cubature = Self::compute_interp_matrix_2d(
            solver_param.polynomial_order,
            basis.inv_vandermonde.view(),
            basis.cub_r.view(),
            basis.cub_s.view(),
        );
        let interp_node_to_enriched_cubature = Self::compute_interp_matrix_2d(
            solver_param.polynomial_order,
            basis.inv_vandermonde.view(),
            enriched_basis.cub_r.view(),
            enriched_basis.cub_s.view(),
        );
        let interp_enriched_node_to_cubature = Self::compute_interp_matrix_2d(
            solver_param.polynomial_order + 1,
            enriched_basis.inv_vandermonde.view(),
            enriched_basis.cub_r.view(),
            enriched_basis.cub_s.view(),
        );
        let gauss_lobatto_points = &basis.quad_p;
        let enriched_gauss_lobatto_points = &enriched_basis.quad_p;
        let inv_vandermonde_1d = TriangleBasis::vandermonde1d(
            solver_param.polynomial_order,
            gauss_lobatto_points.view(),
        )
        .inv()
        .unwrap();
        let interp_node_to_enriched_quadrature = Self::compute_interp_matrix_1d(
            solver_param.polynomial_order,
            inv_vandermonde_1d.view(),
            enriched_gauss_lobatto_points.view(),
        );
        let basis_at_cubature = interp_node_to_cubature.dot(&basis.vandermonde);
        let (dbasis_dr_at_cubature, dbasis_ds_at_cubature) = TriangleBasis::grad_vandermonde_2d(
            solver_param.polynomial_order,
            basis.cub_r.view(),
            basis.cub_s.view(),
        );
        let enriched_basis_at_cubature =
            interp_enriched_node_to_cubature.dot(&enriched_basis.vandermonde);
        let (denriched_basis_dr_at_cubature, denriched_basis_ds_at_cubature) =
            TriangleBasis::grad_vandermonde_2d(
                solver_param.polynomial_order + 1,
                enriched_basis.cub_r.view(),
                enriched_basis.cub_s.view(),
            );
        Disc1dAdvectionSpaceTimeTri {
            basis,
            enriched_basis,
            interp_node_to_cubature,
            interp_node_to_enriched_cubature,
            interp_enriched_node_to_cubature,
            interp_node_to_enriched_quadrature,
            basis_at_cubature,
            dbasis_dr_at_cubature,
            dbasis_ds_at_cubature,
            enriched_basis_at_cubature,
            denriched_basis_dr_at_cubature,
            denriched_basis_ds_at_cubature,
            mesh,
            solver_param,
            advection_speed: 0.1,
        }
    }
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
        let nelem = mesh.elem_num;
        let basis = {
            if is_enriched {
                &self.enriched_basis
            } else {
                &self.basis
            }
        };
        let ncell_basis = basis.r.len();
        let nedge_basis = basis.quad_p.len();
        let cell_weights = &basis.cub_w;
        let edge_weights = &basis.quad_w;
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            let inodes = &elem.inodes;
            let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].x);
            let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].y);
            let interp_sol = if is_enriched {
                self.interp_node_to_enriched_cubature
                    .dot(&solutions.slice(s![ielem, ..]))
            } else {
                self.interp_node_to_cubature
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
                residuals[(ielem, itest_func)] = res;
            }
        }
        for (iedge, edge) in mesh.edges.iter().enumerate() {
            let left_ref_normal = edge.ref_normal;
            let right_ref_normal = edge.ref_normal.map(|x| -x);
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
            let node_along_edges = &basis.nodes_along_edges;
            let local_ids = &edge.local_ids;
            let left_sol_slice = solutions.slice(s![ilelem, ..]).select(
                Axis(0),
                node_along_edges
                    .slice(s![local_ids[0], ..])
                    .as_slice()
                    .unwrap(),
            );
            let right_sol_slice = solutions.slice(s![irelem, ..]).select(
                Axis(0),
                node_along_edges
                    .slice(s![local_ids[1], ..])
                    .as_slice()
                    .unwrap(),
            );
            let left_xi_slice = basis.r.select(
                Axis(0),
                node_along_edges
                    .slice(s![local_ids[0], ..])
                    .as_slice()
                    .unwrap(),
            );
            let left_eta_slice = basis.s.select(
                Axis(0),
                node_along_edges
                    .slice(s![local_ids[0], ..])
                    .as_slice()
                    .unwrap(),
            );
            let right_xi_slice = basis.r.select(
                Axis(0),
                node_along_edges
                    .slice(s![local_ids[1], ..])
                    .as_slice()
                    .unwrap(),
            );
            let right_eta_slice = basis.s.select(
                Axis(0),
                node_along_edges
                    .slice(s![local_ids[1], ..])
                    .as_slice()
                    .unwrap(),
            );
            for i in 0..nedge_basis {
                let left_value = left_sol_slice[i];
                let right_value = right_sol_slice[nedge_basis - 1 - i];
                let num_flux = self.compute_numerical_flux(
                    self.advection_speed,
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
                residuals[(ilelem, left_itest_func)] += edge_weights[i] * left_transformed_flux;
                residuals[(irelem, right_itest_func)] += edge_weights[i] * right_transformed_flux;
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
    ) -> f64 {
        let ngp = basis.cub_r.len();
        let weights = &basis.cub_w;
        let mut res = 0.0;
        for igp in 0..ngp {
            let f = space_time_flux1d(sol[igp], self.advection_speed);
            let xi = basis.cub_r[igp];
            let eta = basis.cub_s[igp];
            let (jacob_det, jacob_inv_t) = evaluate_jacob(xi, eta, x, y);
            let transformed_f = {
                [
                    jacob_det * (f[0] * jacob_inv_t[0] + f[1] * jacob_inv_t[2]),
                    jacob_det * (f[0] * jacob_inv_t[1] + f[1] * jacob_inv_t[3]),
                ]
            };
            let dtest_func_dxi = self.dbasis_dr_at_cubature[(igp, itest_func)];
            let dtest_func_deta = self.dbasis_ds_at_cubature[(igp, itest_func)];
            res -= weights[igp] * transformed_f[0] * dtest_func_dxi
                + weights[igp] * transformed_f[1] * dtest_func_deta;
        }
        res
    }
    fn compute_numerical_flux(
        &self,
        advection_speed: f64,
        ul: f64,
        ur: f64,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    ) -> f64 {
        let normal = compute_normal(x0, y0, x1, y1);
        let beta = [advection_speed, 1.0];
        let beta_dot_normal = beta[0] * normal[0] + beta[1] * normal[1];
        let result = 0.5
            * (beta_dot_normal * (ul + ur)
                + (beta_dot_normal * (100.0 * beta_dot_normal).tanh()) * (ul - ur));
        result
    }
    fn compute_flux_scaling(
        &self,
        xi: f64,
        eta: f64,
        ref_normal: [f64; 2],
        x: &[f64],
        y: &[f64],
    ) -> f64 {
        let (jacob_det, jacob_inv_t) = evaluate_jacob(xi, eta, x, y);
        let transformed_normal = {
            [
                jacob_inv_t[0] * ref_normal[0] + jacob_inv_t[1] * ref_normal[1],
                jacob_inv_t[2] * ref_normal[0] + jacob_inv_t[3] * ref_normal[1],
            ]
        };
        let normal_magnitude =
            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
        jacob_det * normal_magnitude
    }
}
