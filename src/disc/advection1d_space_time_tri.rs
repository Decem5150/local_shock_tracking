mod flux;

use faer::{Col, Mat, linalg::solvers::DenseSolveCore, mat, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use flux::space_time_flux1d;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, array, s};
use ndarray_linalg::Inverse;
use ndarray_stats::QuantileExt;
use std::{autodiff::autodiff_reverse, thread::LocalKey};

use super::{
    basis::triangle::TriangleBasis,
    mesh::mesh2d::{Mesh2d, TriangleElement},
};
use crate::solver::SolverParameters;

fn compute_normal(x0: f64, y0: f64, x1: f64, y1: f64) -> [f64; 2] {
    // normalized normal vector
    let normal = [y1 - y0, x0 - x1];
    let normal_magnitude = (normal[0].powi(2) + normal[1].powi(2)).sqrt();
    [normal[0] / normal_magnitude, normal[1] / normal_magnitude]
}
fn compute_ref_normal(local_id: usize) -> [f64; 2] {
    match local_id {
        0 => {
            // Bottom edge: from (0,0) to (1,0)
            // Outward normal points downward
            [0.0, -1.0]
        }
        1 => {
            // Hypotenuse edge: from (1,0) to (0,1)
            // Edge vector: (-1, 1), normal: (1, 1) normalized
            let sqrt2_inv = 1.0 / (2.0_f64.sqrt());
            [sqrt2_inv, sqrt2_inv]
        }
        2 => {
            // Left edge: from (0,1) to (0,0)
            // Outward normal points leftward
            [-1.0, 0.0]
        }
        _ => {
            panic!("Invalid edge ID");
        }
    }
}
fn compute_ref_edge_length(local_id: usize) -> f64 {
    match local_id {
        0 => 2.0,
        1 => 2.0 * 2.0_f64.sqrt(),
        2 => 2.0,
        _ => panic!("Invalid edge ID"),
    }
}

pub struct Disc1dAdvectionSpaceTimeTri<'a> {
    basis: TriangleBasis,
    enriched_basis: TriangleBasis,
    interp_node_to_cubature: Array2<f64>,
    interp_node_to_enriched_cubature: Array2<f64>,
    interp_enriched_node_to_cubature: Array2<f64>,
    interp_node_to_enriched_quadrature: Array2<f64>,
    dnodal_dr_at_cubature: Array2<f64>,
    dnodal_ds_at_cubature: Array2<f64>,
    enriched_basis_at_cubature: Array2<f64>,
    denriched_basis_dr_at_cubature: Array2<f64>,
    denriched_basis_ds_at_cubature: Array2<f64>,
    pub mesh: &'a mut Mesh2d<TriangleElement>,
    solver_param: &'a SolverParameters,
    advection_speed: f64,
}
fn evaluate_jacob(_xi: f64, _eta: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]) {
    // For triangular elements with reference triangle vertices at:
    // Node 0: (-1, -1)
    // Node 1: (1, -1)
    // Node 2: (-1, 1)
    // Shape functions for linear triangle:
    // N0 = -(xi + eta)/2     (node 0)
    // N1 = (1 + xi)/2        (node 1)
    // N2 = (1 + eta)/2       (node 2)

    let dn_dxi = [
        -0.5, // dN0/dξ
        0.5,  // dN1/dξ
        0.0,  // dN2/dξ
    ];
    let dn_deta = [
        -0.5, // dN0/dη
        0.0,  // dN1/dη
        0.5,  // dN2/dη
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
        basis.validate_modal_derivatives(solver_param.polynomial_order, 1e-6);
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
        let (dnodal_dr_at_cubature, dnodal_ds_at_cubature) = TriangleBasis::dmatrices_2d(
            solver_param.polynomial_order,
            basis.cub_r.view(),
            basis.cub_s.view(),
            basis.vandermonde.view(), // V computed at nodal points
        );
        // println!("dnodal_dr_at_cubature: {:?}", dnodal_dr_at_cubature);
        // println!("dnodal_ds_at_cubature: {:?}", dnodal_ds_at_cubature);
        let enriched_basis_at_cubature =
            interp_enriched_node_to_cubature.dot(&enriched_basis.vandermonde);
        let (denriched_basis_dr_at_cubature, denriched_basis_ds_at_cubature) =
            TriangleBasis::grad_vandermonde_2d(
                solver_param.polynomial_order + 1,
                enriched_basis.cub_r.view(),
                enriched_basis.cub_s.view(),
            );
        println!("=== Testing derivative sum property ===");
        let mut total_sum_dxi = 0.0;
        let mut total_sum_deta = 0.0;

        let ncell_basis = basis.r.len();
        let weights = &basis.cub_w;
        let ngp = basis.cub_r.len();
        for itest_func in 0..ncell_basis {
            let mut total_dxi = 0.0;
            let mut total_deta = 0.0;
            for igp in 0..ngp {
                let dtest_func_dxi = dnodal_dr_at_cubature[(igp, itest_func)];
                let dtest_func_deta = dnodal_ds_at_cubature[(igp, itest_func)];
                total_dxi += weights[igp] * dtest_func_dxi;
                total_deta += weights[igp] * dtest_func_deta;
            }
            total_sum_dxi += total_dxi;
            total_sum_deta += total_deta;
        }
        println!(
            "Sum over all basis functions: [{:.6e}, {:.6e}]",
            total_sum_dxi, total_sum_deta
        );

        println!("=== Checking cubature weight sum ===");
        let total_weight: f64 = basis.cub_w.iter().sum();
        println!("Sum of cubature weights: {}", total_weight);
        println!("Expected (triangle area): 2.0");
        println!("Error: {:.2e}", (total_weight - 2.0).abs());

        println!("=== Checking derivative matrix property ===");
        let ngp = basis.cub_r.len();
        let ncell_basis = basis.r.len();
        let weights = &basis.cub_w;

        for itest_func in 0..ncell_basis {
            let mut total_dxi = 0.0;
            let mut total_deta = 0.0;
            for igp in 0..ngp {
                let dtest_func_dxi = dnodal_dr_at_cubature[(igp, itest_func)];
                let dtest_func_deta = dnodal_ds_at_cubature[(igp, itest_func)];
                total_dxi += weights[igp] * dtest_func_dxi;
                total_deta += weights[igp] * dtest_func_deta;
            }
            println!(
                "Basis {}: ∫∇v dΩ = [{:.6e}, {:.6e}]",
                itest_func, total_dxi, total_deta
            );
        }
        /*
        println!("=== Testing modal basis orthogonality (Mass Matrix) ===");
        // The modal basis functions are rows of V.
        // We want to check ∫ φ_i φ_j dΩ.
        // φ_i at cubature points are given by interp_node_to_cubature.row(i)
        // Mass matrix M_ij = Σ_k w_k * φ_i(r_k, s_k) * φ_j(r_k, s_k)
        // M = interp_node_to_cubature^T * W * interp_node_to_cubature
        // where W is a diagonal matrix with cubature weights.
        // For an orthogonal basis, M should be diagonal.

        let modal_basis_at_cubature = TriangleBasis::vandermonde2d(
            solver_param.polynomial_order,
            basis.cub_r.view(),
            basis.cub_s.view(),
        );

        let mut modal_mass_matrix = Array2::<f64>::zeros((ncell_basis, ncell_basis));
        for i in 0..ncell_basis {
            for j in 0..ncell_basis {
                let mut integral = 0.0;
                for k in 0..ngp {
                    integral += weights[k]
                        * modal_basis_at_cubature[(k, i)]
                        * modal_basis_at_cubature[(k, j)];
                }
                modal_mass_matrix[(i, j)] = integral;
            }
        }

        println!("Mass Matrix M:\n{:?}", modal_mass_matrix);
        */
        Disc1dAdvectionSpaceTimeTri {
            basis,
            enriched_basis,
            interp_node_to_cubature,
            interp_node_to_enriched_cubature,
            interp_enriched_node_to_cubature,
            interp_node_to_enriched_quadrature,
            dnodal_dr_at_cubature,
            dnodal_ds_at_cubature,
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
    pub fn solve(&mut self, mut solutions: ArrayViewMut2<f64>) {
        let nelem = self.mesh.elem_num;
        let ncell_basis = self.basis.r.len();
        let mut residuals: Array2<f64> = Array2::zeros((nelem, ncell_basis));
        self.compute_residuals(self.mesh, solutions.view(), residuals.view_mut(), false);
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
            println!("ielem: {}", ielem);
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
            println!("interp_sol: {:?}", interp_sol);
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
        println!("residuals_after_volume_integral: {:?}", residuals);
        for &iedge in mesh.internal_edges.iter() {
            println!("iedge: {}", iedge);
            let edge = &mesh.edges[iedge];
            let ilelem = edge.parents[0];
            println!("ilelem: {}", ilelem);
            let irelem = edge.parents[1];
            println!("irelem: {}", irelem);
            let left_elem = &mesh.elements[ilelem];
            let right_elem = &mesh.elements[irelem];
            let left_inodes = &left_elem.inodes;
            let right_inodes = &right_elem.inodes;
            let left_x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[left_inodes[i]].x);
            let left_y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[left_inodes[i]].y);
            let right_x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[right_inodes[i]].x);
            let right_y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[right_inodes[i]].y);
            println!("left_x_slice: {:?}", left_x_slice);
            println!("left_y_slice: {:?}", left_y_slice);
            println!("right_x_slice: {:?}", right_x_slice);
            println!("right_y_slice: {:?}", right_y_slice);
            let common_edge = [edge.local_ids[0], edge.local_ids[1]];
            println!("common_edge: {:?}", common_edge);
            let node_along_edges = &basis.nodes_along_edges;
            let local_ids = &edge.local_ids;
            let left_ref_normal = compute_ref_normal(local_ids[0]);
            let right_ref_normal = compute_ref_normal(local_ids[1]);
            let left_edge_length = compute_ref_edge_length(local_ids[0]);
            let right_edge_length = compute_ref_edge_length(local_ids[1]);
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
                let normal = compute_normal(
                    left_x_slice[common_edge[0]],
                    left_y_slice[common_edge[0]],
                    left_x_slice[(common_edge[0] + 1) % 3],
                    left_y_slice[(common_edge[0] + 1) % 3],
                );
                println!("normal: {:?}", normal);
                let num_flux = self.compute_numerical_flux(
                    self.advection_speed,
                    left_value,
                    right_value,
                    left_x_slice[common_edge[0]],
                    left_x_slice[(common_edge[0] + 1) % 3],
                    left_y_slice[common_edge[0]],
                    left_y_slice[(common_edge[0] + 1) % 3],
                );
                println!("num_flux: {}", num_flux);
                println!("left_xi_slice[i]: {}", left_xi_slice[i]);
                println!("left_eta_slice[i]: {}", left_eta_slice[i]);
                let left_scaling = self.compute_flux_scaling(
                    left_xi_slice[i],
                    left_eta_slice[i],
                    left_ref_normal,
                    &left_x_slice,
                    &left_y_slice,
                );
                println!(
                    "right_xi_slice[nedge_basis - 1 - i]: {}",
                    right_xi_slice[nedge_basis - 1 - i]
                );
                println!(
                    "right_eta_slice[nedge_basis - 1 - i]: {}",
                    right_eta_slice[nedge_basis - 1 - i]
                );
                let right_scaling = self.compute_flux_scaling(
                    right_xi_slice[nedge_basis - 1 - i],
                    right_eta_slice[nedge_basis - 1 - i],
                    right_ref_normal,
                    &right_x_slice,
                    &right_y_slice,
                );
                println!("left_scaling: {}", left_scaling);
                println!("right_scaling: {}", right_scaling);
                let left_transformed_flux = num_flux * left_scaling;
                let right_transformed_flux = -num_flux * right_scaling;
                println!("left_transformed_flux: {}", left_transformed_flux);
                println!("right_transformed_flux: {}", right_transformed_flux);
                let left_itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                let right_itest_func = basis.nodes_along_edges[(local_ids[1], nedge_basis - 1 - i)];
                residuals[(ilelem, left_itest_func)] +=
                    0.5 * left_edge_length * edge_weights[i] * left_transformed_flux;
                residuals[(irelem, right_itest_func)] +=
                    0.5 * right_edge_length * edge_weights[i] * right_transformed_flux;
            }
        }
        println!("residuals_after_edge_integral: {:?}", residuals);
        // flow in boundary
        println!("flow in boundary");
        for ibnd in mesh.flow_in_bnds.iter() {
            let iedges = &ibnd.iedges;
            let value = ibnd.value;
            for &iedge in iedges.iter() {
                println!("iedge: {}", iedge);
                let edge = &mesh.edges[iedge];
                let ielem = edge.parents[0];
                println!("ielem: {}", ielem);
                let elem = &mesh.elements[ielem];
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].y);
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_ids = &edge.local_ids;
                let ref_normal = compute_ref_normal(local_ids[0]);
                let edge_length = compute_ref_edge_length(local_ids[0]);
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
                        self.advection_speed,
                        value,
                        x_slice[local_ids[0]],
                        x_slice[(local_ids[0] + 1) % 3],
                        y_slice[local_ids[0]],
                        y_slice[(local_ids[0] + 1) % 3],
                    );
                    println!("boundary_flux: {}", boundary_flux);
                    let scaling =
                        self.compute_flux_scaling(xi, eta, ref_normal, &x_slice, &y_slice);
                    println!("scaling: {}", scaling);
                    let transformed_flux = boundary_flux * scaling;
                    println!("transformed_flux: {}", transformed_flux);
                    let itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                    println!("itest_func: {}", itest_func);
                    residuals[(ielem, itest_func)] +=
                        0.5 * edge_length * edge_weights[i] * transformed_flux;
                }
            }
        }
        // flow out boundary
        println!("flow out boundary");
        for ibnd in mesh.flow_out_bnds.iter() {
            let iedges = &ibnd.iedges;
            for &iedge in iedges.iter() {
                println!("iedge: {}", iedge);
                let edge = &mesh.edges[iedge];
                let ielem = edge.parents[0];
                println!("ielem: {}", ielem);
                let elem = &mesh.elements[ielem];
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| mesh.nodes[inodes[i]].y);
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_ids = &edge.local_ids;
                let ref_normal = compute_ref_normal(local_ids[0]);
                let edge_length = compute_ref_edge_length(local_ids[0]);
                let sol_slice = solutions.slice(s![ielem, ..]).select(
                    Axis(0),
                    nodes_along_edges
                        .slice(s![local_ids[0], ..])
                        .as_slice()
                        .unwrap(),
                );
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
                    println!("i: {}", i);
                    let xi = xi_slice[i];
                    let eta = eta_slice[i];
                    let boundary_flux = self.compute_boundary_flux(
                        self.advection_speed,
                        sol_slice[i],
                        x_slice[local_ids[0]],
                        x_slice[(local_ids[0] + 1) % 3],
                        y_slice[local_ids[0]],
                        y_slice[(local_ids[0] + 1) % 3],
                    );
                    println!("boundary_flux: {}", boundary_flux);
                    let scaling =
                        self.compute_flux_scaling(xi, eta, ref_normal, &x_slice, &y_slice);
                    println!("scaling: {}", scaling);
                    let transformed_flux = boundary_flux * scaling;
                    println!("transformed_flux: {}", transformed_flux);
                    let itest_func = basis.nodes_along_edges[(local_ids[0], i)];
                    println!("itest_func: {}", itest_func);
                    residuals[(ielem, itest_func)] +=
                        0.5 * edge_length * edge_weights[i] * transformed_flux;
                }
            }
        }
        println!("residuals_after_boundary: {:?}", residuals);
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
            let dtest_func_dxi = self.dnodal_dr_at_cubature[(igp, itest_func)];
            let dtest_func_deta = self.dnodal_ds_at_cubature[(igp, itest_func)];
            /*
            println!("itest_func: {}, sol values: {:?}", itest_func, sol);
            println!("flux values: {:?}", f);
            println!("transformed_f: {:?}", transformed_f);
            println!("derivatives: dξ={}, dη={}", dtest_func_dxi, dtest_func_deta);
            */
            res -= weights[igp] * transformed_f[0] * dtest_func_dxi
                + weights[igp] * transformed_f[1] * dtest_func_deta;
        }
        let mut total_dxi = 0.0;
        let mut total_deta = 0.0;
        for igp in 0..ngp {
            let dtest_func_dxi = self.dnodal_dr_at_cubature[(igp, itest_func)];
            let dtest_func_deta = self.dnodal_ds_at_cubature[(igp, itest_func)];
            total_dxi += weights[igp] * dtest_func_dxi;
            total_deta += weights[igp] * dtest_func_deta;
        }
        println!(
            "For test func {}: ∫∇v·dΩ = [{}, {}]",
            itest_func, total_dxi, total_deta
        );
        res
    }
    fn compute_boundary_flux(
        &self,
        advection_speed: f64,
        u: f64,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    ) -> f64 {
        let normal = compute_normal(x0, y0, x1, y1);
        let beta = [advection_speed, 1.0];
        let beta_dot_normal = beta[0] * normal[0] + beta[1] * normal[1];
        let result = beta_dot_normal * u;
        result
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
        println!("jacob_det: {}", jacob_det);
        println!("jacob_inv_t: {:?}", jacob_inv_t);
        println!("ref_normal: {:?}", ref_normal);
        let transformed_normal = {
            [
                jacob_inv_t[0] * ref_normal[0] + jacob_inv_t[1] * ref_normal[1],
                jacob_inv_t[2] * ref_normal[0] + jacob_inv_t[3] * ref_normal[1],
            ]
        };
        println!("transformed_normal: {:?}", transformed_normal);
        let normal_magnitude =
            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
        println!("normal_magnitude: {}", normal_magnitude);
        jacob_det * normal_magnitude
    }
    pub fn initialize_solution(&mut self, mut solutions: ArrayViewMut2<f64>) {
        solutions.slice_mut(s![0, ..]).fill(2.0);
        solutions.slice_mut(s![1, ..]).fill(2.0);
        solutions.slice_mut(s![2, ..]).fill(1.0);
        solutions.slice_mut(s![3, ..]).fill(1.0);
    }
}
