mod flux;
mod p0solver;

use faer::{Col, Mat, linalg::solvers::DenseSolveCore, mat, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use flux::space_time_flux1d;
use ndarray::{
    Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, array, concatenate, s,
};
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
                let dtest_func_dxi = basis.dr_cub[(igp, itest_func)];
                let dtest_func_deta = basis.ds_cub[(igp, itest_func)];
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
                let dtest_func_dxi = basis.dr_cub[(igp, itest_func)];
                let dtest_func_deta = basis.ds_cub[(igp, itest_func)];
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
            enriched_basis_at_cubature,
            denriched_basis_dr_at_cubature,
            denriched_basis_ds_at_cubature,
            mesh,
            solver_param,
            advection_speed: 0.8,
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
        let nelem = self.mesh.elem_num;
        let ncell_basis = self.basis.r.len();
        let free_coords = &self.mesh.free_coords;
        let interior_nnodes = self.mesh.interior_node_num;
        let num_u = nelem * ncell_basis;
        let num_x: usize = free_coords.len() + 2 * interior_nnodes;
        println!("num_u: {:?}", num_u);
        println!("num_x: {:?}", num_x);
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
    pub fn solve(&mut self, mut solutions: ArrayViewMut2<f64>) {
        let nelem = self.mesh.elem_num;
        let nnode = self.mesh.node_num;
        let ncell_basis = self.basis.r.len();
        let enriched_ncell_basis = self.enriched_basis.r.len();
        let epsilon1 = 1e-10;
        let epsilon2 = 1e-15;
        let max_line_search_iter = 20;
        let max_sqp_iter = 30;
        let interior_nnodes = self.mesh.interior_node_num;
        let free_coords = &self.mesh.free_coords;
        // println!("free_coords: {:?}", free_coords);
        let mut node_constraints: Array2<f64> =
            Array2::zeros((2 * nnode, 2 * interior_nnodes + free_coords.len()));
        // node_constraints[(4, 0)] = 1.0;
        node_constraints[(14, 5)] = 1.0;

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
            println!("solutions: {:?}", solutions);
            println!("nodes: {:?}", self.mesh.nodes[free_coords[3] % nnode].y);
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
            hessian_xx += &(1e-5 * &Array2::eye(2 * interior_nnodes + free_coords.len()));

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
                let mut tmp_mesh = self.mesh.clone();
                let delta_u_ndarray = Array::from_iter(delta_u.iter().copied());
                let u_flat = &solutions.flatten() + alpha * &delta_u_ndarray;
                let u = u_flat.to_shape((nelem, ncell_basis)).unwrap();
                for (i, &ix) in free_coords.iter().enumerate() {
                    if ix < nnode {
                        tmp_mesh.nodes[ix].x += alpha * delta_x[2 * interior_nnodes + i];
                    } else {
                        tmp_mesh.nodes[ix - nnode].y += alpha * delta_x[2 * interior_nnodes + i];
                    }
                }
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
            for (i, &ix) in free_coords.iter().enumerate() {
                if ix < nnode {
                    self.mesh.nodes[ix].x += alpha * delta_x[2 * interior_nnodes + i];
                } else {
                    self.mesh.nodes[ix - nnode].y += alpha * delta_x[2 * interior_nnodes + i];
                }
            }
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
            let sol_nodes_along_edges = &self.basis.nodes_along_edges;
            let nodes_along_edges = &basis.nodes_along_edges;
            let local_ids = &edge.local_ids;
            let left_ref_normal = compute_ref_normal(local_ids[0]);
            let right_ref_normal = compute_ref_normal(local_ids[1]);
            let left_edge_length = compute_ref_edge_length(local_ids[0]);
            let right_edge_length = compute_ref_edge_length(local_ids[1]);
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
                        self.interp_node_to_enriched_quadrature.dot(&left_sol_slice),
                        self.interp_node_to_enriched_quadrature
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
                let normal = compute_normal(
                    left_x_slice[common_edge[0]],
                    left_y_slice[common_edge[0]],
                    left_x_slice[(common_edge[0] + 1) % 3],
                    left_y_slice[(common_edge[0] + 1) % 3],
                );
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
                let sol_nodes_along_edges = &self.basis.nodes_along_edges;
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_ids = &edge.local_ids;
                let ref_normal = compute_ref_normal(local_ids[0]);
                let edge_length = compute_ref_edge_length(local_ids[0]);
                let sol_slice = {
                    let sol_slice = solutions.slice(s![ielem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    if is_enriched {
                        self.interp_node_to_enriched_quadrature.dot(&sol_slice)
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
                        self.advection_speed,
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
        let nelem = self.mesh.elem_num;
        let nnode = self.mesh.node_num;
        let basis = {
            if is_enriched {
                &self.enriched_basis
            } else {
                &self.basis
            }
        };
        let unenriched_ncell_basis = self.basis.r.len();
        let unenriched_nedge_basis = self.basis.quad_p.len();
        let ncell_basis = basis.r.len();
        let nedge_basis = basis.quad_p.len();
        let cell_weights = &basis.cub_w;
        let edge_weights = &basis.quad_w;
        for (ielem, elem) in self.mesh.elements.iter().enumerate() {
            let inodes = &elem.inodes;
            let x_slice: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
            let y_slice: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].y);
            let interp_matrix = if is_enriched {
                &self.interp_node_to_enriched_cubature
            } else {
                &self.interp_node_to_cubature
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
        for &iedge in self.mesh.internal_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let ilelem = edge.parents[0];
            let irelem = edge.parents[1];
            let left_elem = &self.mesh.elements[ilelem];
            let right_elem = &self.mesh.elements[irelem];
            let left_inodes = &left_elem.inodes;
            let right_inodes = &right_elem.inodes;
            let left_x_slice: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[left_inodes[i]].x);
            let left_y_slice: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[left_inodes[i]].y);
            let right_x_slice: [f64; 3] =
                std::array::from_fn(|i| self.mesh.nodes[right_inodes[i]].x);
            let right_y_slice: [f64; 3] =
                std::array::from_fn(|i| self.mesh.nodes[right_inodes[i]].y);
            let common_edge = [edge.local_ids[0], edge.local_ids[1]];
            let sol_nodes_along_edges = &self.basis.nodes_along_edges;
            let nodes_along_edges = &basis.nodes_along_edges;
            let local_ids = &edge.local_ids;
            let left_ref_normal = compute_ref_normal(local_ids[0]);
            let right_ref_normal = compute_ref_normal(local_ids[1]);
            let left_edge_length = compute_ref_edge_length(local_ids[0]);
            let right_edge_length = compute_ref_edge_length(local_ids[1]);
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
                        self.interp_node_to_enriched_quadrature.dot(&left_sol_slice),
                        self.interp_node_to_enriched_quadrature
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
                    self.advection_speed,
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
                            * self.interp_node_to_enriched_quadrature[(i, j)]
                            * dleft_transformed_flux_dul;
                        dsol[(row_idx_right, col_idx)] += 0.5
                            * right_edge_length
                            * edge_weights[i]
                            * self.interp_node_to_enriched_quadrature[(i, j)]
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
                            * self.interp_node_to_enriched_quadrature[(nedge_basis - 1 - i, j)]
                            * dleft_transformed_flux_dur;
                        dsol[(row_idx_right, col_idx)] += 0.5
                            * right_edge_length
                            * edge_weights[i]
                            * self.interp_node_to_enriched_quadrature[(nedge_basis - 1 - i, j)]
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
        for ibnd in self.mesh.flow_in_bnds.iter() {
            let iedges = &ibnd.iedges;
            let value = ibnd.value;
            for &iedge in iedges.iter() {
                let edge = &self.mesh.edges[iedge];
                let ielem = edge.parents[0];
                let elem = &self.mesh.elements[ielem];
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].y);
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
                    let (boundary_flux, _dflux_dbnd, dflux_dx0, dflux_dx1, dflux_dy0, dflux_dy1) =
                        self.dbnd_flux(
                            self.advection_speed,
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
                    let scaling = self.dscaling(
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

                    let dtransformed_flux_dx = &ArrayView1::from(&dflux_dx) * scaling
                        + &ArrayView1::from(&dscaling_dx) * boundary_flux;
                    let dtransformed_flux_dy = &ArrayView1::from(&dflux_dy) * scaling
                        + &ArrayView1::from(&dscaling_dy) * boundary_flux;

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
        for ibnd in self.mesh.flow_out_bnds.iter() {
            let iedges = &ibnd.iedges;
            for &iedge in iedges.iter() {
                let edge = &self.mesh.edges[iedge];
                let ielem = edge.parents[0];
                let elem = &self.mesh.elements[ielem];
                let inodes = &elem.inodes;
                let x_slice: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
                let y_slice: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].y);
                let sol_nodes_along_edges = &self.basis.nodes_along_edges;
                let nodes_along_edges = &basis.nodes_along_edges;
                let local_ids = &edge.local_ids;
                let ref_normal = compute_ref_normal(local_ids[0]);
                let edge_length = compute_ref_edge_length(local_ids[0]);
                let sol_slice = {
                    let sol_slice = solutions.slice(s![ielem, ..]).select(
                        Axis(0),
                        sol_nodes_along_edges
                            .slice(s![local_ids[0], ..])
                            .as_slice()
                            .unwrap(),
                    );
                    if is_enriched {
                        self.interp_node_to_enriched_quadrature.dot(&sol_slice)
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
                    let (boundary_flux, dflux_du, dflux_dx0, dflux_dx1, dflux_dy0, dflux_dy1) =
                        self.dbnd_flux(
                            self.advection_speed,
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
                    let scaling = self.dscaling(
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
                    let dtransformed_flux_dx = &ArrayView1::from(&dflux_dx) * scaling
                        + &ArrayView1::from(&dscaling_dx) * boundary_flux;
                    let dtransformed_flux_dy = &ArrayView1::from(&dflux_dy) * scaling
                        + &ArrayView1::from(&dscaling_dy) * boundary_flux;

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
                                * self.interp_node_to_enriched_quadrature[(i, j)]
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
    #[autodiff_reverse(
        dvolume, Const, Const, Const, Duplicated, Duplicated, Duplicated, Active
    )]
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
            let dtest_func_dxi = basis.dr_cub[(igp, itest_func)];
            let dtest_func_deta = basis.ds_cub[(igp, itest_func)];
            /*
            println!("itest_func: {}, sol values: {:?}", itest_func, sol);
            println!("flux values: {:?}", f);
            println!("transformed_f: {:?}", transformed_f);
            println!("derivatives: dξ={}, dη={}", dtest_func_dxi, dtest_func_deta);
            */
            res -= weights[igp] * transformed_f[0] * dtest_func_dxi
                + weights[igp] * transformed_f[1] * dtest_func_deta;
        }
        res
    }
    #[autodiff_reverse(
        dbnd_flux, Const, Const, Active, Active, Active, Active, Active, Active
    )]
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
    #[autodiff_reverse(
        dnum_flux, Const, Const, Active, Active, Active, Active, Active, Active, Active
    )]
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
    #[autodiff_reverse(dscaling, Const, Const, Const, Const, Duplicated, Duplicated, Active)]
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
    pub fn initialize_solution(&mut self, mut solutions: ArrayViewMut2<f64>) {
        /*
        solutions.slice_mut(s![0, ..]).fill(4.0);
        solutions.slice_mut(s![1, ..]).fill(4.0);
        solutions.slice_mut(s![2, ..]).fill(0.0);
        solutions.slice_mut(s![3, ..]).fill(0.0);
        */
        solutions.slice_mut(s![0, ..]).fill(2.0);
        solutions.slice_mut(s![1, ..]).fill(2.0);
        solutions.slice_mut(s![2, ..]).fill(1.0);
        solutions.slice_mut(s![3, ..]).fill(1.0);

        solutions.slice_mut(s![4, ..]).fill(2.0);
        solutions.slice_mut(s![5, ..]).fill(2.0);
        solutions.slice_mut(s![6, ..]).fill(1.0);
        solutions.slice_mut(s![7, ..]).fill(1.0);
    }
}
