pub mod boundary_condition;
mod flux;
mod riemann_solver;
mod shock_tracking;
use faer::{Col, Mat, linalg::solvers::DenseSolveCore, mat, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use flux::space_time_flux1d;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, array, s};
use ndarray_stats::QuantileExt;
use riemann_solver::smoothed_upwind;
use std::autodiff::autodiff;

use crate::solver::SolverParameters;

use super::{basis::lagrange1d::LagrangeBasis1DLobatto, mesh::mesh2d::Mesh2d};
fn compute_normal(x0: f64, y0: f64, x1: f64, y1: f64) -> [f64; 2] {
    [y1 - y0, x0 - x1]
}
fn evaluate_jacob(eta: f64, xi: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]) {
    let dn_dxi = [
        -0.25 * (1.0 - eta), // dN1/dξ
        0.25 * (1.0 - eta),  // dN2/dξ
        0.25 * (1.0 + eta),  // dN3/dξ
        -0.25 * (1.0 + eta), // dN4/dξ
    ];
    let dn_deta = [
        -0.25 * (1.0 - xi), // dN1/dη
        -0.25 * (1.0 + xi), // dN2/dη
        0.25 * (1.0 + xi),  // dN3/dη
        0.25 * (1.0 - xi),  // dN4/dη
    ];
    let mut dx_dxi = 0.0;
    let mut dx_deta = 0.0;
    let mut dy_dxi = 0.0;
    let mut dy_deta = 0.0;
    for k in 0..4 {
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
struct Sqp {
    c: f64,
    tau: f64,
    epsilon1: f64,
    epsilon2: f64,
    max_line_search_iter: usize,
    max_sqp_iter: usize,
}
impl Sqp {
    fn new(mesh: &Mesh2d) -> Self {
        Self {
            c: 1e-4,
            tau: 0.5,
            epsilon1: 1e-5,
            epsilon2: 1e-10,
            max_line_search_iter: 10,
            max_sqp_iter: 100,
        }
    }
}
pub struct Disc1dAdvectionSpaceTime<'a> {
    pub current_iter: usize,
    pub basis: LagrangeBasis1DLobatto,
    pub enriched_basis: LagrangeBasis1DLobatto,
    pub interp_matrix: Array2<f64>,
    pub mesh: &'a mut Mesh2d,
    solver_param: &'a SolverParameters,
    advection_speed: f64,
}
impl<'a> Disc1dAdvectionSpaceTime<'a> {
    pub fn new(
        basis: LagrangeBasis1DLobatto,
        enriched_basis: LagrangeBasis1DLobatto,
        mesh: &'a mut Mesh2d,
        solver_param: &'a SolverParameters,
    ) -> Disc1dAdvectionSpaceTime<'a> {
        let interp_matrix = Self::compute_interp_matrix(&basis, &enriched_basis);
        println!("basis: {:?}", basis.cell_gauss_points);
        println!("enriched_basis: {:?}", enriched_basis.cell_gauss_points);
        Disc1dAdvectionSpaceTime {
            current_iter: 0,
            basis,
            enriched_basis,
            interp_matrix,
            mesh,
            solver_param,
            advection_speed: 0.1,
        }
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
        dx: ArrayView2<f64>,
        obj_dsol: ArrayView1<f64>,
        obj_dx: ArrayView1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let nelem = self.mesh.elem_num;
        let unenriched_cell_ngp = self.solver_param.cell_gp_num;
        let free_x = &self.mesh.free_x;
        let num_u = nelem * unenriched_cell_ngp * unenriched_cell_ngp;
        let num_x: usize = free_x.len();
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
            .assign(&dx.dot(&node_constraints).t());

        A_ndarray
            .slice_mut(s![num_u + num_x..n_total, 0..num_u])
            .assign(&dsol);
        A_ndarray
            .slice_mut(s![num_u + num_x..n_total, num_u..num_u + num_x])
            .assign(&dx.dot(&node_constraints));

        b_ndarray
            .slice_mut(s![0..num_u])
            .assign(&(&obj_dsol * -1.0));
        b_ndarray
            .slice_mut(s![num_u..num_u + num_x])
            .assign(&(&obj_dx * -1.0));
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
    #[allow(non_snake_case)]
    pub fn solve(&mut self, mut solutions: ArrayViewMut2<f64>) {
        let nelem = self.mesh.elem_num;
        let nnode = self.mesh.node_num;
        let unenriched_cell_ngp = self.solver_param.cell_gp_num;
        let enriched_cell_ngp = unenriched_cell_ngp + 1;
        let epsilon1 = 1e-5;
        let epsilon2 = 1e-10;
        let max_line_search_iter = 20;
        let max_sqp_iter = 100;
        let interior_nnodes = self.mesh.interior_node_num;
        let free_x = &self.mesh.free_x;
        let mut node_constraints: Array2<f64> =
            Array2::zeros((2 * nnode, 2 * interior_nnodes + free_x.len()));
        node_constraints[(4, 0)] = 1.0;

        let mut residuals: Array2<f64> =
            Array2::zeros((nelem, unenriched_cell_ngp * unenriched_cell_ngp));
        let mut dsol: Array2<f64> = Array2::zeros((
            nelem * unenriched_cell_ngp * unenriched_cell_ngp,
            nelem * unenriched_cell_ngp * unenriched_cell_ngp,
        ));
        let mut dx: Array2<f64> =
            Array2::zeros((nelem * unenriched_cell_ngp * unenriched_cell_ngp, 2 * nnode));
        let mut enriched_residuals: Array2<f64> =
            Array2::zeros((nelem, enriched_cell_ngp * enriched_cell_ngp));
        let mut enriched_dsol: Array2<f64> = Array2::zeros((
            nelem * enriched_cell_ngp * enriched_cell_ngp,
            nelem * unenriched_cell_ngp * unenriched_cell_ngp,
        ));
        let mut enriched_dx: Array2<f64> =
            Array2::zeros((nelem * enriched_cell_ngp * enriched_cell_ngp, 2 * nnode));

        let mut iter: usize = 0;
        while iter < max_sqp_iter {
            println!("iter: {:?}", iter);
            // reset residuals, dsol, dx, enriched_residuals, enriched_dsol, enriched_dx
            residuals.fill(0.0);
            dsol.fill(0.0);
            dx.fill(0.0);
            enriched_residuals.fill(0.0);
            enriched_dsol.fill(0.0);
            enriched_dx.fill(0.0);

            println!("solutions: {:?}", solutions);
            println!("node: {:?}", self.mesh.nodes[free_x[0]].x);
            self.compute_residuals_and_derivatives(
                solutions.view(),
                residuals.view_mut(),
                dsol.view_mut(),
                dx.view_mut(),
                false,
            );
            println!("residuals: {:?}", residuals);
            //println!("dsol: {:?}", dsol);
            //println!("dx: {:?}", dx);

            let enriched_solutions = self.interpolate_to_enriched(solutions.view());
            self.compute_residuals_and_derivatives(
                enriched_solutions.view(),
                enriched_residuals.view_mut(),
                enriched_dsol.view_mut(),
                enriched_dx.view_mut(),
                true,
            );
            //println!("enriched_residuals: {:?}", enriched_residuals);
            //println!("enriched_dsol: {:?}", enriched_dsol.slice(s![.., 5]));
            //println!("enriched_dx: {:?}", enriched_dx);

            let obj_dsol = enriched_dsol.t().dot(&enriched_residuals.flatten());
            let obj_dx = enriched_dx
                .t()
                .dot(&enriched_residuals.flatten())
                .dot(&node_constraints);
            println!("obj_dsol: {:?}", obj_dsol);
            println!("obj_dx: {:?}", obj_dx);
            let dsol_faer = dsol.view().into_faer();
            let dsol_inv = dsol_faer.partial_piv_lu().inverse();
            let dsol_inv_t = dsol_inv.transpose().into_ndarray();
            let obj_dsol_t = obj_dsol.t();
            let lambda_hat = dsol_inv_t.dot(&obj_dsol_t);
            let mu = lambda_hat.mapv(f64::abs).max().copied().unwrap() * 2.0;
            println!("mu: {:?}", mu);
            // termination criteria
            let optimality = &obj_dx.t()
                - &dx
                    .dot(&node_constraints)
                    .t()
                    .dot(&dsol_inv_t)
                    .dot(&obj_dsol.t());
            let optimality_norm = optimality.mapv(|x| x.powi(2)).sum().sqrt();
            let feasibility_norm = residuals.mapv(|x| x.powi(2)).sum().sqrt();
            println!("optimality: {:?}", optimality_norm);
            println!("feasibility: {:?}", feasibility_norm);
            if optimality_norm < epsilon1 && feasibility_norm < epsilon2 {
                println!("Terminating SQP at iter: {:?}", iter);
                break;
            }

            let hessian_uu = enriched_dsol.t().dot(&enriched_dsol);
            println!("hessian_uu: {:?}", hessian_uu);
            let hessian_ux = enriched_dsol.t().dot(&enriched_dx.dot(&node_constraints));
            println!("hessian_ux: {:?}", hessian_ux);
            let mut hessian_xx = enriched_dx
                .dot(&node_constraints)
                .t()
                .dot(&enriched_dx.dot(&node_constraints));
            println!("hessian_xx: {:?}", hessian_xx);
            // add an identity matrix to hessian_xx
            hessian_xx += &(1e-8 * &Array2::eye(2 * interior_nnodes + free_x.len()));

            let (delta_u, delta_x) = self.solve_linear_subproblem(
                node_constraints.view(),
                residuals.view(),
                hessian_uu.view(),
                hessian_ux.view(),
                hessian_xx.view(),
                dsol.view(),
                dx.view(),
                obj_dsol.view(),
                obj_dx.view(),
            );
            println!("delta_u: {:?}", delta_u);
            println!("delta_x: {:?}", delta_x);

            // backtracking line search
            let merit_func = |alpha: f64| -> f64 {
                let mut tmp_mesh = self.mesh.clone();
                let delta_u_ndarray = Array::from_iter(delta_u.iter().copied());
                let u_flat = &solutions.flatten() + alpha * &delta_u_ndarray;
                let u = u_flat
                    .to_shape((nelem, unenriched_cell_ngp * unenriched_cell_ngp))
                    .unwrap();
                tmp_mesh.nodes[free_x[0]].x += alpha * delta_x[0];
                let mut tmp_res = Array2::zeros((nelem, unenriched_cell_ngp * unenriched_cell_ngp));
                self.compute_residuals(&tmp_mesh, u.view(), tmp_res.view_mut(), false);
                let enriched_u = self.interpolate_to_enriched(u.view());
                let mut tmp_enr_res = Array2::zeros((nelem, enriched_cell_ngp * enriched_cell_ngp));
                self.compute_residuals(&tmp_mesh, enriched_u.view(), tmp_enr_res.view_mut(), true);
                let f = 0.5 * &tmp_enr_res.flatten().dot(&tmp_enr_res.flatten());
                let l1_norm = tmp_res.mapv(f64::abs).sum();

                f + mu * l1_norm
            };

            let merit_func_0 = merit_func(0.0);
            let dir_deriv =
                obj_dsol.dot(&delta_u) + obj_dx.dot(&delta_x) - mu * residuals.mapv(f64::abs).sum();
            println!("dir_deriv: {:?}", dir_deriv);
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
            println!("alpha: {:?}", alpha);
            if line_search_iter == max_line_search_iter {
                panic!(
                    "Warning: Line search did not converge within {} iterations.",
                    max_line_search_iter
                );
            }

            solutions.scaled_add(alpha, &delta_u.to_shape(solutions.shape()).unwrap());
            for (i, &ix) in free_x.iter().enumerate() {
                self.mesh.nodes[ix].x += alpha * delta_x[i];
            }
            iter += 1;
        }
    }
    fn compute_interp_matrix(
        basis: &LagrangeBasis1DLobatto,
        enriched_basis: &LagrangeBasis1DLobatto,
    ) -> Array2<f64> {
        let cell_ngp = basis.cell_gauss_points.len();
        let enriched_ngp = cell_ngp + 1;
        let mut interp_matrix = Array2::zeros((enriched_ngp * enriched_ngp, cell_ngp * cell_ngp));
        for i_enr in 0..enriched_ngp {
            let xi_enr = enriched_basis.cell_gauss_points[i_enr];
            for j_enr in 0..enriched_ngp {
                let eta_enr = enriched_basis.cell_gauss_points[j_enr];
                let enr_idx = i_enr * enriched_ngp + j_enr;
                // Evaluate each standard basis at this enriched point
                for i_std in 0..cell_ngp {
                    for j_std in 0..cell_ngp {
                        let std_idx = i_std * cell_ngp + j_std;
                        // Tensor product of 1D basis functions
                        let phi_xi = basis.evaluate_basis_at(i_std, xi_enr);
                        let phi_eta = basis.evaluate_basis_at(j_std, eta_enr);
                        interp_matrix[[enr_idx, std_idx]] = phi_xi * phi_eta;
                    }
                }
            }
        }
        interp_matrix
    }
    fn interpolate_to_enriched(&self, sol: ArrayView2<f64>) -> Array2<f64> {
        let interp_matrix = self.interp_matrix.view();
        let mut enriched_solutions: Array2<f64> = Array2::zeros((
            self.mesh.elem_num,
            (self.solver_param.cell_gp_num + 1) * (self.solver_param.cell_gp_num + 1),
        ));
        // Apply interpolation using ndarray's dot method
        for ielem in 0..self.mesh.elem_num {
            enriched_solutions
                .slice_mut(s![ielem, ..])
                .assign(&interp_matrix.dot(&sol.slice(s![ielem, ..])));
        }
        enriched_solutions
    }

    fn compute_residuals(
        &self,
        mesh: &Mesh2d,
        solutions: ArrayView2<f64>,
        mut residuals: ArrayViewMut2<f64>,
        is_enriched: bool,
    ) {
        let nelem = mesh.elem_num;
        let cell_ngp = {
            if is_enriched {
                self.solver_param.cell_gp_num + 1
            } else {
                self.solver_param.cell_gp_num
            }
        };
        let basis = {
            if is_enriched {
                &self.enriched_basis
            } else {
                &self.basis
            }
        };
        let weights = &basis.cell_gauss_weights;
        for ielem in 0..nelem {
            let inodes = &mesh.elements[ielem].inodes;
            let mut x_slice = [0.0; 4];
            let mut y_slice = [0.0; 4];
            for i in 0..4 {
                x_slice[i] = mesh.nodes[inodes[i]].x;
                y_slice[i] = mesh.nodes[inodes[i]].y;
            }
            for itest_func in 0..cell_ngp * cell_ngp {
                let itest_func_eta = itest_func / cell_ngp;
                let itest_func_xi = itest_func % cell_ngp;
                let res = self.volume_integral(
                    basis,
                    itest_func_eta,
                    itest_func_xi,
                    solutions.slice(s![ielem, ..]).as_slice().unwrap(),
                    x_slice.as_slice(),
                    y_slice.as_slice(),
                );
                residuals[(ielem, itest_func)] += res;
            }
        }
        for &iedge in mesh.internal_edges.iter() {
            let edge = &mesh.edges[iedge];
            let left_ref_normal = [1.0, 0.0];
            let right_ref_normal = [-1.0, 0.0];
            let ilelem = edge.parents[0];
            let irelem = edge.parents[1];
            let left_elem = &mesh.elements[ilelem];
            let right_elem = &mesh.elements[irelem];
            let (mut left_x, mut left_y, mut right_x, mut right_y) =
                ([0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
            for i in 0..4 {
                left_x[i] = mesh.nodes[left_elem.inodes[i]].x;
                left_y[i] = mesh.nodes[left_elem.inodes[i]].y;
                right_x[i] = mesh.nodes[right_elem.inodes[i]].x;
                right_y[i] = mesh.nodes[right_elem.inodes[i]].y;
            }
            let common_edge = [edge.local_ids[0], edge.local_ids[1]];
            let mut grouped_x: [f64; 6] = [0.0; 6];
            let mut grouped_y: [f64; 6] = [0.0; 6];
            let mut grouped_index = 0;
            let mut left_nodes_ids: [usize; 4] = [0; 4]; // ids in grouped nodes
            let mut right_nodes_ids: [usize; 4] = [0; 4]; // ids in grouped nodes
            for i in 0..4 {
                if i != common_edge[0] && i != (common_edge[0] + 1) % 4 {
                    grouped_x[grouped_index] = left_x[i];
                    grouped_y[grouped_index] = left_y[i];
                    left_nodes_ids[i] = grouped_index;
                    grouped_index += 1;
                }
            }
            for i in 0..4 {
                if i != common_edge[1] && i != (common_edge[1] + 1) % 4 {
                    grouped_x[grouped_index] = right_x[i];
                    grouped_y[grouped_index] = right_y[i];
                    right_nodes_ids[i] = grouped_index;
                    grouped_index += 1;
                }
            }
            grouped_x[4] = left_x[common_edge[0]];
            grouped_x[5] = left_x[(common_edge[0] + 1) % 4];
            grouped_y[4] = left_y[common_edge[0]];
            grouped_y[5] = left_y[(common_edge[0] + 1) % 4];
            left_nodes_ids[common_edge[0]] = 4;
            left_nodes_ids[(common_edge[0] + 1) % 4] = 5;
            right_nodes_ids[common_edge[1]] = 4;
            right_nodes_ids[(common_edge[1] + 1) % 4] = 5;

            let left_sol_slice = solutions.slice(s![ilelem, ..]);
            let right_sol_slice = solutions.slice(s![irelem, ..]);
            for itest_func in 0..cell_ngp * cell_ngp {
                let itest_func_eta = itest_func / cell_ngp;
                let itest_func_xi = itest_func % cell_ngp;
                for edge_gp in 0..cell_ngp {
                    // Map quadrature points based on edge orientation
                    let left_kgp = edge_gp;
                    let left_igp = cell_ngp - 1;
                    let right_kgp = edge_gp;
                    let right_igp = 0;
                    let left_eta = basis.cell_gauss_points[left_kgp];
                    let left_xi = basis.cell_gauss_points[left_igp];
                    let right_eta = basis.cell_gauss_points[right_kgp];
                    let right_xi = basis.cell_gauss_points[right_igp];
                    let left_value = left_sol_slice[left_igp + left_kgp * cell_ngp];
                    let right_value = right_sol_slice[right_igp + right_kgp * cell_ngp];
                    let num_flux = self.compute_numerical_flux(
                        self.advection_speed,
                        left_value,
                        right_value,
                        left_x[common_edge[0]],
                        left_x[(common_edge[0] + 1) % 4],
                        left_y[common_edge[0]],
                        left_y[(common_edge[0] + 1) % 4],
                    );
                    let left_scaling = self.compute_flux_scaling(
                        left_eta,
                        left_xi,
                        left_ref_normal,
                        left_x.as_slice(),
                        left_y.as_slice(),
                    );
                    let right_scaling = self.compute_flux_scaling(
                        right_eta,
                        right_xi,
                        right_ref_normal,
                        right_x.as_slice(),
                        right_y.as_slice(),
                    );

                    let left_transformed_flux = num_flux * left_scaling;
                    let right_transformed_flux = -num_flux * right_scaling;

                    let left_phi = basis.phis_cell_gps[[itest_func_xi, left_igp]]
                        * basis.phis_cell_gps[[itest_func_eta, left_kgp]];
                    let right_phi = basis.phis_cell_gps[[itest_func_xi, right_igp]]
                        * basis.phis_cell_gps[[itest_func_eta, right_kgp]];

                    residuals[(ilelem, itest_func)] +=
                        weights[edge_gp] * left_transformed_flux * left_phi;
                    residuals[(irelem, itest_func)] +=
                        weights[edge_gp] * right_transformed_flux * right_phi;
                }
            }
        }

        for &iedge in mesh.boundary_edges.iter() {
            let edge = &mesh.edges[iedge];
            let ielem = edge.parents[0];

            let elem = &mesh.elements[ielem];
            let inodes = &elem.inodes;
            let mut x_slice = [0.0; 4];
            let mut y_slice = [0.0; 4];
            for i in 0..4 {
                x_slice[i] = self.mesh.nodes[inodes[i]].x;
                y_slice[i] = self.mesh.nodes[inodes[i]].y;
            }
            let sol_slice = solutions.slice(s![ielem, ..]);
            for itest_func in 0..cell_ngp * cell_ngp {
                let itest_func_eta = itest_func / cell_ngp;
                let itest_func_xi = itest_func % cell_ngp;
                for edge_gp in 0..cell_ngp {
                    let (kgp, igp, ref_normal) = match iedge {
                        0 | 1 => (0, edge_gp, [0.0, -1.0]),
                        2 => (edge_gp, cell_ngp - 1, [1.0, 0.0]),
                        3 | 4 => (cell_ngp - 1, edge_gp, [0.0, 1.0]),
                        5 => (edge_gp, 0, [-1.0, 0.0]),
                        _ => unreachable!(),
                    };
                    let eta = basis.cell_gauss_points[kgp];
                    let xi = basis.cell_gauss_points[igp];
                    let flux = match iedge {
                        0 => {
                            let u = 2.0;
                            -u
                        }
                        1 => {
                            let u = 0.0;
                            u
                        }
                        2 => {
                            let u = sol_slice[kgp * cell_ngp + igp];
                            self.advection_speed * u
                        }
                        3 | 4 => {
                            let u = sol_slice[igp + kgp * cell_ngp];
                            u
                        }
                        5 => {
                            let u = 2.0;
                            -self.advection_speed * u
                        }
                        _ => unreachable!(),
                    };
                    let scaling = self.compute_flux_scaling(
                        eta,
                        xi,
                        ref_normal,
                        x_slice.as_slice(),
                        y_slice.as_slice(),
                    );
                    let transformed_flux = scaling * flux;
                    let phi = basis.phis_cell_gps[(itest_func_eta, kgp)]
                        * basis.phis_cell_gps[(itest_func_xi, igp)];
                    residuals[(ielem, itest_func)] += weights[edge_gp] * phi * transformed_flux;
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
        // mut dy: ArrayViewMut2<f64>,
        is_enriched: bool,
    ) {
        let nelem = self.mesh.elem_num;
        let nnode = self.mesh.node_num;
        let cell_ngp = {
            if is_enriched {
                self.solver_param.cell_gp_num + 1
            } else {
                self.solver_param.cell_gp_num
            }
        };
        let unenriched_cell_ngp = self.solver_param.cell_gp_num;
        let basis = {
            if is_enriched {
                &self.enriched_basis
            } else {
                &self.basis
            }
        };
        let weights = &basis.cell_gauss_weights;
        for ielem in 0..nelem {
            let elem = &self.mesh.elements[ielem];
            let inodes = &self.mesh.elements[ielem].inodes;
            let mut x_slice = [0.0; 4];
            let mut y_slice = [0.0; 4];
            for i in 0..4 {
                x_slice[i] = self.mesh.nodes[inodes[i]].x;
                y_slice[i] = self.mesh.nodes[inodes[i]].y;
            }
            for itest_func in 0..cell_ngp * cell_ngp {
                let itest_func_eta = itest_func / cell_ngp;
                let itest_func_xi = itest_func % cell_ngp;
                let mut dvol_sol: Array1<f64> = Array1::zeros(cell_ngp * cell_ngp);
                let mut dvol_x: Array1<f64> = Array1::zeros(4);
                let mut dvol_y: Array1<f64> = Array1::zeros(4);
                let res = self.dvolume(
                    basis,
                    itest_func_eta,
                    itest_func_xi,
                    solutions.slice(s![ielem, ..]).as_slice().unwrap(),
                    dvol_sol.as_slice_mut().unwrap(),
                    x_slice.as_slice(),
                    dvol_x.as_slice_mut().unwrap(),
                    y_slice.as_slice(),
                    dvol_y.as_slice_mut().unwrap(),
                    1.0,
                );
                residuals[(ielem, itest_func)] += res;
                if is_enriched {
                    dsol.slice_mut(s![
                        ielem * cell_ngp * cell_ngp + itest_func,
                        ielem * unenriched_cell_ngp * unenriched_cell_ngp
                            ..(ielem + 1) * unenriched_cell_ngp * unenriched_cell_ngp
                    ])
                    .scaled_add(1.0, &self.interp_matrix.t().dot(&dvol_sol));
                } else {
                    dsol.slice_mut(s![
                        ielem * cell_ngp * cell_ngp + itest_func,
                        ielem * cell_ngp * cell_ngp..(ielem + 1) * cell_ngp * cell_ngp
                    ])
                    .scaled_add(1.0, &dvol_sol);
                }
                for i in 0..4 {
                    dx[(ielem * cell_ngp * cell_ngp + itest_func, elem.inodes[i])] += dvol_x[i];
                    dx[(
                        ielem * cell_ngp * cell_ngp + itest_func,
                        nnode + elem.inodes[i],
                    )] += dvol_y[i];
                }
                /*
                let res = self.volume_integral(
                    itest_func_eta,
                    itest_func_xi,
                    solutions.slice(s![ielem, ..]).as_slice().unwrap(),
                    x_slice.as_slice(),
                    y_slice.as_slice(),
                );
                */
                /*
                let mut sol_perturbed = solutions.slice(s![ielem, ..]).to_owned();
                sol_perturbed[0] += 1e-4;
                let res_perturbed = self.volume_integral(
                    itest_func_eta,
                    itest_func_xi,
                    sol_perturbed.as_slice().unwrap(),
                    x_slice.as_slice(),
                    y_slice.as_slice(),
                );
                // finite difference
                let dres_dsol0 = (res_perturbed - res) / 1e-4;
                println!("dres_dsol0: {:?}", dres_dsol0);

                */
                /*
                if ielem == 0 {
                    println!(
                        "dx_ad[{:?}]: {:?}",
                        itest_func,
                        dx[(ielem * cell_ngp * cell_ngp + itest_func, 2)]
                    );
                    // perturb x[2]
                    x_slice[2] += 1e-4;
                    let res_perturbed = self.volume_integral(
                        itest_func_eta,
                        itest_func_xi,
                        solutions.slice(s![ielem, ..]).as_slice().unwrap(),
                        x_slice.as_slice(),
                        y_slice.as_slice(),
                    );
                    let dx_fd = (res_perturbed - res) / 1e-4;
                    println!("dx_fd: {:?}", dx_fd);
                }
                */
                /*
                println!(
                    "dy[{:?}]: {:?}",
                    itest_func,
                    dy.slice(s![ielem, itest_func, ..])
                );
                */
            }

            /*
            let mut residuals_slice: ArrayViewMut2<f64> = residuals.slice_mut(s![ielem, .., ..]);
            residuals_slice.scaled_add(1.0, &self.volume_integral(solutions, ielem));
            */
            // println!("volume_residuals_slice: {:?}", residuals_slice);
        }
        // println!("residuals after volume integral: {:?}", residuals);

        for &iedge in self.mesh.internal_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let left_ref_normal = [1.0, 0.0];
            let right_ref_normal = [-1.0, 0.0];
            let ilelem = edge.parents[0];
            let irelem = edge.parents[1];
            let left_elem = &self.mesh.elements[ilelem];
            let right_elem = &self.mesh.elements[irelem];
            let (mut left_x, mut left_y, mut right_x, mut right_y) =
                ([0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
            for i in 0..4 {
                left_x[i] = self.mesh.nodes[left_elem.inodes[i]].x;
                left_y[i] = self.mesh.nodes[left_elem.inodes[i]].y;
                right_x[i] = self.mesh.nodes[right_elem.inodes[i]].x;
                right_y[i] = self.mesh.nodes[right_elem.inodes[i]].y;
            }
            let mut common_edge = [edge.local_ids[0], edge.local_ids[1]];
            let mut grouped_x: [f64; 6] = [0.0; 6];
            let mut grouped_y: [f64; 6] = [0.0; 6];
            let mut grouped_index = 0;
            let mut left_nodes_ids: [usize; 4] = [0; 4]; // ids in grouped nodes
            let mut right_nodes_ids: [usize; 4] = [0; 4]; // ids in grouped nodes
            for i in 0..4 {
                if i != common_edge[0] && i != (common_edge[0] + 1) % 4 {
                    grouped_x[grouped_index] = left_x[i];
                    grouped_y[grouped_index] = left_y[i];
                    left_nodes_ids[i] = grouped_index;
                    grouped_index += 1;
                }
            }
            for i in 0..4 {
                if i != common_edge[1] && i != (common_edge[1] + 1) % 4 {
                    grouped_x[grouped_index] = right_x[i];
                    grouped_y[grouped_index] = right_y[i];
                    right_nodes_ids[i] = grouped_index;
                    grouped_index += 1;
                }
            }
            grouped_x[4] = left_x[common_edge[0]];
            grouped_x[5] = left_x[(common_edge[0] + 1) % 4];
            grouped_y[4] = left_y[common_edge[0]];
            grouped_y[5] = left_y[(common_edge[0] + 1) % 4];
            left_nodes_ids[common_edge[0]] = 4;
            left_nodes_ids[(common_edge[0] + 1) % 4] = 5;
            right_nodes_ids[common_edge[1]] = 4;
            right_nodes_ids[(common_edge[1] + 1) % 4] = 5;

            let left_sol_slice = solutions.slice(s![ilelem, ..]);
            let right_sol_slice = solutions.slice(s![irelem, ..]);
            for itest_func in 0..cell_ngp * cell_ngp {
                let itest_func_eta = itest_func / cell_ngp;
                let itest_func_xi = itest_func % cell_ngp;
                for edge_gp in 0..cell_ngp {
                    // Map quadrature points based on edge orientation
                    let left_kgp = edge_gp;
                    let left_igp = cell_ngp - 1;
                    let right_kgp = edge_gp;
                    let right_igp = 0;
                    let left_eta = basis.cell_gauss_points[left_kgp];
                    let left_xi = basis.cell_gauss_points[left_igp];
                    let right_eta = basis.cell_gauss_points[right_kgp];
                    let right_xi = basis.cell_gauss_points[right_igp];
                    let left_value = left_sol_slice[left_igp + left_kgp * cell_ngp];
                    let right_value = right_sol_slice[right_igp + right_kgp * cell_ngp];
                    let mut dflux_dleft_x = [0.0; 4];
                    let mut dflux_dleft_y = [0.0; 4];
                    let mut dflux_dright_x = [0.0; 4];
                    let mut dflux_dright_y = [0.0; 4];
                    let (
                        num_flux,
                        dflux_dul,
                        dflux_dur,
                        dflux_dx0,
                        dflux_dx1,
                        dflux_dy0,
                        dflux_dy1,
                    ) = self.dnum_flux(
                        self.advection_speed,
                        left_value,
                        right_value,
                        left_x[common_edge[0]],
                        left_x[(common_edge[0] + 1) % 4],
                        left_y[common_edge[0]],
                        left_y[(common_edge[0] + 1) % 4],
                        1.0,
                    );

                    dflux_dleft_x[common_edge[0]] = dflux_dx0;
                    dflux_dleft_x[(common_edge[0] + 1) % 4] = dflux_dx1;
                    dflux_dleft_y[common_edge[0]] = dflux_dy0;
                    dflux_dleft_y[(common_edge[0] + 1) % 4] = dflux_dy1;
                    dflux_dright_x[common_edge[1]] = dflux_dx1;
                    dflux_dright_x[(common_edge[1] + 1) % 4] = dflux_dx0;
                    dflux_dright_y[common_edge[1]] = dflux_dy1;
                    dflux_dright_y[(common_edge[1] + 1) % 4] = dflux_dy0;
                    /*
                    let num_flux = self.compute_numerical_flux(
                        self.advection_speed,
                        left_value,
                        right_value,
                        grouped_x[4],
                        grouped_x[5],
                        grouped_y[4],
                        grouped_y[5],
                    );
                    */

                    let mut dleft_scaling_dx = [0.0; 4];
                    let mut dleft_scaling_dy = [0.0; 4];
                    let mut dright_scaling_dx = [0.0; 4];
                    let mut dright_scaling_dy = [0.0; 4];
                    let left_scaling = self.dscaling(
                        left_eta,
                        left_xi,
                        left_ref_normal,
                        left_x.as_slice(),
                        dleft_scaling_dx.as_mut_slice(),
                        left_y.as_slice(),
                        dleft_scaling_dy.as_mut_slice(),
                        1.0,
                    );
                    let right_scaling = self.dscaling(
                        right_eta,
                        right_xi,
                        right_ref_normal,
                        right_x.as_slice(),
                        dright_scaling_dx.as_mut_slice(),
                        right_y.as_slice(),
                        dright_scaling_dy.as_mut_slice(),
                        1.0,
                    );

                    let left_transformed_flux = num_flux * left_scaling;
                    let right_transformed_flux = -num_flux * right_scaling;

                    let dleft_transformed_flux_dul = left_scaling * dflux_dul;
                    let dleft_transformed_flux_dur = left_scaling * dflux_dur;

                    let dleft_transformed_flux_dx = array![
                        dleft_scaling_dx[0] * num_flux + left_scaling * dflux_dleft_x[0],
                        dleft_scaling_dx[1] * num_flux + left_scaling * dflux_dleft_x[1],
                        dleft_scaling_dx[2] * num_flux + left_scaling * dflux_dleft_x[2],
                        dleft_scaling_dx[3] * num_flux + left_scaling * dflux_dleft_x[3],
                    ];
                    let dleft_transformed_flux_dy = array![
                        dleft_scaling_dy[0] * num_flux + left_scaling * dflux_dleft_y[0],
                        dleft_scaling_dy[1] * num_flux + left_scaling * dflux_dleft_y[1],
                        dleft_scaling_dy[2] * num_flux + left_scaling * dflux_dleft_y[2],
                        dleft_scaling_dy[3] * num_flux + left_scaling * dflux_dleft_y[3],
                    ];

                    let dright_transformed_flux_dul = -right_scaling * dflux_dul;
                    let dright_transformed_flux_dur = -right_scaling * dflux_dur;

                    let dright_transformed_flux_dx = array![
                        -(dright_scaling_dx[0] * num_flux + right_scaling * dflux_dright_x[0]),
                        -(dright_scaling_dx[1] * num_flux + right_scaling * dflux_dright_x[1]),
                        -(dright_scaling_dx[2] * num_flux + right_scaling * dflux_dright_x[2]),
                        -(dright_scaling_dx[3] * num_flux + right_scaling * dflux_dright_x[3]),
                    ];
                    let dright_transformed_flux_dy = array![
                        -(dright_scaling_dy[0] * num_flux + right_scaling * dflux_dright_y[0]),
                        -(dright_scaling_dy[1] * num_flux + right_scaling * dflux_dright_y[1]),
                        -(dright_scaling_dy[2] * num_flux + right_scaling * dflux_dright_y[2]),
                        -(dright_scaling_dy[3] * num_flux + right_scaling * dflux_dright_y[3]),
                    ];

                    /*
                    let mut ul_perturbed = left_value + 1e-4;
                    let mut ur_perturbed = right_value + 1e-4;
                    let ul_perturbed_num_flux = self.compute_numerical_flux(
                        self.advection_speed,
                        ul_perturbed,
                        right_value,
                        left_x[common_edge[0]],
                        left_x[(common_edge[0] + 1) % 4],
                        left_y[common_edge[0]],
                        left_y[(common_edge[0] + 1) % 4],
                    );
                    let ur_perturbed_num_flux = self.compute_numerical_flux(
                        self.advection_speed,
                        left_value,
                        ur_perturbed,
                        left_x[common_edge[0]],
                        left_x[(common_edge[0] + 1) % 4],
                        left_y[common_edge[0]],
                        left_y[(common_edge[0] + 1) % 4],
                    );
                    let ul_perturbed_transformed_flux = ul_perturbed_num_flux * left_scaling;
                    let ur_perturbed_transformed_flux = ur_perturbed_num_flux * right_scaling;
                    let dleft_transformed_flux_dul_FD =
                        (ul_perturbed_transformed_flux - left_transformed_flux) / 1e-4;
                    let dleft_transformed_flux_dur_FD =
                        (ur_perturbed_transformed_flux - left_transformed_flux) / 1e-4;

                    let mut x1_perturbed = left_x.clone();
                    x1_perturbed[1] += 1e-5;
                    let mut y1_perturbed = left_y.clone();
                    y1_perturbed[1] += 1e-5;
                    let x1_perturbed_scaling = self.compute_flux_scaling(
                        left_eta,
                        left_xi,
                        left_ref_normal,
                        x1_perturbed.as_slice(),
                        left_y.as_slice(),
                    );
                    let x1_perturbed_num_flux = self.compute_numerical_flux(
                        self.advection_speed,
                        left_value,
                        right_value,
                        x1_perturbed[1],
                        x1_perturbed[2],
                        left_y[1],
                        left_y[2],
                    );
                    let x1_perturbed_transformed_flux =
                        x1_perturbed_num_flux * x1_perturbed_scaling;
                    let dleft_transformed_flux_dx_FD =
                        (x1_perturbed_transformed_flux - left_transformed_flux) / 1e-5;

                    println!(
                        "dleft_transformed_flux_dul_FD: {:?}",
                        dleft_transformed_flux_dul_FD
                    );
                    println!(
                        "dleft_transformed_flux_dur_FD: {:?}",
                        dleft_transformed_flux_dur_FD
                    );
                    println!(
                        "dleft_transformed_flux_dx[1]_FD: {:?}",
                        dleft_transformed_flux_dx_FD
                    );
                    */
                    let left_phi = basis.phis_cell_gps[[itest_func_xi, left_igp]]
                        * basis.phis_cell_gps[[itest_func_eta, left_kgp]];
                    let right_phi = basis.phis_cell_gps[[itest_func_xi, right_igp]]
                        * basis.phis_cell_gps[[itest_func_eta, right_kgp]];

                    residuals[(ilelem, itest_func)] +=
                        weights[edge_gp] * left_transformed_flux * left_phi;
                    residuals[(irelem, itest_func)] +=
                        weights[edge_gp] * right_transformed_flux * right_phi;

                    if is_enriched {
                        dsol.slice_mut(s![
                            ilelem * cell_ngp * cell_ngp + itest_func,
                            ilelem * unenriched_cell_ngp * unenriched_cell_ngp
                                ..(ilelem + 1) * unenriched_cell_ngp * unenriched_cell_ngp
                        ])
                        .scaled_add(
                            weights[edge_gp] * left_phi,
                            &(&self.interp_matrix.slice(s![itest_func, ..])
                                * dleft_transformed_flux_dul),
                        );
                        dsol.slice_mut(s![
                            ilelem * cell_ngp * cell_ngp + itest_func,
                            irelem * unenriched_cell_ngp * unenriched_cell_ngp
                                ..(irelem + 1) * unenriched_cell_ngp * unenriched_cell_ngp
                        ])
                        .scaled_add(
                            weights[edge_gp] * left_phi,
                            &(&self.interp_matrix.slice(s![itest_func, ..])
                                * dleft_transformed_flux_dur),
                        );
                        dsol.slice_mut(s![
                            irelem * cell_ngp * cell_ngp + itest_func,
                            ilelem * unenriched_cell_ngp * unenriched_cell_ngp
                                ..(ilelem + 1) * unenriched_cell_ngp * unenriched_cell_ngp
                        ])
                        .scaled_add(
                            weights[edge_gp] * right_phi,
                            &(&self.interp_matrix.slice(s![itest_func, ..])
                                * dright_transformed_flux_dur),
                        );
                        dsol.slice_mut(s![
                            irelem * cell_ngp * cell_ngp + itest_func,
                            ilelem * unenriched_cell_ngp * unenriched_cell_ngp
                                ..(ilelem + 1) * unenriched_cell_ngp * unenriched_cell_ngp
                        ])
                        .scaled_add(
                            weights[edge_gp] * right_phi,
                            &(&self.interp_matrix.slice(s![itest_func, ..])
                                * dright_transformed_flux_dul),
                        );
                    } else {
                        dsol[(
                            ilelem * cell_ngp * cell_ngp + itest_func,
                            ilelem * cell_ngp * cell_ngp + left_kgp * cell_ngp + left_igp,
                        )] += weights[edge_gp] * dleft_transformed_flux_dul * left_phi;
                        dsol[(
                            ilelem * cell_ngp * cell_ngp + itest_func,
                            irelem * cell_ngp * cell_ngp + right_kgp * cell_ngp + right_igp,
                        )] += weights[edge_gp] * dleft_transformed_flux_dur * left_phi;
                        dsol[(
                            irelem * cell_ngp * cell_ngp + itest_func,
                            irelem * cell_ngp * cell_ngp + right_kgp * cell_ngp + right_igp,
                        )] += weights[edge_gp] * dright_transformed_flux_dur * right_phi;
                        dsol[(
                            irelem * cell_ngp * cell_ngp + itest_func,
                            ilelem * cell_ngp * cell_ngp + left_kgp * cell_ngp + left_igp,
                        )] += weights[edge_gp] * dright_transformed_flux_dul * right_phi;
                    }
                    for i in 0..4 {
                        dx[(
                            ilelem * cell_ngp * cell_ngp + itest_func,
                            left_elem.inodes[i],
                        )] += weights[edge_gp] * left_phi * dleft_transformed_flux_dx[i];
                        dx[(
                            ilelem * cell_ngp * cell_ngp + itest_func,
                            nnode + left_elem.inodes[i],
                        )] += weights[edge_gp] * left_phi * dleft_transformed_flux_dy[i];
                        dx[(
                            irelem * cell_ngp * cell_ngp + itest_func,
                            right_elem.inodes[i],
                        )] += weights[edge_gp] * right_phi * dright_transformed_flux_dx[i];
                        dx[(
                            irelem * cell_ngp * cell_ngp + itest_func,
                            nnode + right_elem.inodes[i],
                        )] += weights[edge_gp] * right_phi * dright_transformed_flux_dy[i];
                    }
                    /*
                    dx.slice_mut(s![ilelem * cell_ngp * cell_ngp + itest_func, ..])
                        .scaled_add(weights[edge_gp] * left_phi, &dleft_transformed_flux_dx);
                    dy.slice_mut(s![ilelem * cell_ngp * cell_ngp + itest_func, ..])
                        .scaled_add(weights[edge_gp] * left_phi, &dleft_transformed_flux_dy);
                    dx.slice_mut(s![irelem * cell_ngp * cell_ngp + itest_func, ..])
                        .scaled_add(weights[edge_gp] * right_phi, &dright_transformed_flux_dx);
                    dy.slice_mut(s![irelem * cell_ngp * cell_ngp + itest_func, ..])
                        .scaled_add(weights[edge_gp] * right_phi, &dright_transformed_flux_dy);
                    */
                }
            }
        }

        for &iedge in self.mesh.boundary_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let ielem = edge.parents[0];
            let elem = &self.mesh.elements[ielem];
            let inodes = &self.mesh.elements[ielem].inodes;
            let mut x_slice = [0.0; 4];
            let mut y_slice = [0.0; 4];
            for i in 0..4 {
                x_slice[i] = self.mesh.nodes[inodes[i]].x;
                y_slice[i] = self.mesh.nodes[inodes[i]].y;
            }
            let sol_slice = solutions.slice(s![ielem, ..]);
            for itest_func in 0..cell_ngp * cell_ngp {
                let itest_func_eta = itest_func / cell_ngp;
                let itest_func_xi = itest_func % cell_ngp;
                for edge_gp in 0..cell_ngp {
                    let (kgp, igp, ref_normal) = match iedge {
                        0 | 1 => (0, edge_gp, [0.0, -1.0]),
                        2 => (edge_gp, cell_ngp - 1, [1.0, 0.0]),
                        3 | 4 => (cell_ngp - 1, edge_gp, [0.0, 1.0]),
                        5 => (edge_gp, 0, [-1.0, 0.0]),
                        _ => unreachable!(),
                    };
                    let eta = basis.cell_gauss_points[kgp];
                    let xi = basis.cell_gauss_points[igp];
                    let (flux, dflux_du) = match iedge {
                        0 => {
                            let u = 2.0;
                            (-u, 0.0)
                        }
                        1 => {
                            let u = 0.0;
                            (u, 0.0)
                        }
                        2 => {
                            let u = sol_slice[kgp * cell_ngp + igp];
                            (self.advection_speed * u, self.advection_speed)
                        }
                        3 | 4 => {
                            let u = sol_slice[igp + kgp * cell_ngp];
                            (u, 1.0)
                        }
                        5 => {
                            let u = 2.0;
                            (-self.advection_speed * u, 0.0)
                        }
                        _ => unreachable!(),
                    };
                    let mut dscaling_dx = [0.0; 4];
                    let mut dscaling_dy = [0.0; 4];
                    let scaling = self.dscaling(
                        eta,
                        xi,
                        ref_normal,
                        x_slice.as_slice(),
                        dscaling_dx.as_mut_slice(),
                        y_slice.as_slice(),
                        dscaling_dy.as_mut_slice(),
                        1.0,
                    );
                    let transformed_flux = scaling * flux;
                    let dtransformed_flux_du = scaling * dflux_du;
                    let dtransformed_flux_dx = array![
                        dscaling_dx[0] * flux,
                        dscaling_dx[1] * flux,
                        dscaling_dx[2] * flux,
                        dscaling_dx[3] * flux,
                    ];
                    let dtransformed_flux_dy = array![
                        dscaling_dy[0] * flux,
                        dscaling_dy[1] * flux,
                        dscaling_dy[2] * flux,
                        dscaling_dy[3] * flux,
                    ];
                    let phi = basis.phis_cell_gps[(itest_func_eta, kgp)]
                        * basis.phis_cell_gps[(itest_func_xi, igp)];
                    residuals[(ielem, itest_func)] += weights[edge_gp] * phi * transformed_flux;
                    if is_enriched {
                        dsol.slice_mut(s![
                            ielem * cell_ngp * cell_ngp + itest_func,
                            ielem * unenriched_cell_ngp * unenriched_cell_ngp
                                ..(ielem + 1) * unenriched_cell_ngp * unenriched_cell_ngp
                        ])
                        .scaled_add(
                            weights[edge_gp] * phi,
                            &(&self.interp_matrix.slice(s![itest_func, ..]) * dtransformed_flux_du),
                        );
                    } else {
                        dsol[(
                            ielem * cell_ngp * cell_ngp + itest_func,
                            ielem * cell_ngp * cell_ngp + kgp * cell_ngp + igp,
                        )] += weights[edge_gp] * dtransformed_flux_du * phi;
                    }
                    for i in 0..4 {
                        dx[(ielem * cell_ngp * cell_ngp + itest_func, elem.inodes[i])] +=
                            weights[edge_gp] * phi * dtransformed_flux_dx[i];
                        dx[(
                            ielem * cell_ngp * cell_ngp + itest_func,
                            nnode + elem.inodes[i],
                        )] += weights[edge_gp] * phi * dtransformed_flux_dy[i];
                    }
                    /*
                    dx.slice_mut(s![ielem * cell_ngp * cell_ngp + itest_func, ..])
                        .scaled_add(weights[edge_gp] * phi, &dtransformed_flux_dx);
                    dy.slice_mut(s![ielem * cell_ngp * cell_ngp + itest_func, ..])
                        .scaled_add(weights[edge_gp] * phi, &dtransformed_flux_dy);
                    */
                }
            }
        }
    }

    #[autodiff(
        dvolume, Reverse, Const, Const, Const, Const, Duplicated, Duplicated, Duplicated, Active
    )]
    fn volume_integral(
        &self,
        basis: &LagrangeBasis1DLobatto,
        itest_func_eta: usize,
        itest_func_xi: usize,
        sol: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> f64 {
        let cell_ngp = basis.cell_gauss_points.len();
        let weights = &basis.cell_gauss_weights;
        let mut res = 0.0;
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let f = space_time_flux1d(sol[igp + kgp * cell_ngp], self.advection_speed);
                let eta = basis.cell_gauss_points[kgp];
                let xi = basis.cell_gauss_points[igp];

                let (jacob_det, jacob_inv_t) = evaluate_jacob(eta, xi, x, y);
                let transformed_f = {
                    [
                        jacob_det * (f[0] * jacob_inv_t[0] + f[1] * jacob_inv_t[2]),
                        jacob_det * (f[0] * jacob_inv_t[1] + f[1] * jacob_inv_t[3]),
                    ]
                };

                // println!("transformed_f: {:?}", transformed_f);

                /*
                println!("itest_func: {:?}", itest_func);
                println!("itest_func_x: {:?}", itest_func_x);
                println!("itest_func_t: {:?}", itest_func_t);
                */

                let test_func_xi = basis.phis_cell_gps[[itest_func_xi, igp]];
                let test_func_eta = basis.phis_cell_gps[[itest_func_eta, kgp]];
                let dtest_func_dxi = basis.dphis_cell_gps[[itest_func_xi, igp]];
                let dtest_func_deta = basis.dphis_cell_gps[[itest_func_eta, kgp]];

                res -= weights[igp]
                    * weights[kgp]
                    * (transformed_f[0] * dtest_func_dxi * test_func_eta
                        + transformed_f[1] * dtest_func_deta * test_func_xi);
            }
        }
        res
    }

    #[autodiff(
        dnum_flux, Reverse, Const, Const, Active, Active, Active, Active, Active, Active, Active
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
    #[autodiff(
        dscaling, Reverse, Const, Const, Const, Const, Duplicated, Duplicated, Active
    )]
    fn compute_flux_scaling(
        &self,
        eta: f64,
        xi: f64,
        ref_normal: [f64; 2],
        x: &[f64],
        y: &[f64],
    ) -> f64 {
        let (jacob_det, jacob_inv_t) = evaluate_jacob(eta, xi, x, y);
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
    fn compute_surface_flux(
        &self,
        left_eta: f64,
        left_xi: f64,
        right_eta: f64,
        right_xi: f64,
        left_ref_normal: [f64; 2],
        right_ref_normal: [f64; 2],
        left_ids: [usize; 4],  // ids in grouped nodes
        right_ids: [usize; 4], // ids in grouped nodes
        left_value: &f64,
        right_value: &f64,
        x: &[f64],
        y: &[f64],
        left_transformed_flux: &mut f64,
        right_transformed_flux: &mut f64,
    ) {
        let left_x = [
            x[left_ids[0]],
            x[left_ids[1]],
            x[left_ids[2]],
            x[left_ids[3]],
        ];
        let left_y = [
            y[left_ids[0]],
            y[left_ids[1]],
            y[left_ids[2]],
            y[left_ids[3]],
        ];
        let right_x = [
            x[right_ids[0]],
            x[right_ids[1]],
            x[right_ids[2]],
            x[right_ids[3]],
        ];
        let right_y = [
            y[right_ids[0]],
            y[right_ids[1]],
            y[right_ids[2]],
            y[right_ids[3]],
        ];
        let normal = compute_normal(x[4], y[4], x[5], y[5]);
        let num_flux = smoothed_upwind(*left_value, *right_value, normal, self.advection_speed);
        let (left_jacob_det, left_jacob_inv_t) =
            evaluate_jacob(left_eta, left_xi, left_x.as_slice(), left_y.as_slice());
        let (right_jacob_det, right_jacob_inv_t) =
            evaluate_jacob(right_eta, right_xi, right_x.as_slice(), right_y.as_slice());
        let left_transformed_normal = {
            [
                left_jacob_inv_t[0] * left_ref_normal[0] + left_jacob_inv_t[1] * left_ref_normal[1],
                left_jacob_inv_t[2] * left_ref_normal[0] + left_jacob_inv_t[3] * left_ref_normal[1],
            ]
        };
        let right_transformed_normal = {
            [
                right_jacob_inv_t[0] * right_ref_normal[0]
                    + right_jacob_inv_t[1] * right_ref_normal[1],
                right_jacob_inv_t[2] * right_ref_normal[0]
                    + right_jacob_inv_t[3] * right_ref_normal[1],
            ]
        };
        let left_normal_magnitude =
            (left_transformed_normal[0].powi(2) + left_transformed_normal[1].powi(2)).sqrt();
        let left_scaling = left_jacob_det * left_normal_magnitude;
        *left_transformed_flux = left_scaling * num_flux;
        let right_normal_magnitude =
            (right_transformed_normal[0].powi(2) + right_transformed_normal[1].powi(2)).sqrt();
        let right_scaling = right_jacob_det * right_normal_magnitude;
        *right_transformed_flux = right_scaling * (-num_flux);
        // (left_transformed_flux, right_transformed_flux)
    }
    fn surface_integral(
        &self,
        itest_func_eta: usize,
        itest_func_xi: usize,
        left_sol: ArrayView1<f64>,
        right_sol: ArrayView1<f64>,
        x: [f64; 6],
        y: [f64; 6],
        left_res: &mut f64,
        right_res: &mut f64,
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        for edge_gp in 0..cell_ngp {
            // Map quadrature points based on edge orientation
            let left_kgp = edge_gp;
            let left_igp = cell_ngp - 1;
            let right_kgp = edge_gp;
            let right_igp = 0;
            let left_eta = self.basis.cell_gauss_points[left_kgp];
            let left_xi = self.basis.cell_gauss_points[left_igp];
            let right_eta = self.basis.cell_gauss_points[right_kgp];
            let right_xi = self.basis.cell_gauss_points[right_igp];
            let left_value = left_sol[left_igp + left_kgp * cell_ngp];
            let right_value = right_sol[right_igp + right_kgp * cell_ngp];
            let left_x = [x[0], x[4], x[5], x[1]];
            let left_y = [y[0], y[4], y[5], y[1]];
            let right_x = [x[4], x[2], x[3], x[5]];
            let right_y = [y[4], y[2], y[3], y[5]];
            let normal = compute_normal(x[4], y[4], x[5], y[5]);

            let num_flux = smoothed_upwind(left_value, right_value, normal, self.advection_speed);

            let (left_jacob_det, left_jacob_inv_t) =
                evaluate_jacob(left_eta, left_xi, left_x.as_slice(), left_y.as_slice());
            let left_n_ref = [1.0, 0.0];
            let left_transformed_normal = {
                [
                    left_jacob_inv_t[0] * left_n_ref[0] + left_jacob_inv_t[1] * left_n_ref[1],
                    left_jacob_inv_t[2] * left_n_ref[0] + left_jacob_inv_t[3] * left_n_ref[1],
                ]
            };
            let left_normal_magnitude =
                (left_transformed_normal[0].powi(2) + left_transformed_normal[1].powi(2)).sqrt();
            let left_scaling = left_jacob_det * left_normal_magnitude;
            let left_transformed_flux = left_scaling * num_flux;

            let (right_jacob_det, right_jacob_inv_t) =
                evaluate_jacob(right_eta, right_xi, right_x.as_slice(), right_y.as_slice());
            let right_n_ref = [-1.0, 0.0];
            let right_transformed_normal = {
                [
                    right_jacob_inv_t[0] * right_n_ref[0] + right_jacob_inv_t[1] * right_n_ref[1],
                    right_jacob_inv_t[2] * right_n_ref[0] + right_jacob_inv_t[3] * right_n_ref[1],
                ]
            };
            let right_normal_magnitude =
                (right_transformed_normal[0].powi(2) + right_transformed_normal[1].powi(2)).sqrt();
            let right_scaling = right_jacob_det * right_normal_magnitude;
            let right_transformed_flux = right_scaling * (-num_flux);
            let left_phi = self.basis.phis_cell_gps[[itest_func_xi, left_igp]]
                * self.basis.phis_cell_gps[[itest_func_eta, left_kgp]];
            let right_phi = self.basis.phis_cell_gps[[itest_func_xi, right_igp]]
                * self.basis.phis_cell_gps[[itest_func_eta, right_kgp]];
            // println!("left_transformed_flux: {:?}", left_transformed_flux);
            // println!("right_transformed_flux: {:?}", right_transformed_flux);
            *left_res += weights[left_kgp] * left_transformed_flux * left_phi;
            *right_res += weights[right_kgp] * right_transformed_flux * right_phi;
        }
    }

    fn boundary_condition(
        &self,
        iedge: usize,
        itest_func_eta: usize,
        itest_func_xi: usize,
        sol: ArrayView1<f64>,
        x: [f64; 4],
        y: [f64; 4],
        res: &mut f64,
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        for edge_gp in 0..cell_ngp {
            let mut kgp: usize = 0;
            let mut igp: usize = 0;
            let mut ref_normal = [0.0; 2];
            let mut flux = 0.0;
            let mut weight = 0.0;

            // lower boundary
            if iedge == 0 || iedge == 1 {
                kgp = 0;
                igp = edge_gp;
                let u = { if iedge == 0 { 2.0 } else { 0.0 } };
                flux = -u;
                ref_normal = [0.0, -1.0];
                weight = weights[igp];
            }
            // right boundary
            if iedge == 2 {
                kgp = edge_gp;
                igp = cell_ngp - 1;
                let u = sol[igp + kgp * cell_ngp];
                flux = self.advection_speed * u;
                ref_normal = [1.0, 0.0];
                weight = weights[kgp];
            }
            // upper boundary
            if iedge == 3 || iedge == 4 {
                kgp = cell_ngp - 1;
                igp = edge_gp;
                let u = sol[igp + kgp * cell_ngp];
                flux = u;
                ref_normal = [0.0, 1.0];
                weight = weights[igp];
            }
            // left boundary
            if iedge == 5 {
                kgp = edge_gp;
                igp = 0;
                let u = 2.0;
                flux = -self.advection_speed * u;
                ref_normal = [-1.0, 0.0];
                weight = weights[kgp];
            }

            //println!("iedge: {}, kgp: {}", iedge, kgp);
            let (jacob_det, jacob_inv_t) = evaluate_jacob(
                self.basis.cell_gauss_points[kgp],
                self.basis.cell_gauss_points[igp],
                x.as_slice(),
                y.as_slice(),
            );
            let transformed_normal = {
                [
                    jacob_inv_t[0] * ref_normal[0] + jacob_inv_t[1] * ref_normal[1],
                    jacob_inv_t[2] * ref_normal[0] + jacob_inv_t[3] * ref_normal[1],
                ]
            };
            let normal_magnitude =
                (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
            let scaling = jacob_det * normal_magnitude;
            let transformed_flux = scaling * flux;
            let phi = self.basis.phis_cell_gps[(itest_func_eta, kgp)]
                * self.basis.phis_cell_gps[(itest_func_xi, igp)];
            *res += weight * phi * transformed_flux;
        }
    }
    pub fn initialize_solution(&mut self, mut solutions: ArrayViewMut2<f64>) {
        let cell_ngp = self.solver_param.cell_gp_num;
        for igp in 0..cell_ngp * cell_ngp {
            solutions[[0, igp]] = 2.0;
            solutions[[1, igp]] = 0.0;
        }
    }
}
