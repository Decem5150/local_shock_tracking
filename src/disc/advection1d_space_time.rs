pub mod boundary_condition;
mod flux;
mod riemann_solver;
mod shock_tracking;
use flux::space_time_flux1d;
use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2,
    ArrayViewMut3, s,
};
use riemann_solver::smoothed_upwind;

use crate::solver::SolverParameters;

use super::{
    basis::lagrange1d::LagrangeBasis1DLobatto,
    mesh::mesh2d::{Element2d, Mesh2d},
};

pub struct Disc1dAdvectionSpaceTime<'a> {
    pub current_iter: usize,
    pub basis: LagrangeBasis1DLobatto,
    pub enriched_basis: LagrangeBasis1DLobatto,
    mesh: &'a Mesh2d,
    solver_param: &'a SolverParameters,
    advection_speed: f64,
    ss_m_mat: Array2<f64>,     // mass matrix of two space polynomials
    ss_im_mat: Array2<f64>,    // inverse mass matrix of two space polynomials
    sst_kxi_mat: Array2<f64>, // spatial stiffness matrix of space polynomial and space-time polynomial
    stst_m_mat: Array2<f64>,  // mass matrix of two space-time polynomials
    stst_im_mat: Array2<f64>, // inverse mass matrix of two space-time polynomials
    stst_kxi_mat: Array2<f64>, // spatial stiffness matrix of two space-time polynomials
    stst_ik1_mat: Array2<f64>, // inverse temporal stiffness matrix of two space-time polynomials
    sts_f0_mat: Array2<f64>,  // mass matrix at relative time 0 for two space-time polynomials
}
impl<'a> Disc1dAdvectionSpaceTime<'a> {
    pub fn solve(&mut self, mut solutions: ArrayViewMut3<f64>) {
        let nelem = self.mesh.elem_num;
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut residuals: Array3<f64> = Array3::zeros((nelem, cell_ngp * cell_ngp, 1));
        let mut enriched_residuals: Array3<f64> =
            Array3::zeros((nelem, (cell_ngp + 1) * (cell_ngp + 1), 1));
        let mut assembled_linear_system: Array3<f64> =
            Array3::zeros((nelem, cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        while self.current_iter < self.solver_param.final_step {
            self.compute_residuals(solutions.view(), residuals.view_mut());
            self.compute_enriched_residuals(solutions.view(), enriched_residuals.view_mut());
            self.current_iter += 1;
        }
    }
    fn interpolate_to_enriched(&self, sol: ArrayView2<f64>) -> Array2<f64> {
        let cell_ngp = self.solver_param.cell_gp_num;
        let enriched_ngp = cell_ngp + 1;
        // Precompute interpolation matrix (standard basis evaluated at enriched points)
        // This could be stored as a class member for efficiency
        let mut interp_matrix = Array2::zeros((enriched_ngp * enriched_ngp, cell_ngp * cell_ngp));

        for i_enr in 0..enriched_ngp {
            let xi_enr = self.enriched_basis.cell_gauss_points[i_enr];

            for j_enr in 0..enriched_ngp {
                let eta_enr = self.enriched_basis.cell_gauss_points[j_enr];
                let enr_idx = i_enr * enriched_ngp + j_enr;

                // Evaluate each standard basis at this enriched point
                for i_std in 0..cell_ngp {
                    for j_std in 0..cell_ngp {
                        let std_idx = i_std * cell_ngp + j_std;

                        // Tensor product of 1D basis functions
                        let phi_xi = self.basis.evaluate_basis_at(i_std, xi_enr);
                        let phi_eta = self.basis.evaluate_basis_at(j_std, eta_enr);
                        interp_matrix[[enr_idx, std_idx]] = phi_xi * phi_eta;
                    }
                }
            }
        }

        // Apply interpolation using ndarray's dot method
        let enriched_sol = interp_matrix.dot(&sol);

        enriched_sol
    }
    fn compute_residuals(&mut self, solutions: ArrayView3<f64>, mut residuals: ArrayViewMut3<f64>) {
        let nelem = self.mesh.elem_num;
        let cell_ngp = self.solver_param.cell_gp_num;
        for ielem in 0..nelem {
            let elem = &self.mesh.elements[ielem];
            let solutions_slice: ArrayView2<f64> = solutions.slice(s![ielem, .., ..]);
            let residuals_slice: ArrayViewMut2<f64> = residuals.slice_mut(s![ielem, .., ..]);
            self.volume_integral(solutions_slice, residuals_slice, elem);
        }
        for &iedge in self.mesh.internal_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let ilelem = edge.parent_elements[0];
            let irelem = edge.parent_elements[1];
            let left_elem = &self.mesh.elements[ilelem as usize];
            let right_elem = &self.mesh.elements[irelem as usize];
            let left_local_id = edge.local_ids[0] as usize;
            let right_local_id = edge.local_ids[1] as usize;
            let left_sol = match left_local_id {
                0 => solutions.slice(s![ilelem, 0..cell_ngp, ..]),
                1 => solutions.slice(s![ilelem, cell_ngp..; cell_ngp, ..]),
                2 => solutions.slice(s![ilelem, (-(cell_ngp as isize)).., ..]),
                3 => solutions.slice(s![ilelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]),
                _ => panic!("Invalid left local id"),
            };
            let right_sol = match right_local_id {
                0 => solutions.slice(s![irelem, 0..cell_ngp, ..]),
                1 => solutions.slice(s![irelem, cell_ngp..; cell_ngp, ..]),
                2 => solutions.slice(s![irelem, (-(cell_ngp as isize)).., ..]),
                3 => solutions.slice(s![irelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]),
                _ => panic!("Invalid right local id"),
            };
            let (left_res, right_res) =
                residuals.multi_slice_mut((s![ilelem, .., ..], s![irelem, .., ..]));
            let normal = edge.normal;
            self.surface_integral(
                left_sol,
                right_sol,
                left_res,
                right_res,
                left_elem,
                right_elem,
                left_local_id,
                right_local_id,
                normal,
            );
        }
    }
    fn compute_enriched_residuals(
        &mut self,
        solutions: ArrayView3<f64>,
        mut enriched_residuals: ArrayViewMut3<f64>,
    ) {
        let nelem = self.mesh.elem_num;
        let cell_ngp = self.solver_param.cell_gp_num;
        let enriched_ngp = cell_ngp + 1;
        let mut enriched_sol = Array3::zeros((nelem, enriched_ngp * enriched_ngp, 1));
        for ielem in 0..nelem {
            let solutions_slice: ArrayView2<f64> = solutions.slice(s![ielem, .., ..]);
            let mut enriched_sol_slice: ArrayViewMut2<f64> =
                enriched_sol.slice_mut(s![ielem, .., ..]);
            enriched_sol_slice.assign(&self.interpolate_to_enriched(solutions_slice));
        }
        for ielem in 0..nelem {
            let elem = &self.mesh.elements[ielem];
            let enriched_sol_slice: ArrayView2<f64> = enriched_sol.slice(s![ielem, .., ..]);
            let enriched_residuals_slice: ArrayViewMut2<f64> =
                enriched_residuals.slice_mut(s![ielem, .., ..]);
            self.volume_integral(enriched_sol_slice, enriched_residuals_slice, elem);
        }
        for &iedge in self.mesh.internal_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let ilelem = edge.parent_elements[0];
            let irelem = edge.parent_elements[1];
            let left_elem = &self.mesh.elements[ilelem as usize];
            let right_elem = &self.mesh.elements[irelem as usize];
            let left_local_id = edge.local_ids[0] as usize;
            let right_local_id = edge.local_ids[1] as usize;
            let left_sol = match left_local_id {
                0 => enriched_sol.slice(s![ilelem, 0..enriched_ngp, ..]),
                1 => enriched_sol.slice(s![ilelem, cell_ngp..; cell_ngp, ..]),
                2 => enriched_sol.slice(s![ilelem, (-(enriched_ngp as isize)).., ..]),
                3 => {
                    enriched_sol.slice(s![ilelem, 0..=(-(enriched_ngp as isize)); enriched_ngp, ..])
                }
                _ => panic!("Invalid left local id"),
            };
            let right_sol = match right_local_id {
                0 => enriched_sol.slice(s![irelem, 0..enriched_ngp, ..]),
                1 => enriched_sol.slice(s![irelem, cell_ngp..; cell_ngp, ..]),
                2 => enriched_sol.slice(s![irelem, (-(enriched_ngp as isize)).., ..]),
                3 => {
                    enriched_sol.slice(s![irelem, 0..=(-(enriched_ngp as isize)); enriched_ngp, ..])
                }
                _ => panic!("Invalid right local id"),
            };
            let (left_res, right_res) =
                enriched_residuals.multi_slice_mut((s![ilelem, .., ..], s![irelem, .., ..]));
            let normal = edge.normal;
            self.surface_integral(
                left_sol,
                right_sol,
                left_res,
                right_res,
                left_elem,
                right_elem,
                left_local_id,
                right_local_id,
                normal,
            );
        }
    }
    fn volume_integral(
        &mut self,
        sol: ArrayView2<f64>,
        mut res: ArrayViewMut2<f64>,
        elem: &Element2d,
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut lfh: Array4<f64> = Array4::zeros((2, cell_ngp, cell_ngp, 1));
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let f: Array1<f64> = space_time_flux1d(sol[[kgp, igp]], self.advection_speed);
                let transformed_f: Array1<f64> = elem.jacob_det[[kgp, igp]]
                    * f.dot(&elem.jacob_inv_t.slice(s![kgp, igp, .., ..]));
                lfh.slice_mut(s![.., kgp, igp, 0]).assign(&transformed_f);
            }
        }
        let lfh_slice: ArrayView3<f64> =
            lfh.view().into_shape((2, cell_ngp * cell_ngp, 1)).unwrap();
        let f_slice: ArrayView2<f64> = lfh_slice.slice(s![0, .., ..]);
        let g_slice: ArrayView2<f64> = lfh_slice.slice(s![1, .., ..]);
        res.scaled_add(1.0, &self.stst_kxi_mat.dot(&f_slice));
        res.scaled_add(1.0, &self.stst_kxi_mat.dot(&g_slice));
    }
    #[allow(clippy::too_many_arguments)]
    fn surface_integral(
        &self,
        left_sol: ArrayView2<f64>,         // (ntgp, neq)
        right_sol: ArrayView2<f64>,        // (ntgp, neq)
        mut left_res: ArrayViewMut2<f64>,  // (nxdof, neq)
        mut right_res: ArrayViewMut2<f64>, // (nxdof, neq)
        left_elem: &Element2d,
        right_elem: &Element2d,
        left_local_id: usize,
        right_local_id: usize,
        normal: [f64; 2],
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        for ibasis in 0..cell_ngp {
            for kgp in 0..cell_ngp {
                // Map quadrature points based on edge orientation
                let left_kgp = match left_local_id {
                    0 => kgp,                // Bottom: natural order
                    1 => kgp,                // Right: natural order
                    2 => cell_ngp - 1 - kgp, // Top: reversed
                    3 => cell_ngp - 1 - kgp, // Left: reversed
                    _ => panic!("Invalid edge"),
                };
                let right_kgp = match right_local_id {
                    0 => kgp,                // Bottom: natural order
                    1 => kgp,                // Right: natural order
                    2 => cell_ngp - 1 - kgp, // Top: reversed
                    3 => cell_ngp - 1 - kgp, // Left: reversed
                    _ => panic!("Invalid edge"),
                };

                let left_value: ArrayView1<f64> = left_sol.slice(s![left_kgp, ..]);
                let right_value: ArrayView1<f64> = right_sol.slice(s![right_kgp, ..]);
                let num_flux: Array1<f64> =
                    smoothed_upwind(left_value, right_value, normal, self.advection_speed);

                let left_jacob_inv_t = left_elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
                let left_n_ref = match left_local_id {
                    0 => [0.0, -1.0], // Bottom edge
                    1 => [1.0, 0.0],  // Right edge
                    2 => [0.0, 1.0],  // Top edge
                    3 => [-1.0, 0.0], // Left edge
                    _ => panic!("Invalid edge"),
                };
                let left_n_ref_array = Array1::from_vec(left_n_ref.to_vec());
                let left_transformed_normal: Array1<f64> = left_n_ref_array.dot(&left_jacob_inv_t);
                let left_normal_magnitude = (left_transformed_normal[0].powi(2)
                    + left_transformed_normal[1].powi(2))
                .sqrt();
                let left_scaling = left_elem.jacob_det * left_normal_magnitude;
                let left_transformed_flux = left_scaling * &num_flux;

                let right_jacob_inv_t = right_elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
                let right_n_ref = match right_local_id {
                    0 => [0.0, -1.0], // Bottom edge
                    1 => [1.0, 0.0],  // Right edge
                    2 => [0.0, 1.0],  // Top edge
                    3 => [-1.0, 0.0], // Left edge
                    _ => panic!("Invalid edge"),
                };
                let right_n_ref_array = Array1::from_vec(right_n_ref.to_vec());
                let right_transformed_normal: Array1<f64> =
                    right_n_ref_array.dot(&right_jacob_inv_t);
                let right_normal_magnitude = (right_transformed_normal[0].powi(2)
                    + right_transformed_normal[1].powi(2))
                .sqrt();
                let right_scaling = right_elem.jacob_det * right_normal_magnitude;
                let right_transformed_flux = right_scaling * &num_flux;
                let left_phi = match left_local_id {
                    0 => {
                        self.basis.phis_cell_gps[[left_kgp, ibasis]]
                            * self.basis.phis_cell_gps[[0, ibasis]]
                    }
                    1 => {
                        self.basis.phis_cell_gps[[cell_ngp - 1, ibasis]]
                            * self.basis.phis_cell_gps[[left_kgp, ibasis]]
                    }
                    2 => {
                        self.basis.phis_cell_gps[[left_kgp, ibasis]]
                            * self.basis.phis_cell_gps[[cell_ngp - 1, ibasis]]
                    }
                    3 => {
                        self.basis.phis_cell_gps[[0, ibasis]]
                            * self.basis.phis_cell_gps[[left_kgp, ibasis]]
                    }
                    _ => panic!("Invalid edge"),
                };
                let right_phi = match right_local_id {
                    0 => {
                        self.basis.phis_cell_gps[[right_kgp, ibasis]]
                            * self.basis.phis_cell_gps[[0, ibasis]]
                    }
                    1 => {
                        self.basis.phis_cell_gps[[cell_ngp - 1, ibasis]]
                            * self.basis.phis_cell_gps[[right_kgp, ibasis]]
                    }
                    2 => {
                        self.basis.phis_cell_gps[[right_kgp, ibasis]]
                            * self.basis.phis_cell_gps[[cell_ngp - 1, ibasis]]
                    }
                    3 => {
                        self.basis.phis_cell_gps[[0, ibasis]]
                            * self.basis.phis_cell_gps[[right_kgp, ibasis]]
                    }
                    _ => panic!("Invalid edge"),
                };
                left_res[[ibasis, 0]] -= weights[left_kgp] * left_transformed_flux[0] * left_phi;
                right_res[[ibasis, 0]] +=
                    weights[right_kgp] * right_transformed_flux[0] * right_phi;
            }
        }
    }
    fn compute_residual_derivatives(
        &self,
        sol: ArrayView2<f64>,
        elem: &Element2d,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        let inodes = elem.inodes;
        let mut x = Array1::zeros(4);
        let mut y = Array1::zeros(4);
        let a = self.advection_speed;
        for i in 0..4 {
            x[i] = self.mesh.nodes[inodes[i] as usize].x;
            y[i] = self.mesh.nodes[inodes[i] as usize].y;
        }
        let mut dr_du: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, 1));
        let mut dr_dx: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, 1));
        let mut dr_dy: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, 1));
        // derivatives of volume integral
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let u = sol[[igp, kgp]];
                let xi = self.basis.cell_gauss_points[igp];
                let eta = self.basis.cell_gauss_points[kgp];
                let mut dN_dxi: Array1<f64> = Array1::zeros(4);
                let mut dN_deta: Array1<f64> = Array1::zeros(4);
                dN_dxi[0] = 0.25 * eta - 0.25;
                dN_deta[0] = 0.25 * xi - 0.25;
                dN_dxi[1] = 0.25 * eta + 0.25;
                dN_deta[1] = 0.25 * xi - 0.25;
                dN_dxi[2] = 0.25 * eta + 0.25;
                dN_deta[2] = 0.25 * xi + 0.25;
                dN_dxi[3] = 0.25 * eta - 0.25;
                for ibasis in 0..cell_ngp {
                    let dphi_dxi = self.basis.dphis_cell_gps[[igp, ibasis]];
                    let dphi_deta = self.basis.dphis_cell_gps[[kgp, ibasis]];
                    let mut volume_du_sum = 0.0;
                    let mut volume_dx_sum = 0.0;
                    let mut volume_dy_sum = 0.0;
                    for i in 0..4 {
                        volume_du_sum += y[i]
                            * (a * dphi_dxi * dN_deta[i] - a * dphi_deta * dN_dxi[i])
                            + x[i] * (dphi_deta * dN_dxi[i] - dphi_dxi * dN_deta[i]);
                        volume_dx_sum += u * (dphi_deta * dN_dxi[i] - dphi_dxi * dN_deta[i]);
                        volume_dy_sum += a * u * (dphi_dxi * dN_deta[i] - dphi_deta * dN_dxi[i]);
                    }
                    dr_du[[ibasis, 0]] += weights[igp] * weights[kgp] * volume_du_sum;
                    dr_dx[[ibasis, 0]] += weights[igp] * weights[kgp] * volume_dx_sum;
                    dr_dy[[ibasis, 0]] += weights[igp] * weights[kgp] * volume_dy_sum;
                }
            }
        }
        // derivatives of surface integral
        for &iedge in self.mesh.internal_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let left_elem = &self.mesh.elements[edge.parent_elements[0] as usize];
            let right_elem = &self.mesh.elements[edge.parent_elements[1] as usize];
        }
        (dr_du, dr_dx, dr_dy)
    }
    fn compute_enriched_residual_derivatives(
        &mut self,
        solutions: ArrayView3<f64>,
        mut residuals: ArrayViewMut3<f64>,
    ) {
    }
}
