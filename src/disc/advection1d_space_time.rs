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
            let left_sol = solutions.slice(s![ilelem, (cell_ngp - 1)..; cell_ngp, ..]);
            let right_sol = solutions.slice(s![irelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]);
            /*
            let left_sol = match left_local_id {
                0 => solutions.slice(s![ilelem, 0..cell_ngp, ..]),
                1 => solutions.slice(s![ilelem, (cell_ngp - 1)..; cell_ngp, ..]),
                2 => solutions.slice(s![ilelem, (-(cell_ngp as isize)).., ..]),
                3 => solutions.slice(s![ilelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]),
                _ => panic!("Invalid left local id"),
            };
            let right_sol = match right_local_id {
                0 => solutions.slice(s![irelem, 0..cell_ngp, ..]),
                1 => solutions.slice(s![irelem, (cell_ngp - 1)..; cell_ngp, ..]),
                2 => solutions.slice(s![irelem, (-(cell_ngp as isize)).., ..]),
                3 => solutions.slice(s![irelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]),
                _ => panic!("Invalid right local id"),
            };
            */
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
            let left_sol = enriched_sol.slice(s![ilelem, (cell_ngp - 1)..; cell_ngp, ..]);
            let right_sol =
                enriched_sol.slice(s![irelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]);
            /*
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
            */
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
                let left_kgp = kgp;
                let right_kgp = kgp;
                /*
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
                */
                let left_value: ArrayView1<f64> = left_sol.slice(s![kgp, ..]);
                let right_value: ArrayView1<f64> = right_sol.slice(s![kgp, ..]);
                let num_flux =
                    smoothed_upwind(left_value[0], right_value[0], normal, self.advection_speed);

                let left_jacob_inv_t: ArrayView2<f64> =
                    left_elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
                let left_n_ref = [1.0, 0.0];
                /*
                let left_n_ref = match left_local_id {
                    0 => [0.0, -1.0], // Bottom edge
                    1 => [1.0, 0.0],  // Right edge
                    2 => [0.0, 1.0],  // Top edge
                    3 => [-1.0, 0.0], // Left edge
                    _ => panic!("Invalid edge"),
                };
                */
                let left_n_ref_array = Array1::from_vec(left_n_ref.to_vec());
                let left_transformed_normal: Array1<f64> = left_jacob_inv_t.dot(&left_n_ref_array);
                let left_normal_magnitude = (left_transformed_normal[0].powi(2)
                    + left_transformed_normal[1].powi(2))
                .sqrt();
                let left_scaling = left_elem.jacob_det * left_normal_magnitude;
                let left_transformed_flux = left_scaling * &num_flux;

                let right_jacob_inv_t: ArrayView2<f64> =
                    right_elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
                let right_n_ref = [-1.0, 0.0];
                /*
                let right_n_ref = match right_local_id {
                    0 => [0.0, -1.0], // Bottom edge
                    1 => [1.0, 0.0],  // Right edge
                    2 => [0.0, 1.0],  // Top edge
                    3 => [-1.0, 0.0], // Left edge
                    _ => panic!("Invalid edge"),
                };
                */
                let right_n_ref_array = Array1::from_vec(right_n_ref.to_vec());
                let right_transformed_normal: Array1<f64> =
                    right_jacob_inv_t.to_owned().dot(&right_n_ref_array);
                let right_normal_magnitude = (right_transformed_normal[0].powi(2)
                    + right_transformed_normal[1].powi(2))
                .sqrt();
                let right_scaling = right_elem.jacob_det * right_normal_magnitude;
                let right_transformed_flux = right_scaling * &num_flux;
                let left_phi = self.basis.phis_cell_gps[[cell_ngp - 1, ibasis]]
                    * self.basis.phis_cell_gps[[left_kgp, ibasis]];
                let right_phi = self.basis.phis_cell_gps[[0, ibasis]]
                    * self.basis.phis_cell_gps[[right_kgp, ibasis]];
                /*
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
                */
                left_res[[ibasis, 0]] -= weights[left_kgp] * left_transformed_flux[0] * left_phi;
                right_res[[ibasis, 0]] +=
                    weights[right_kgp] * right_transformed_flux[0] * right_phi;
            }
        }
    }
    fn volume_residual_derivatives(
        &self,
        sol: ArrayView2<f64>,
        ielem: usize,
    ) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let elem = &self.mesh.elements[ielem];
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        let inodes = &elem.inodes;
        let mut x = Array1::zeros(4);
        let mut y = Array1::zeros(4);
        let a = self.advection_speed;
        for i in 0..4 {
            x[i] = self.mesh.nodes[inodes[i] as usize].x;
            y[i] = self.mesh.nodes[inodes[i] as usize].y;
        }
        let mut dr_du: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp, 1)); // (ntest_func, ntrial_func, neq)
        let mut dr_dx: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, 4, 1)); // (ntest_func, ntrial_func, neq)
        let mut dr_dy: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, 4, 1)); // (ntest_func, ntrial_func, neq)
        // derivatives of volume integral
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let u = sol[[igp, kgp]];
                let xi = self.basis.cell_gauss_points[igp];
                let eta = self.basis.cell_gauss_points[kgp];
                let mut dn_dxi: Array1<f64> = Array1::zeros(4);
                let mut dn_deta: Array1<f64> = Array1::zeros(4);
                dn_dxi[0] = 0.25 * eta - 0.25;
                dn_deta[0] = 0.25 * xi - 0.25;
                dn_dxi[1] = 0.25 * eta + 0.25;
                dn_deta[1] = 0.25 * xi - 0.25;
                dn_dxi[2] = 0.25 * eta + 0.25;
                dn_deta[2] = 0.25 * xi + 0.25;
                dn_dxi[3] = 0.25 * eta - 0.25;
                dn_deta[3] = 0.25 * xi + 0.25;
                let y_dot_dn_dxi = y.dot(&dn_dxi);
                let x_dot_dn_dxi = x.dot(&dn_dxi);
                let y_dot_dn_deta = y.dot(&dn_deta);
                let x_dot_dn_deta = x.dot(&dn_deta);
                for itest_func in 0..(cell_ngp * cell_ngp) {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    let test_func_xi = self.basis.phis_cell_gps[[igp, itest_func_x]];
                    let test_func_eta = self.basis.phis_cell_gps[[kgp, itest_func_t]];
                    let dtest_func_dxi = self.basis.dphis_cell_gps[[igp, itest_func_x]];
                    let dtest_func_deta = self.basis.dphis_cell_gps[[kgp, itest_func_t]];
                    for itrial_func in 0..(cell_ngp * cell_ngp) {
                        let itrial_func_x = itrial_func % cell_ngp; // spatial index
                        let itrial_func_t = itrial_func / cell_ngp; // temporal index
                        let trial_func_xi = self.basis.phis_cell_gps[[igp, itrial_func_x]];
                        let trial_func_eta = self.basis.phis_cell_gps[[kgp, itrial_func_t]];
                        dr_du[[itest_func, itrial_func, 0]] += (weights[igp] * weights[kgp])
                            * (trial_func_xi * trial_func_eta)
                            * ((test_func_xi * dtest_func_deta)
                                * ((-a * y_dot_dn_dxi) + x_dot_dn_dxi)
                                - (test_func_eta * dtest_func_dxi)
                                    * ((-a * x_dot_dn_deta) + y_dot_dn_deta));
                    }

                    for inode in 0..4 {
                        dr_dx[[itest_func, inode, 0]] += (weights[igp] * weights[kgp])
                            * u
                            * (test_func_xi * dtest_func_deta * dn_dxi[inode]
                                - test_func_eta * dtest_func_dxi * dn_deta[inode]);
                        dr_dy[[itest_func, inode, 0]] += (weights[igp] * weights[kgp])
                            * (a * u)
                            * (test_func_eta * dtest_func_dxi * dn_deta[inode]
                                - test_func_xi * dtest_func_deta * dn_dxi[inode]);
                    }
                }
            }
        }
        (dr_du, dr_dx, dr_dy)
    }
    fn volume_enriched_residual_derivatives(
        &self,
        enriched_sol: ArrayView2<f64>,
        ielem: usize,
    ) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let elem = &self.mesh.elements[ielem];
        let cell_ngp = self.solver_param.cell_gp_num;
        let enriched_ngp = cell_ngp + 1;
        let weights = &self.enriched_basis.cell_gauss_weights;
        let inodes = &elem.inodes;
        let mut x = Array1::zeros(4);
        let mut y = Array1::zeros(4);
        let a = self.advection_speed;
        for i in 0..4 {
            x[i] = self.mesh.nodes[inodes[i] as usize].x;
            y[i] = self.mesh.nodes[inodes[i] as usize].y;
        }
        let mut denrr_du: Array3<f64> =
            Array3::zeros((enriched_ngp * enriched_ngp, cell_ngp * cell_ngp, 1));
        let mut denrr_dx: Array3<f64> = Array3::zeros((enriched_ngp * enriched_ngp, 4, 1));
        let mut denrr_dy: Array3<f64> = Array3::zeros((enriched_ngp * enriched_ngp, 4, 1));
        // derivatives of volume integral
        for kgp in 0..enriched_ngp {
            for igp in 0..enriched_ngp {
                let u = enriched_sol[[kgp, igp]];
                let xi = self.enriched_basis.cell_gauss_points[igp];
                let eta = self.enriched_basis.cell_gauss_points[kgp];
                let mut dn_dxi: Array1<f64> = Array1::zeros(4);
                let mut dn_deta: Array1<f64> = Array1::zeros(4);
                dn_dxi[0] = 0.25 * eta - 0.25;
                dn_deta[0] = 0.25 * xi - 0.25;
                dn_dxi[1] = 0.25 * eta + 0.25;
                dn_deta[1] = 0.25 * xi - 0.25;
                dn_dxi[2] = 0.25 * eta + 0.25;
                dn_deta[2] = 0.25 * xi + 0.25;
                dn_dxi[3] = 0.25 * eta - 0.25;
                dn_deta[3] = 0.25 * xi + 0.25;
                let y_dot_dn_dxi = y.dot(&dn_dxi);
                let x_dot_dn_dxi = x.dot(&dn_dxi);
                let y_dot_dn_deta = y.dot(&dn_deta);
                let x_dot_dn_deta = x.dot(&dn_deta);
                for itest_func in 0..(cell_ngp + 1) * (cell_ngp + 1) {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    let test_func_xi = self.enriched_basis.phis_cell_gps[[igp, itest_func_x]];
                    let test_func_eta = self.enriched_basis.phis_cell_gps[[kgp, itest_func_t]];
                    let dtest_func_dxi = self.enriched_basis.dphis_cell_gps[[igp, itest_func_x]];
                    let dtest_func_deta = self.enriched_basis.dphis_cell_gps[[kgp, itest_func_t]];
                    for itrial_func in 0..(cell_ngp * cell_ngp) {
                        let itrial_func_x = itrial_func % cell_ngp; // spatial index
                        let itrial_func_t = itrial_func / cell_ngp; // temporal index
                        let trial_func_xi = self.basis.phis_cell_gps[[igp, itrial_func_x]];
                        let trial_func_eta = self.basis.phis_cell_gps[[kgp, itrial_func_t]];
                        denrr_du[[itest_func, itrial_func, 0]] += (weights[igp] * weights[kgp])
                            * (trial_func_xi * trial_func_eta)
                            * ((test_func_xi * dtest_func_deta)
                                * ((-a * y_dot_dn_dxi) + x_dot_dn_dxi)
                                - (test_func_eta * dtest_func_dxi)
                                    * ((-a * x_dot_dn_deta) + y_dot_dn_deta));
                    }

                    for inode in 0..4 {
                        denrr_dx[[itest_func, inode, 0]] += (weights[igp] * weights[kgp])
                            * u
                            * (test_func_xi * dtest_func_deta * dn_dxi[inode]
                                - test_func_eta * dtest_func_dxi * dn_deta[inode]);
                        denrr_dy[[itest_func, inode, 0]] += (weights[igp] * weights[kgp])
                            * (a * u)
                            * (test_func_eta * dtest_func_dxi * dn_deta[inode]
                                - test_func_xi * dtest_func_deta * dn_dxi[inode]);
                    }
                }
            }
        }
        (denrr_du, denrr_dx, denrr_dy)
    }
    fn surface_residual_derivatives(
        &self,
        sol: ArrayView3<f64>,
        iedge: usize,
    ) -> (
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
        Array3<f64>,
    ) {
        let edge = &self.mesh.edges[iedge];
        let ilelem = edge.parent_elements[0];
        let irelem = edge.parent_elements[1];
        let left_elem = &self.mesh.elements[ilelem as usize];
        let right_elem = &self.mesh.elements[irelem as usize];
        let left_local_id = edge.local_ids[0] as usize;
        let right_local_id = edge.local_ids[1] as usize;
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        let left_sol = sol.slice(s![ilelem, (cell_ngp - 1)..; cell_ngp, ..]);
        let right_sol = sol.slice(s![irelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]);
        /*
        let left_sol = match left_local_id {
            0 => sol.slice(s![ilelem, 0..cell_ngp, ..]),
            1 => sol.slice(s![ilelem, (cell_ngp - 1)..; cell_ngp, ..]),
            2 => sol.slice(s![ilelem, (-(cell_ngp as isize)).., ..]),
            3 => sol.slice(s![ilelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]),
            _ => panic!("Invalid left local id"),
        };
        let right_sol = match right_local_id {
            0 => sol.slice(s![irelem, 0..cell_ngp, ..]),
            1 => sol.slice(s![irelem, (cell_ngp - 1)..; cell_ngp, ..]),
            2 => sol.slice(s![irelem, (-(cell_ngp as isize)).., ..]),
            3 => sol.slice(s![irelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]),
            _ => panic!("Invalid right local id"),
        };
        */
        let normal = edge.normal;
        let left_n_ref = [1.0, 0.0];
        let right_n_ref = [-1.0, 0.0];
        let mut left_dr_du: Array3<f64> =
            Array3::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp, 1)); // (ntest_func, ntrial_func, neq)
        let mut left_dr_dx: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, 4, 1)); // (ntest_func, ntrial_func, neq)
        let mut left_dr_dy: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, 4, 1)); // (ntest_func, ntrial_func, neq)
        let mut right_dr_du: Array3<f64> =
            Array3::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp, 1)); // (ntest_func, ntrial_func, neq)
        let mut right_dr_dx: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, 4, 1)); // (ntest_func, ntrial_func, neq)
        let mut right_dr_dy: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, 4, 1)); // (ntest_func, ntrial_func, neq)
        let mut left_x = Array1::zeros(4);
        let mut left_y = Array1::zeros(4);
        let mut right_x = Array1::zeros(4);
        let mut right_y = Array1::zeros(4);
        for i in 0..4 {
            left_x[i] = self.mesh.nodes[left_elem.inodes[i] as usize].x;
            left_y[i] = self.mesh.nodes[left_elem.inodes[i] as usize].y;
            right_x[i] = self.mesh.nodes[right_elem.inodes[i] as usize].x;
            right_y[i] = self.mesh.nodes[right_elem.inodes[i] as usize].y;
        }
        for kgp in 0..cell_ngp {
            let left_kgp = kgp;
            let right_kgp = kgp;
            let left_value = left_sol.slice(s![kgp, ..]);
            let right_value = right_sol.slice(s![kgp, ..]);
            let left_jacob_det = left_elem.jacob_det[[kgp, cell_ngp - 1]];
            let right_jacob_det = right_elem.jacob_det[[kgp, 0]];
            let left_jacob_inv_t = left_elem.jacob_inv_t.slice(s![kgp, cell_ngp - 1, .., ..]);
            let right_jacob_inv_t = right_elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
            let left_xi = self.basis.cell_gauss_points[cell_ngp - 1];
            let right_xi = self.basis.cell_gauss_points[0];
            let left_eta = self.basis.cell_gauss_points[kgp];
            let right_eta = self.basis.cell_gauss_points[kgp];
            let mut left_dn_dxi: Array1<f64> = Array1::zeros(4);
            let mut left_dn_deta: Array1<f64> = Array1::zeros(4);
            left_dn_dxi[0] = 0.25 * left_eta - 0.25;
            left_dn_deta[0] = 0.25 * left_xi - 0.25;
            left_dn_dxi[1] = 0.25 * left_eta + 0.25;
            left_dn_deta[1] = 0.25 * left_xi - 0.25;
            left_dn_dxi[2] = 0.25 * left_eta + 0.25;
            left_dn_deta[2] = 0.25 * left_xi + 0.25;
            left_dn_dxi[3] = 0.25 * left_eta - 0.25;
            left_dn_deta[3] = 0.25 * left_xi + 0.25;
            let left_y_dot_dn_dxi = left_y.dot(&left_dn_dxi);
            let left_x_dot_dn_dxi = left_x.dot(&left_dn_dxi);
            let left_y_dot_dn_deta = left_y.dot(&left_dn_deta);
            let left_x_dot_dn_deta = left_x.dot(&left_dn_deta);
            let mut right_dn_dxi: Array1<f64> = Array1::zeros(4);
            let mut right_dn_deta: Array1<f64> = Array1::zeros(4);
            right_dn_dxi[0] = 0.25 * right_eta - 0.25;
            right_dn_deta[0] = 0.25 * right_xi - 0.25;
            right_dn_dxi[1] = 0.25 * right_eta + 0.25;
            right_dn_deta[1] = 0.25 * right_xi - 0.25;
            right_dn_dxi[2] = 0.25 * right_eta + 0.25;
            right_dn_deta[2] = 0.25 * right_xi + 0.25;
            right_dn_dxi[3] = 0.25 * right_eta - 0.25;
            right_dn_deta[3] = 0.25 * right_xi + 0.25;
            let right_y_dot_dn_dxi = right_y.dot(&right_dn_dxi);
            let right_x_dot_dn_dxi = right_x.dot(&right_dn_dxi);
            let right_y_dot_dn_deta = right_y.dot(&right_dn_deta);
            let right_x_dot_dn_deta = right_x.dot(&right_dn_deta);

            // Map quadrature points based on edge orientation
            /*
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
            */
            for itest_func in 0..(cell_ngp * cell_ngp) {
                let itest_func_x = itest_func % cell_ngp; // spatial index
                let itest_func_t = itest_func / cell_ngp; // temporal index
                let left_phi = self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]]
                    * self.basis.phis_cell_gps[[left_kgp, itest_func_t]];
                let right_phi = self.basis.phis_cell_gps[[0, itest_func_x]]
                    * self.basis.phis_cell_gps[[right_kgp, itest_func_t]];
                /*
                let (test_func_xi, test_func_eta) = match left_local_id {
                    0 => (
                        self.basis.phis_cell_gps[[left_kgp, itest_func_x]],
                        self.basis.phis_cell_gps[[0, itest_func_t]],
                    ),
                    1 => (
                        self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]],
                        self.basis.phis_cell_gps[[left_kgp, itest_func_t]],
                    ),
                    2 => (
                        self.basis.phis_cell_gps[[left_kgp, itest_func_t]],
                        self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]],
                    ),
                    3 => (
                        self.basis.phis_cell_gps[[0, itest_func_t]],
                        self.basis.phis_cell_gps[[left_kgp, itest_func_x]],
                    ),
                    _ => panic!("Invalid left local id"),
                };
                let (right_test_func_xi, right_test_func_eta) = match right_local_id {
                    0 => (
                        self.basis.phis_cell_gps[[right_kgp, itest_func_x]],
                        self.basis.phis_cell_gps[[0, itest_func_t]],
                    ),
                    1 => (
                        self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]],
                        self.basis.phis_cell_gps[[right_kgp, itest_func_t]],
                    ),
                    2 => (
                        self.basis.phis_cell_gps[[right_kgp, itest_func_t]],
                        self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]],
                    ),
                    3 => (
                        self.basis.phis_cell_gps[[0, itest_func_x]],
                        self.basis.phis_cell_gps[[right_kgp, itest_func_t]],
                    ),
                    _ => panic!("Invalid right local id"),
                };
                */
                let numerical_flux = riemann_solver::smoothed_upwind(
                    left_value[0],
                    right_value[0],
                    normal,
                    self.advection_speed,
                );
                let dflux_dul = riemann_solver::dflux_dul(normal, self.advection_speed);
                let dflux_dur = riemann_solver::dflux_dur(normal, self.advection_speed);
                let (dflux_dnx, dflux_dny) = riemann_solver::dflux_dnormal(
                    left_value[0],
                    right_value[0],
                    normal,
                    self.advection_speed,
                );
                let left_n_ref_array: Array1<f64> = Array1::from_vec(left_n_ref.to_vec());
                let left_transformed_normal: Array1<f64> = left_jacob_inv_t.dot(&left_n_ref_array);
                let left_normal_magnitude = (left_transformed_normal[0].powi(2)
                    + left_transformed_normal[1].powi(2))
                .sqrt();
                let left_scaling = left_jacob_det * left_normal_magnitude;

                let right_n_ref_array: Array1<f64> = Array1::from_vec(right_n_ref.to_vec());
                let right_transformed_normal: Array1<f64> =
                    right_jacob_inv_t.dot(&right_n_ref_array);
                let right_normal_magnitude = (right_transformed_normal[0].powi(2)
                    + right_transformed_normal[1].powi(2))
                .sqrt();
                let right_scaling = right_jacob_det * right_normal_magnitude;

                left_dr_du[[itest_func, cell_ngp - 1 + kgp * cell_ngp, 0]] =
                    left_scaling * dflux_dul * left_phi;
                right_dr_du[[itest_func, kgp * cell_ngp, 0]] =
                    right_scaling * (-dflux_dur) * right_phi;
                for inode in 0..4 {
                    let left_scaling_x_i = {
                        let numerator = (left_n_ref[0] * left_x_dot_dn_deta
                            - left_n_ref[1] * left_x_dot_dn_dxi)
                            * (left_n_ref[0] * left_dn_deta[inode]
                                - left_n_ref[1] * left_dn_dxi[inode]);
                        let denominator = ((left_n_ref[0] * left_x_dot_dn_deta
                            - left_n_ref[1] * left_x_dot_dn_dxi)
                            .powf(2.0)
                            + (left_n_ref[0] * left_y_dot_dn_deta
                                - left_n_ref[1] * left_y_dot_dn_dxi)
                                .powf(2.0))
                        .sqrt();
                        numerator / denominator
                    };
                    let left_scaling_y_i = {
                        let numerator = (left_n_ref[0] * left_y_dot_dn_deta
                            - left_n_ref[1] * left_y_dot_dn_dxi)
                            * (left_n_ref[0] * left_dn_deta[inode]
                                - left_n_ref[1] * left_dn_dxi[inode]);
                        let denominator = ((left_n_ref[0] * left_x_dot_dn_deta
                            - left_n_ref[1] * left_x_dot_dn_dxi)
                            .powf(2.0)
                            + (left_n_ref[0] * left_y_dot_dn_deta
                                - left_n_ref[1] * left_y_dot_dn_dxi)
                                .powf(2.0))
                        .sqrt();
                        numerator / denominator
                    };
                    let right_scaling_x_i = {
                        let numerator = (right_n_ref[0] * right_x_dot_dn_deta
                            - right_n_ref[1] * right_x_dot_dn_dxi)
                            * (right_n_ref[0] * right_dn_deta[inode]
                                - right_n_ref[1] * right_dn_dxi[inode]);
                        let denominator = ((right_n_ref[0] * right_x_dot_dn_deta
                            - right_n_ref[1] * right_x_dot_dn_dxi)
                            .powf(2.0)
                            + (right_n_ref[0] * right_y_dot_dn_deta
                                - right_n_ref[1] * right_y_dot_dn_dxi)
                                .powf(2.0))
                        .sqrt();
                        numerator / denominator
                    };
                    let right_scaling_y_i = {
                        let numerator = (right_n_ref[0] * right_y_dot_dn_deta
                            - right_n_ref[1] * right_y_dot_dn_dxi)
                            * (right_n_ref[0] * right_dn_deta[inode]
                                - right_n_ref[1] * right_dn_dxi[inode]);
                        let denominator = ((right_n_ref[0] * right_x_dot_dn_deta
                            - right_n_ref[1] * right_x_dot_dn_dxi)
                            .powf(2.0)
                            + (right_n_ref[0] * right_y_dot_dn_deta
                                - right_n_ref[1] * right_y_dot_dn_dxi)
                                .powf(2.0))
                        .sqrt();
                        numerator / denominator
                    };
                }
            }
        }
        (
            left_dr_du,
            left_dr_dx,
            left_dr_dy,
            right_dr_du,
            right_dr_dx,
            right_dr_dy,
        )
    }
}
