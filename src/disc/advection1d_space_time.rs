pub mod boundary_condition;
mod flux;
mod riemann_solver;
mod shock_tracking;
use flux::space_time_flux1d;
use std::autodiff::autodiff;
use ndarray::{
    s, Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Order
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
}
impl<'a> Disc1dAdvectionSpaceTime<'a> {
    pub fn new(
        basis: LagrangeBasis1DLobatto,
        enriched_basis: LagrangeBasis1DLobatto,
        mesh: &'a Mesh2d,
        solver_param: &'a SolverParameters,
    ) -> Disc1dAdvectionSpaceTime<'a> {
        Disc1dAdvectionSpaceTime {
            current_iter: 0,
            basis,
            enriched_basis,
            mesh,
            solver_param,
            advection_speed: 0.1,
        }
    }
    pub fn solve(&mut self, mut solutions: ArrayViewMut3<f64>) {
        let nelem = self.mesh.elem_num;
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut residuals: Array3<f64> = Array3::zeros((nelem, cell_ngp * cell_ngp, 1));
        let mut enriched_residuals: Array3<f64> =
            Array3::zeros((nelem, (cell_ngp + 1) * (cell_ngp + 1), 1));

        // let mut assembled_linear_system: Array3<f64> =
        self.compute_residual_derivatives(solutions.view());
        // Array3::zeros((nelem, cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        // print solutions
        // println!("solutions: {:?}", solutions);
        // println!("basis_derivatives: {:?}", self.basis.dphis_cell_gps);
        self.compute_residuals(solutions.view(), residuals.view_mut());
        // print residuals
        println!("residuals: {:?}", residuals);

        self.compute_enriched_residuals(solutions.view(), enriched_residuals.view_mut());
        println!("enriched_residuals: {:?}", enriched_residuals);
        // print enriched residuals
        // println!("enriched_residuals: {:?}", enriched_residuals);
        /*
        while self.current_iter < self.solver_param.final_step {
            self.compute_residuals(solutions.view(), residuals.view_mut());
            self.compute_enriched_residuals(solutions.view(), enriched_residuals.view_mut());
            self.current_iter += 1;
        }
        */
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
        interp_matrix.dot(&sol)
    }
    fn compute_residual_derivatives(&mut self, solutions: ArrayView3<f64>) {
        let nelem = self.mesh.elem_num;
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut dres_du: Array4<f64> = Array4::zeros((nelem, cell_ngp * cell_ngp, cell_ngp * cell_ngp, 1));
        let mut dres_dx: Array4<f64> = Array4::zeros((nelem, cell_ngp * cell_ngp, 4, 1));
        let mut dres_dy: Array4<f64> = Array4::zeros((nelem, cell_ngp * cell_ngp, 4, 1));
        let mut denrr_du: Array4<f64> = Array4::zeros((nelem, (cell_ngp + 1) * (cell_ngp + 1), cell_ngp * cell_ngp, 1));
        let mut denrr_dx: Array4<f64> = Array4::zeros((nelem, (cell_ngp + 1) * (cell_ngp + 1), 4, 1));
        let mut denrr_dy: Array4<f64> = Array4::zeros((nelem, (cell_ngp + 1) * (cell_ngp + 1), 4, 1));
        for ielem in 0..nelem {
            let mut dres_du_slice: ArrayViewMut3<f64> = dres_du.slice_mut(s![ielem, .., .., ..]);
            let mut dres_dx_slice: ArrayViewMut3<f64> = dres_dx.slice_mut(s![ielem, .., .., ..]);
            let mut dres_dy_slice: ArrayViewMut3<f64> = dres_dy.slice_mut(s![ielem, .., .., ..]);
            let (dres_du_to_be_added, dres_dx_to_be_added, dres_dy_to_be_added) = self.volume_residual_derivatives(solutions.slice(s![ielem, .., ..]), ielem);
            dres_du_slice.scaled_add(1.0, &dres_du_to_be_added);
            dres_dx_slice.scaled_add(1.0, &dres_dx_to_be_added);
            dres_dy_slice.scaled_add(1.0, &dres_dy_to_be_added);
        }
        //println!("dres_dx[(0, .., 2, 0)]: {:?}", dres_dx.slice(s![0, .., 2, 0]));
        for &iedge in self.mesh.internal_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let ilelem = edge.parent_elements[0];
            let irelem = edge.parent_elements[1];
            let (mut left_dres_du_slice, mut right_dres_du_slice) = dres_du.multi_slice_mut((s![ilelem, .., .., ..], s![irelem, .., .., ..]));
            let (mut left_dres_dx_slice, mut right_dres_dx_slice) = dres_dx.multi_slice_mut((s![ilelem, .., .., ..], s![irelem, .., .., ..]));
            let (mut left_dres_dy_slice, mut right_dres_dy_slice) = dres_dy.multi_slice_mut((s![ilelem, .., .., ..], s![irelem, .., .., ..]));
            let (
                left_dres_du_tmp, 
                left_dres_dx_tmp, 
                left_dres_dy_tmp, 
                right_dres_du_tmp, 
                right_dres_dx_tmp, 
                right_dres_dy_tmp
            ) = self.surface_residual_derivatives(solutions, iedge);
            left_dres_du_slice.scaled_add(1.0, &left_dres_du_tmp);
            right_dres_du_slice.scaled_add(1.0, &right_dres_du_tmp);
            left_dres_dx_slice.scaled_add(1.0, &left_dres_dx_tmp);
            right_dres_dx_slice.scaled_add(1.0, &right_dres_dx_tmp);
            left_dres_dy_slice.scaled_add(1.0, &left_dres_dy_tmp);
            right_dres_dy_slice.scaled_add(1.0, &right_dres_dy_tmp);
        }
        //println!("dres_dx[(0, .., 2, 0)]: {:?}", dres_dx.slice(s![0, .., 2, 0]));
        // boundary
        let (dres_du_tmp, dres_dx_tmp, dres_dy_tmp) = self.boundary_residual_derivatives(solutions);
        dres_du.scaled_add(1.0, &dres_du_tmp);
        dres_dx.scaled_add(1.0, &dres_dx_tmp);
        dres_dy.scaled_add(1.0, &dres_dy_tmp);
        //println!("dres_dx[(0, .., 2, 0)]: {:?}", dres_dx.slice(s![0, .., 2, 0]));
    }
    fn compute_residuals(&mut self, solutions: ArrayView3<f64>, mut residuals: ArrayViewMut3<f64>) {
        let nelem = self.mesh.elem_num;
        let ngp = self.solver_param.cell_gp_num;
        for ielem in 0..nelem {
            let sol_slice= solutions.slice(s![ielem, .., ..]).into_shape_with_order(((ngp * ngp), Order::RowMajor)).unwrap();
            let mut res_slice = residuals.slice_mut(s![ielem, .., ..]).into_shape_with_order(((ngp * ngp), Order::RowMajor)).unwrap();
            let inodes = &self.mesh.elements[ielem].inodes;
            let mut x_slice: Array1<f64> = Array::zeros(4);
            let mut y_slice: Array1<f64> = Array::zeros(4);
            for i in 0..4{
                x_slice[i] = self.mesh.nodes[inodes[i]].x;
                y_slice[i] = self.mesh.nodes[inodes[i]].y;
            }
            let mut dsol = Array1::zeros(ngp * ngp);
            let mut dx = Array1::zeros(4);
            let mut dy = Array1::zeros(4);
            let mut dres = Array1::ones(ngp * ngp);
            self.dvolume_integral(&sol_slice.as_slice().unwrap(), &mut dsol.as_slice_mut().unwrap(), &x_slice.as_slice().unwrap(), &mut dx.as_slice_mut().unwrap(), &y_slice.as_slice().unwrap(), &mut dy.as_slice_mut().unwrap(), &mut res_slice.as_slice_mut().unwrap(), &mut dres.as_slice_mut().unwrap());
            /* 
            let mut residuals_slice: ArrayViewMut2<f64> = residuals.slice_mut(s![ielem, .., ..]);
            residuals_slice.scaled_add(1.0, &self.volume_integral(solutions, ielem));
            */
            // println!("volume_residuals_slice: {:?}", residuals_slice);
        }
        for &iedge in self.mesh.internal_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let ilelem = edge.parent_elements[0];
            let irelem = edge.parent_elements[1];
            let (mut left_res_slice, mut right_res_slice) =
                residuals.multi_slice_mut((s![ilelem, .., ..], s![irelem, .., ..]));
            let (left_res, right_res) = self.surface_integral(solutions, iedge);
            left_res_slice.scaled_add(1.0, &left_res);
            right_res_slice.scaled_add(1.0, &right_res);
        }
        self.boundary_condition(residuals, solutions);
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
            let mut enriched_residuals_slice: ArrayViewMut2<f64> =
                enriched_residuals.slice_mut(s![ielem, .., ..]);
            enriched_residuals_slice.scaled_add(
                1.0,
                &self.enriched_volume_integral(enriched_sol.view(), ielem),
            );
        }
        for &iedge in self.mesh.internal_edges.iter() {
            let edge = &self.mesh.edges[iedge];
            let ilelem = edge.parent_elements[0];
            let irelem = edge.parent_elements[1];
            let (mut left_enriched_res_slice, mut right_enriched_res_slice) =
                enriched_residuals.multi_slice_mut((s![ilelem, .., ..], s![irelem, .., ..]));
            let (left_enriched_res, right_enriched_res) =
                self.enriched_surface_integral(enriched_sol.view(), iedge);
            left_enriched_res_slice.scaled_add(1.0, &left_enriched_res);
            right_enriched_res_slice.scaled_add(1.0, &right_enriched_res);
        }
        self.enriched_boundary_condition(enriched_residuals, enriched_sol.view());
    }
    #[autodiff(dvolume_integral, Reverse, Const, Duplicated, Duplicated, Duplicated, Duplicated)]
    fn volume_integral(&self, sol: &[f64], x: &[f64], y: &[f64], res: &mut [f64]) {
        let basis = &self.basis;
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let f: Array1<f64> =
                    space_time_flux1d(sol[igp + kgp * cell_ngp], self.advection_speed);
                /*
                if kgp == 1 && igp == 1 {
                    println!("f: {:?}", f);
                }
                */
                let (jacob_det, jacob_inv_t) = self.mesh.evaluate_jacob(&self.basis, &x, &y);
                
                let transformed_f: Array1<f64> = jacob_det[[kgp, igp]]
                    * f.dot(&jacob_inv_t.slice(s![kgp, igp, .., ..]));
                // println!("transformed_f: {:?}", transformed_f);
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    /* 
                    println!("itest_func: {:?}", itest_func);
                    println!("itest_func_x: {:?}", itest_func_x);
                    println!("itest_func_t: {:?}", itest_func_t);
                    */
                    let test_func_xi = self.basis.phis_cell_gps[[igp, itest_func_x]];
                    let test_func_eta = self.basis.phis_cell_gps[[kgp, itest_func_t]];
                    let dtest_func_dxi = self.basis.dphis_cell_gps[[igp, itest_func_x]];
                    let dtest_func_deta = self.basis.dphis_cell_gps[[kgp, itest_func_t]];
                    
                    /*
                    println!(
                        "dtest_func_dxi * test_func_eta: {:?}",
                        dtest_func_dxi * test_func_eta
                    );
                    println!(
                        "dtest_func_deta * test_func_xi: {:?}",
                        dtest_func_deta * test_func_xi
                    );
                    */
                    res[itest_func] -= weights[igp]
                        * weights[kgp]
                        * (transformed_f[0] * dtest_func_dxi * test_func_eta
                            + transformed_f[1] * dtest_func_deta * test_func_xi);
                }
            }
        }
    }
    fn enriched_volume_integral(
        &mut self,
        enriched_sol: ArrayView3<f64>,
        ielem: usize,
    ) -> Array2<f64> {
        let elem = &self.mesh.elements[ielem];
        let cell_ngp = self.solver_param.cell_gp_num;
        let enriched_ngp = cell_ngp + 1;
        let weights = &self.enriched_basis.cell_gauss_weights;
        let mut res = Array2::zeros((enriched_ngp * enriched_ngp, 1));
        for kgp in 0..enriched_ngp {
            for igp in 0..enriched_ngp {
                let f: Array1<f64> = space_time_flux1d(
                    enriched_sol[[ielem, igp + kgp * enriched_ngp, 0]],
                    self.advection_speed,
                );
                let transformed_f: Array1<f64> = elem.enriched_jacob_det[[kgp, igp]]
                    * f.dot(&elem.enriched_jacob_inv_t.slice(s![kgp, igp, .., ..]));
                for itest_func in 0..enriched_ngp * enriched_ngp {
                    let itest_func_x = itest_func % enriched_ngp; // spatial index
                    let itest_func_t = itest_func / enriched_ngp; // temporal index
                    let test_func_xi = self.enriched_basis.phis_cell_gps[[igp, itest_func_x]];
                    let test_func_eta = self.enriched_basis.phis_cell_gps[[kgp, itest_func_t]];
                    let dtest_func_dxi = self.enriched_basis.dphis_cell_gps[[igp, itest_func_x]];
                    let dtest_func_deta = self.enriched_basis.dphis_cell_gps[[kgp, itest_func_t]];
                    res[[itest_func, 0]] -= weights[igp]
                        * weights[kgp]
                        * (transformed_f[0] * dtest_func_dxi * test_func_eta
                            + transformed_f[1] * dtest_func_deta * test_func_xi);
                }
            }
        }
        res
    }
    fn surface_integral(&self, sol: ArrayView3<f64>, iedge: usize) -> (Array2<f64>, Array2<f64>) {
        let edge = &self.mesh.edges[iedge];
        let ilelem = edge.parent_elements[0];
        let irelem = edge.parent_elements[1];
        let left_elem = &self.mesh.elements[ilelem];
        let right_elem = &self.mesh.elements[irelem];
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        let left_sol = sol.slice(s![ilelem, (cell_ngp - 1)..; cell_ngp, ..]);
        let right_sol = sol.slice(s![irelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]);
        let normal = edge.normal;
        let mut left_res = Array2::zeros((cell_ngp * cell_ngp, 1));
        let mut right_res = Array2::zeros((cell_ngp * cell_ngp, 1));
        for itest_func in 0..cell_ngp * cell_ngp {
            let itest_func_x = itest_func % cell_ngp; // spatial index
            let itest_func_t = itest_func / cell_ngp; // temporal index
            for kgp in 0..cell_ngp {
                // Map quadrature points based on edge orientation
                let left_kgp = kgp;
                let right_kgp = kgp;
                let left_value: ArrayView1<f64> = left_sol.slice(s![kgp, ..]);
                let right_value: ArrayView1<f64> = right_sol.slice(s![kgp, ..]);
                let num_flux =
                    smoothed_upwind(left_value[0], right_value[0], normal, self.advection_speed);
                let left_jacob_det = left_elem.jacob_det[[kgp, cell_ngp - 1]];
                let left_jacob_inv_t: ArrayView2<f64> =
                    left_elem.jacob_inv_t.slice(s![kgp, cell_ngp - 1, .., ..]);
                let left_n_ref = [1.0, 0.0];
                let left_n_ref_array = Array1::from_vec(left_n_ref.to_vec());
                let left_transformed_normal: Array1<f64> = left_jacob_inv_t.dot(&left_n_ref_array);
                let left_normal_magnitude = (left_transformed_normal[0].powi(2)
                    + left_transformed_normal[1].powi(2))
                .sqrt();
                let left_scaling = left_jacob_det * left_normal_magnitude;
                let left_transformed_flux = left_scaling * num_flux;

                let right_jacob_det = right_elem.jacob_det[[kgp, 0]];
                let right_jacob_inv_t: ArrayView2<f64> =
                    right_elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
                let right_n_ref = [-1.0, 0.0];
                let right_n_ref_array = Array1::from_vec(right_n_ref.to_vec());
                let right_transformed_normal: Array1<f64> =
                    right_jacob_inv_t.dot(&right_n_ref_array);
                let right_normal_magnitude = (right_transformed_normal[0].powi(2)
                    + right_transformed_normal[1].powi(2))
                .sqrt();
                let right_scaling = right_jacob_det * right_normal_magnitude;
                let right_transformed_flux = right_scaling * (-num_flux);
                let left_phi = self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]]
                    * self.basis.phis_cell_gps[[left_kgp, itest_func_t]];
                let right_phi = self.basis.phis_cell_gps[[0, itest_func_x]]
                    * self.basis.phis_cell_gps[[right_kgp, itest_func_t]];
                // println!("left_transformed_flux: {:?}", left_transformed_flux);
                // println!("right_transformed_flux: {:?}", right_transformed_flux);
                left_res[[itest_func, 0]] += weights[left_kgp] * left_transformed_flux * left_phi;
                right_res[[itest_func, 0]] +=
                    weights[right_kgp] * right_transformed_flux * right_phi;
            }
        }
        (left_res, right_res) 
    }
    fn enriched_surface_integral(
        &self,
        enriched_sol: ArrayView3<f64>,
        iedge: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let edge = &self.mesh.edges[iedge];
        let ilelem = edge.parent_elements[0];
        let irelem = edge.parent_elements[1];
        let left_elem = &self.mesh.elements[ilelem];
        let right_elem = &self.mesh.elements[irelem];
        let cell_ngp = self.solver_param.cell_gp_num;
        let enriched_ngp = cell_ngp + 1;
        let weights = &self.enriched_basis.cell_gauss_weights;
        let left_sol = enriched_sol.slice(s![ilelem, (cell_ngp - 1)..; cell_ngp, ..]);
        let right_sol = enriched_sol.slice(s![irelem, 0..=(-(cell_ngp as isize)); cell_ngp, ..]);
        let normal = edge.normal;
        let mut left_res = Array2::zeros((enriched_ngp * enriched_ngp, 1));
        let mut right_res = Array2::zeros((enriched_ngp * enriched_ngp, 1));
        for itest_func in 0..enriched_ngp {
            for kgp in 0..enriched_ngp {
                // Map quadrature points based on edge orientation
                let left_kgp = kgp;
                let right_kgp = kgp;
                let left_value: ArrayView1<f64> = left_sol.slice(s![kgp, ..]);
                let right_value: ArrayView1<f64> = right_sol.slice(s![kgp, ..]);
                let num_flux =
                    smoothed_upwind(left_value[0], right_value[0], normal, self.advection_speed);

                let left_jacob_det = left_elem.enriched_jacob_det[[kgp, cell_ngp - 1]];
                let left_jacob_inv_t: ArrayView2<f64> =
                    left_elem.enriched_jacob_inv_t.slice(s![kgp, 0, .., ..]);
                let left_n_ref = [1.0, 0.0];
                let left_n_ref_array = Array1::from_vec(left_n_ref.to_vec());
                let left_transformed_normal: Array1<f64> = left_jacob_inv_t.dot(&left_n_ref_array);
                let left_normal_magnitude = (left_transformed_normal[0].powi(2)
                    + left_transformed_normal[1].powi(2))
                .sqrt();
                let left_scaling = left_jacob_det * left_normal_magnitude;
                let left_transformed_flux = left_scaling * num_flux;

                let right_jacob_det = right_elem.enriched_jacob_det[[kgp, 0]];
                let right_jacob_inv_t: ArrayView2<f64> =
                    right_elem.enriched_jacob_inv_t.slice(s![kgp, 0, .., ..]);
                let right_n_ref = [-1.0, 0.0];
                let right_n_ref_array = Array1::from_vec(right_n_ref.to_vec());
                let right_transformed_normal: Array1<f64> =
                    right_jacob_inv_t.dot(&right_n_ref_array);
                let right_normal_magnitude = (right_transformed_normal[0].powi(2)
                    + right_transformed_normal[1].powi(2))
                .sqrt();
                let right_scaling = right_jacob_det * right_normal_magnitude;
                let right_transformed_flux = right_scaling * (-num_flux);
                let left_phi = self.enriched_basis.phis_cell_gps[[enriched_ngp - 1, itest_func]]
                    * self.enriched_basis.phis_cell_gps[[left_kgp, itest_func]];
                let right_phi = self.enriched_basis.phis_cell_gps[[0, itest_func]]
                    * self.enriched_basis.phis_cell_gps[[right_kgp, itest_func]];
                left_res[[itest_func, 0]] += weights[left_kgp] * left_transformed_flux * left_phi;
                right_res[[itest_func, 0]] +=
                    weights[right_kgp] * right_transformed_flux * right_phi;
            }
        }
        (left_res, right_res)
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
            x[i] = self.mesh.nodes[inodes[i]].x;
            y[i] = self.mesh.nodes[inodes[i]].y;
        }
        let mut dr_du: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp, 1)); // (ntest_func, ntrial_func, neq)
        let mut dr_dx: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, 4, 1)); // (ntest_func, ntrial_func, neq)
        let mut dr_dy: Array3<f64> = Array3::zeros((cell_ngp * cell_ngp, 4, 1)); // (ntest_func, ntrial_func, neq)
        // derivatives of volume integral
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let u = sol[[igp + kgp * cell_ngp, 0]];
                let xi = self.basis.cell_gauss_points[igp];
                let eta = self.basis.cell_gauss_points[kgp];
                let mut dn_dxi: Array1<f64> = Array1::zeros(4);
                let mut dn_deta: Array1<f64> = Array1::zeros(4);
                dn_dxi[0] = -0.25 * (1.0 - eta);
                dn_deta[0] = -0.25 * (1.0 - xi);
                dn_dxi[1] = 0.25 * (1.0 - eta);
                dn_deta[1] = -0.25 * (1.0 + xi);
                dn_dxi[2] = 0.25 * (1.0 + eta);
                dn_deta[2] = 0.25 * (1.0 + xi);
                dn_dxi[3] = -0.25 * (1.0 + eta);
                dn_deta[3] = 0.25 * (1.0 - xi);
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
                        dr_du[[itest_func, itrial_func, 0]] -= (weights[igp] * weights[kgp])
                            * (trial_func_xi * trial_func_eta)
                            * ((test_func_xi * dtest_func_deta)
                                * ((-a * y_dot_dn_dxi) + x_dot_dn_dxi)
                                - (test_func_eta * dtest_func_dxi)
                                    * ((-a * x_dot_dn_deta) + y_dot_dn_deta));
                    }

                    for inode in 0..4 {
                        dr_dx[[itest_func, inode, 0]] -= (weights[igp] * weights[kgp])
                            * u
                            * (test_func_xi * dtest_func_deta * dn_dxi[inode]
                                - test_func_eta * dtest_func_dxi * dn_deta[inode]);
                        dr_dy[[itest_func, inode, 0]] -= (weights[igp] * weights[kgp])
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
            x[i] = self.mesh.nodes[inodes[i]].x;
            y[i] = self.mesh.nodes[inodes[i]].y;
        }
        let mut denrr_du: Array3<f64> =
            Array3::zeros((enriched_ngp * enriched_ngp, cell_ngp * cell_ngp, 1));
        let mut denrr_dx: Array3<f64> = Array3::zeros((enriched_ngp * enriched_ngp, 4, 1));
        let mut denrr_dy: Array3<f64> = Array3::zeros((enriched_ngp * enriched_ngp, 4, 1));
        // derivatives of volume integral
        for kgp in 0..enriched_ngp {
            for igp in 0..enriched_ngp {
                let u = enriched_sol[[igp + kgp * enriched_ngp, 0]];
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
                for itest_func in 0..enriched_ngp * enriched_ngp {
                    let itest_func_x = itest_func % enriched_ngp; // spatial index
                    let itest_func_t = itest_func / enriched_ngp; // temporal index
                    let test_func_xi = self.enriched_basis.phis_cell_gps[[igp, itest_func_x]];
                    let test_func_eta = self.enriched_basis.phis_cell_gps[[kgp, itest_func_t]];
                    let dtest_func_dxi = self.enriched_basis.dphis_cell_gps[[igp, itest_func_x]];
                    let dtest_func_deta = self.enriched_basis.dphis_cell_gps[[kgp, itest_func_t]];
                    for itrial_func in 0..(cell_ngp * cell_ngp) {
                        let itrial_func_x = itrial_func % cell_ngp; // spatial index
                        let itrial_func_t = itrial_func / cell_ngp; // temporal index
                        let trial_func_xi = self.basis.evaluate_basis_at(itrial_func_x, xi);
                        let trial_func_eta = self.basis.evaluate_basis_at(itrial_func_t, eta);
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
        let left_elem = &self.mesh.elements[ilelem];
        let right_elem = &self.mesh.elements[irelem];
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
            left_x[i] = self.mesh.nodes[left_elem.inodes[i]].x;
            left_y[i] = self.mesh.nodes[left_elem.inodes[i]].y;
            right_x[i] = self.mesh.nodes[right_elem.inodes[i]].x;
            right_y[i] = self.mesh.nodes[right_elem.inodes[i]].y;
        }
        for kgp in 0..cell_ngp {
            let left_kgp = kgp;
            let right_kgp = kgp;
            let left_value = left_sol.slice(s![kgp, ..]);
            let right_value = right_sol.slice(s![kgp, ..]);
            let numerical_flux =
                smoothed_upwind(left_value[0], right_value[0], normal, self.advection_speed);
            let left_jacob_det = left_elem.jacob_det[[kgp, cell_ngp - 1]];
            let right_jacob_det = right_elem.jacob_det[[kgp, 0]];
            let left_jacob_inv_t: ArrayView2<f64> =
                left_elem.jacob_inv_t.slice(s![kgp, cell_ngp - 1, .., ..]);
            let right_jacob_inv_t: ArrayView2<f64> =
                right_elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
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

                let dflux_dul = riemann_solver::dflux_dul(normal, self.advection_speed);
                let dflux_dur = riemann_solver::dflux_dur(normal, self.advection_speed);
                let (dflux_dnx, dflux_dny) = riemann_solver::dflux_dnormal(
                    left_value[0],
                    right_value[0],
                    normal,
                    self.advection_speed,
                );

                let mut left_dflux_dx = Array1::zeros(4);
                let mut left_dflux_dy = Array1::zeros(4);
                let mut right_dflux_dx = Array1::zeros(4);
                let mut right_dflux_dy = Array1::zeros(4);
                left_dflux_dx[1] = dflux_dny;
                left_dflux_dy[1] = -dflux_dnx;
                left_dflux_dx[2] = -dflux_dny;
                left_dflux_dy[2] = dflux_dnx;
                right_dflux_dx[0] = -dflux_dny;
                right_dflux_dy[0] = dflux_dnx;
                right_dflux_dx[3] = dflux_dny;
                right_dflux_dy[3] = -dflux_dnx;

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

                left_dr_du[[itest_func, cell_ngp - 1 + kgp * cell_ngp, 0]] +=
                    weights[left_kgp] * left_scaling * dflux_dul * left_phi;
                right_dr_du[[itest_func, kgp * cell_ngp, 0]] +=
                    weights[right_kgp] * right_scaling * (-dflux_dur) * right_phi;

                for inode in 0..4 {
                    let left_dscaling_dx_i = {
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
                    let left_dscaling_dy_i = {
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
                    let right_dscaling_dx_i = {
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
                    let right_dscaling_dy_i = {
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

                    left_dr_dx[[itest_func, inode, 0]] += weights[left_kgp]
                        * left_phi
                        * (left_dscaling_dx_i * numerical_flux
                            + left_dscaling_dy_i * left_dflux_dx[inode]);
                    left_dr_dy[[itest_func, inode, 0]] += weights[left_kgp]
                        * left_phi
                        * (left_dscaling_dy_i * numerical_flux
                            + left_dscaling_dx_i * left_dflux_dy[inode]);
                    right_dr_dx[[itest_func, inode, 0]] += weights[right_kgp]
                        * right_phi
                        * (right_dscaling_dx_i * (-numerical_flux)
                            + right_dscaling_dy_i * right_dflux_dx[inode]);
                    right_dr_dy[[itest_func, inode, 0]] += weights[right_kgp]
                        * right_phi
                        * (right_dscaling_dy_i * (-numerical_flux)
                            + right_dscaling_dx_i * right_dflux_dy[inode]);
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
    pub fn surface_enriched_residual_derivatives(
        &self,
        enriched_sol: ArrayView3<f64>,
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
        let left_elem = &self.mesh.elements[ilelem];
        let right_elem = &self.mesh.elements[irelem];
        let cell_ngp = self.solver_param.cell_gp_num;
        let enriched_ngp = cell_ngp + 1;
        let weights = &self.basis.cell_gauss_weights; // Use standard quadrature points/weights
        let left_sol = enriched_sol.slice(s![ilelem, (enriched_ngp - 1)..; enriched_ngp, ..]); // Standard solution on edge
        let right_sol =
            enriched_sol.slice(s![irelem, 0..=(-(enriched_ngp as isize)); enriched_ngp, ..]); // Standard solution on edge (Assuming simple case)
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
            left_x[i] = self.mesh.nodes[left_elem.inodes[i]].x;
            left_y[i] = self.mesh.nodes[left_elem.inodes[i]].y;
            right_x[i] = self.mesh.nodes[right_elem.inodes[i]].x;
            right_y[i] = self.mesh.nodes[right_elem.inodes[i]].y;
        }
        for kgp in 0..enriched_ngp {
            let left_kgp = kgp;
            let right_kgp = kgp;
            let left_value = left_sol.slice(s![kgp, ..]);
            let right_value = right_sol.slice(s![kgp, ..]);
            let numerical_flux =
                smoothed_upwind(left_value[0], right_value[0], normal, self.advection_speed);
            let left_jacob_det = left_elem.jacob_det[[kgp, cell_ngp - 1]];
            let right_jacob_det = right_elem.jacob_det[[kgp, 0]];
            let left_jacob_inv_t: ArrayView2<f64> =
                left_elem.jacob_inv_t.slice(s![kgp, cell_ngp - 1, .., ..]);
            let right_jacob_inv_t: ArrayView2<f64> =
                right_elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
            let left_xi = self.enriched_basis.cell_gauss_points[cell_ngp - 1];
            let right_xi = self.enriched_basis.cell_gauss_points[0];
            let left_eta = self.enriched_basis.cell_gauss_points[kgp];
            let right_eta = self.enriched_basis.cell_gauss_points[kgp];
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

            for itest_func in 0..(enriched_ngp * enriched_ngp) {
                let itest_func_x = itest_func % enriched_ngp; // spatial index
                let itest_func_t = itest_func / enriched_ngp; // temporal index
                let left_phi = self.enriched_basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]]
                    * self.enriched_basis.phis_cell_gps[[left_kgp, itest_func_t]];
                let right_phi = self.enriched_basis.phis_cell_gps[[0, itest_func_x]]
                    * self.enriched_basis.phis_cell_gps[[right_kgp, itest_func_t]];

                let dflux_dul = riemann_solver::dflux_dul(normal, self.advection_speed);
                let dflux_dur = riemann_solver::dflux_dur(normal, self.advection_speed);
                let (dflux_dnx, dflux_dny) = riemann_solver::dflux_dnormal(
                    left_value[0],
                    right_value[0],
                    normal,
                    self.advection_speed,
                );

                let mut left_dflux_dx = Array1::zeros(4);
                let mut left_dflux_dy = Array1::zeros(4);
                let mut right_dflux_dx = Array1::zeros(4);
                let mut right_dflux_dy = Array1::zeros(4);
                left_dflux_dx[1] = dflux_dny;
                left_dflux_dy[1] = -dflux_dnx;
                left_dflux_dx[2] = -dflux_dny;
                left_dflux_dy[2] = dflux_dnx;
                right_dflux_dx[0] = -dflux_dny;
                right_dflux_dy[0] = dflux_dnx;
                right_dflux_dx[3] = dflux_dny;
                right_dflux_dy[3] = -dflux_dnx;

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

                for itrial_func in 0..cell_ngp {
                    let left_trial_func_eta = self.basis.evaluate_basis_at(itrial_func, left_eta);
                    let right_trial_func_eta = self.basis.evaluate_basis_at(itrial_func, right_eta);
                    left_dr_du[[itest_func, cell_ngp - 1 + itrial_func * cell_ngp, 0]] += weights
                        [left_kgp]
                        * left_scaling
                        * dflux_dul
                        * left_trial_func_eta
                        * left_phi;
                    right_dr_du[[itest_func, itrial_func * cell_ngp, 0]] += weights[right_kgp]
                        * right_scaling
                        * (-dflux_dur)
                        * right_trial_func_eta
                        * right_phi;
                }
                for inode in 0..4 {
                    let left_dscaling_dx_i = {
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
                    let left_dscaling_dy_i = {
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
                    let right_dscaling_dx_i = {
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
                    let right_dscaling_dy_i = {
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

                    left_dr_dx[[itest_func, inode, 0]] += weights[left_kgp]
                        * left_phi
                        * (left_dscaling_dx_i * numerical_flux
                            + left_dscaling_dy_i * left_dflux_dx[inode]);
                    left_dr_dy[[itest_func, inode, 0]] += weights[left_kgp]
                        * left_phi
                        * (left_dscaling_dy_i * numerical_flux
                            + left_dscaling_dx_i * left_dflux_dy[inode]);
                    right_dr_dx[[itest_func, inode, 0]] += weights[right_kgp]
                        * right_phi
                        * (right_dscaling_dx_i * (-numerical_flux)
                            + right_dscaling_dy_i * right_dflux_dx[inode]);
                    right_dr_dy[[itest_func, inode, 0]] += weights[right_kgp]
                        * right_phi
                        * (right_dscaling_dy_i * (-numerical_flux)
                            + right_dscaling_dx_i * right_dflux_dy[inode]);
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
    fn boundary_condition(&self, mut res: ArrayViewMut3<f64>, sol: ArrayView3<f64>) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        for iedge in 0..6 {
            let edge = &self.mesh.edges[iedge];
            let ielem = edge.parent_elements[0];
            let elem = &self.mesh.elements[ielem];
            // lower boundary
            if iedge == 0 || iedge == 1 {
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    for kgp in 0..cell_ngp {
                        //println!("iedge: {}, kgp: {}", iedge, kgp);
                        let u = { if iedge == 0 { 2.0 } else { 0.0 } };
                        let phi = self.basis.phis_cell_gps[[kgp, itest_func_x]]
                            * self.basis.phis_cell_gps[[0, itest_func_t]];
                        let jacob_det = elem.jacob_det[[0, kgp]];
                        let jacob_inv_t: ArrayView2<f64> =
                            elem.jacob_inv_t.slice(s![0, kgp, .., ..]);
                        let n_ref = [0.0, -1.0];
                        let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                        let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                        let normal_magnitude =
                            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                        let scaling = jacob_det * normal_magnitude;
                        let flux = -u;
                        let transformed_flux = scaling * flux;
                        res[[ielem, itest_func, 0]] += weights[kgp] * phi * transformed_flux;
                    }
                }
            }
            // upper boundary
            if iedge == 3 || iedge == 4 {
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    for kgp in 0..cell_ngp {
                        let u = sol[[ielem, cell_ngp * cell_ngp - 1 - kgp, 0]];
                        let phi = self.basis.phis_cell_gps[[cell_ngp - 1 - kgp, itest_func_x]]
                            * self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_t]];
                        let jacob_det = elem.jacob_det[[cell_ngp - 1, cell_ngp - 1 - kgp]];
                        let jacob_inv_t: ArrayView2<f64> =
                            elem.jacob_inv_t
                                .slice(s![cell_ngp - 1, cell_ngp - 1 - kgp, .., ..]);
                        let n_ref = [0.0, 1.0];
                        let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                        let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                        let normal_magnitude =
                            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                        let scaling = jacob_det * normal_magnitude;
                        let flux = u;
                        let transformed_flux = scaling * flux;
                        res[[ielem, itest_func, 0]] += weights[cell_ngp - 1 - kgp] * phi * transformed_flux;
                    }
                }
            }
            // right boundary
            if iedge == 2 {
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    for kgp in 0..cell_ngp {
                        let u = sol[[ielem, cell_ngp - 1 + kgp * cell_ngp, 0]];
                        let phi = self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]]
                            * self.basis.phis_cell_gps[[kgp, itest_func_t]];
                        let jacob_det = elem.jacob_det[[kgp, cell_ngp - 1]];
                        let jacob_inv_t: ArrayView2<f64> =
                            elem.jacob_inv_t.slice(s![kgp, cell_ngp - 1, .., ..]);
                        let n_ref = [1.0, 0.0];
                        let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                        let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                        let normal_magnitude =
                            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                        let scaling = jacob_det * normal_magnitude;
                        let flux = self.advection_speed * u;
                        let transformed_flux = scaling * flux;
                        res[[ielem, itest_func, 0]] += weights[kgp] * phi * transformed_flux;
                    }
                }
            }
            // left boundary
            if iedge == 5 {
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    for kgp in 0..cell_ngp {
                        let u = 2.0;
                        let phi = self.basis.phis_cell_gps[[0, itest_func_x]]
                            * self.basis.phis_cell_gps[[kgp, itest_func_t]];
                        let jacob_det = elem.jacob_det[[kgp, 0]];
                        let jacob_inv_t: ArrayView2<f64> =
                            elem.jacob_inv_t.slice(s![kgp, 0, .., ..]);
                        let n_ref = [-1.0, 0.0];
                        let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                        let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                        let normal_magnitude =
                            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                        let scaling = jacob_det * normal_magnitude;
                        let flux = - self.advection_speed * u;
                        let transformed_flux = scaling * flux;
                        res[[ielem, itest_func, 0]] += weights[kgp] * phi * transformed_flux;
                    }
                }
            }
        }
    }
    fn enriched_boundary_condition(&self, mut res: ArrayViewMut3<f64>, sol: ArrayView3<f64>) {
        let cell_ngp = self.solver_param.cell_gp_num + 1;
        let weights = &self.enriched_basis.cell_gauss_weights;
        for iedge in 0..6 {
            let edge = &self.mesh.edges[iedge];
            let ielem = edge.parent_elements[0];
            let elem = &self.mesh.elements[ielem];
            // lower boundary
            if iedge == 0 || iedge == 1 {
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    for kgp in 0..cell_ngp {
                        let u = { if iedge == 0 { 2.0 } else { 0.0 } };
                        let phi = self.enriched_basis.phis_cell_gps[[kgp, itest_func_x]]
                            * self.enriched_basis.phis_cell_gps[[0, itest_func_t]];
                        let jacob_det = elem.enriched_jacob_det[[0, kgp]];
                        let jacob_inv_t: ArrayView2<f64> =
                            elem.enriched_jacob_inv_t.slice(s![0, kgp, .., ..]);
                        let n_ref = [0.0, -1.0];
                        let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                        let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                        let normal_magnitude =
                            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                        let scaling = jacob_det * normal_magnitude;
                        let flux = -u;
                        let transformed_flux = scaling * flux;
                        res[[ielem, itest_func, 0]] += weights[kgp] * phi * transformed_flux;
                    }
                }
            }
            // upper boundary
            if iedge == 3 || iedge == 4 {
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    for kgp in 0..cell_ngp {
                        let u = sol[[ielem, cell_ngp * cell_ngp - 1 - kgp, 0]];
                        let phi = self.enriched_basis.phis_cell_gps
                            [[cell_ngp - 1 - kgp, itest_func_x]]
                            * self.enriched_basis.phis_cell_gps[[cell_ngp - 1, itest_func_t]];
                        let jacob_det = elem.enriched_jacob_det[[cell_ngp - 1, cell_ngp - 1 - kgp]];
                        let jacob_inv_t: ArrayView2<f64> = elem.enriched_jacob_inv_t.slice(s![
                            cell_ngp - 1,
                            cell_ngp - 1 - kgp,
                            ..,
                            ..
                        ]);
                        let n_ref = [0.0, 1.0];
                        let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                        let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                        let normal_magnitude =
                            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                        let scaling = jacob_det * normal_magnitude;
                        let flux = u;
                        let transformed_flux = scaling * flux;
                        res[[ielem, itest_func, 0]] += weights[kgp] * phi * transformed_flux;
                    }
                }
            }
            // right boundary
            if iedge == 2 {
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    for kgp in 0..cell_ngp {
                        let u = sol[[ielem, cell_ngp - 1 + kgp * cell_ngp, 0]];
                        let phi = self.enriched_basis.phis_cell_gps[[cell_ngp - 1, itest_func_x]]
                            * self.enriched_basis.phis_cell_gps[[kgp, itest_func_t]];
                        let jacob_det = elem.enriched_jacob_det[[kgp, cell_ngp - 1]];
                        let jacob_inv_t: ArrayView2<f64> =
                            elem.enriched_jacob_inv_t
                                .slice(s![kgp, cell_ngp - 1, .., ..]);
                        let n_ref = [1.0, 0.0];
                        let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                        let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                        let normal_magnitude =
                            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                        let scaling = jacob_det * normal_magnitude;
                        let flux = self.advection_speed * u;
                        let transformed_flux = scaling * flux;
                        res[[ielem, itest_func, 0]] += weights[kgp] * phi * transformed_flux;
                    }
                }
            }
            // left boundary
            if iedge == 5 {
                for itest_func in 0..cell_ngp * cell_ngp {
                    let itest_func_x = itest_func % cell_ngp; // spatial index
                    let itest_func_t = itest_func / cell_ngp; // temporal index
                    for kgp in 0..cell_ngp {
                        let u = 2.0;
                        let phi = self.enriched_basis.phis_cell_gps[[0, itest_func_x]]
                            * self.enriched_basis.phis_cell_gps[[kgp, itest_func_t]];
                        let jacob_det = elem.enriched_jacob_det[[kgp, 0]];
                        let jacob_inv_t: ArrayView2<f64> =
                            elem.enriched_jacob_inv_t.slice(s![kgp, 0, .., ..]);
                        let n_ref = [-1.0, 0.0];
                        let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                        let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                        let normal_magnitude =
                            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                        let scaling = jacob_det * normal_magnitude;
                        let flux = - self.advection_speed * u;
                        let transformed_flux = scaling * flux;
                        res[[ielem, itest_func, 0]] += weights[kgp] * phi * transformed_flux;
                    }
                }
            }
        }
    }
    fn boundary_residual_derivatives(&mut self, solutions: ArrayView3<f64>) -> (Array4<f64>, Array4<f64>, Array4<f64>) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        let nelem = self.mesh.elem_num;
        let mut dres_du: Array4<f64> = Array4::zeros((nelem, cell_ngp * cell_ngp, cell_ngp * cell_ngp, 1));
        let mut dres_dx: Array4<f64> = Array4::zeros((nelem, cell_ngp * cell_ngp, 4, 1));
        let mut dres_dy: Array4<f64> = Array4::zeros((nelem, cell_ngp * cell_ngp, 4, 1));
        // edge 3 and edge 4
        for iedge in 3..=4 {
            let edge = &self.mesh.edges[iedge];
            let ielem = edge.parent_elements[0];
            let elem = &self.mesh.elements[ielem];
            // create sol filled with 0
            let normal = edge.normal;
            let n_ref = [0.0, 1.0];
            let mut dr_du = dres_du.slice_mut(s![ielem, .., .., ..]);
            let mut dr_dx = dres_dx.slice_mut(s![ielem, .., .., ..]);
            let mut dr_dy = dres_dy.slice_mut(s![ielem, .., .., ..]);
            let mut x = Array1::zeros(4);
            let mut y = Array1::zeros(4);
            for i in 0..4 {
                x[i] = self.mesh.nodes[elem.inodes[i]].x;
                y[i] = self.mesh.nodes[elem.inodes[i]].y;
            }
            for kgp in 0..cell_ngp {
                let u = solutions[(ielem, cell_ngp * cell_ngp - 1 - kgp, 0)];
                let flux = u;
                let jacob_det = elem.jacob_det[(cell_ngp - 1, cell_ngp - 1 - kgp)];
                let jacob_inv_t: ArrayView2<f64> = elem.jacob_inv_t.slice(s![cell_ngp - 1, cell_ngp - 1 - kgp, .., ..]);
                let xi = self.basis.cell_gauss_points[cell_ngp - 1 - kgp];
                let eta = self.basis.cell_gauss_points[cell_ngp - 1];
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
                    let phi = self.basis.phis_cell_gps[[cell_ngp - 1 - kgp, itest_func_x]]
                    * self.basis.phis_cell_gps[[cell_ngp - 1, itest_func_t]];
                    let dflux_du = riemann_solver::dflux_dul(normal, self.advection_speed);
                    let (dflux_dnx, dflux_dny) = riemann_solver::dflux_dnormal(
                        u,
                        0.0,
                        normal,
                        self.advection_speed,
                    );
                    let mut dflux_dx = Array1::zeros(4);
                    let mut dflux_dy = Array1::zeros(4);
                    dflux_dx[2] = dflux_dny;
                    dflux_dy[2] = -dflux_dnx;
                    dflux_dx[3] = -dflux_dny;
                    dflux_dy[3] = dflux_dnx;
                    let n_ref_array: Array1<f64> = Array1::from_vec(n_ref.to_vec());
                    let transformed_normal: Array1<f64> = jacob_inv_t.dot(&n_ref_array);
                    let normal_magnitude = (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
                    let scaling = jacob_det * normal_magnitude;
                    dr_du[(itest_func, cell_ngp * cell_ngp - 1 - kgp, 0)] += weights[cell_ngp - 1 -kgp] * scaling * dflux_du * phi;

                    for inode in 0..4 {
                        let dscaling_dx_i = {
                            let numerator = (n_ref[0] * x_dot_dn_deta - n_ref[1] * x_dot_dn_dxi)
                            * (n_ref[0] * dn_deta[inode] - n_ref[1] * dn_dxi[inode]);
                            let denominator = ((n_ref[0] * x_dot_dn_deta
                                - n_ref[1] * x_dot_dn_dxi)
                                .powf(2.0)
                                + (n_ref[0] * y_dot_dn_deta
                                    - n_ref[1] * y_dot_dn_dxi)
                                    .powf(2.0))
                            .sqrt();
                            numerator / denominator
                        };
                        let dscaling_dy_i = {
                            let numerator = (n_ref[0] * y_dot_dn_deta
                                - n_ref[1] * y_dot_dn_dxi)
                                * (n_ref[0] * dn_deta[inode]
                                    - n_ref[1] * dn_dxi[inode]);
                            let denominator = ((n_ref[0] * x_dot_dn_deta
                                - n_ref[1] * x_dot_dn_dxi)
                                .powf(2.0)
                                + (n_ref[0] * y_dot_dn_deta
                                    - n_ref[1] * y_dot_dn_dxi)
                                    .powf(2.0))
                            .sqrt();
                            numerator / denominator
                        };

                        dr_dx[(itest_func, inode, 0)] += weights[cell_ngp - 1 - kgp]
                        * phi
                        * (dscaling_dx_i * flux
                            + dscaling_dy_i * dflux_dx[inode]);

                        dr_dy[(itest_func, inode, 0)] += weights[cell_ngp - 1 - kgp]
                        * phi
                        * (dscaling_dy_i * flux
                            + dscaling_dx_i * dflux_dy[inode]);
                    }
                }
            }
        }
        (dres_du, dres_dx, dres_dy)
    }
    pub fn initialize_solution(&mut self, mut solutions: ArrayViewMut3<f64>) {
        let cell_ngp = self.solver_param.cell_gp_num;
        for igp in 0..cell_ngp * cell_ngp {
            solutions[[0, igp, 0]] = 2.0;
            solutions[[1, igp, 0]] = 0.0;
        }
    }
}
