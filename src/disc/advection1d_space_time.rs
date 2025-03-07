pub mod boundary_condition;
mod flux;
mod riemann_solver;
use flux::space_time_flux1d;
use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2,
    ArrayViewMut3, Axis, s,
};
use riemann_solver::smoothed_upwind;

use crate::solver::SolverParameters;

use super::{
    basis::lagrange1d::LagrangeBasis1DLobatto,
    mesh::mesh2d::{Element2d, Mesh2d},
};

pub struct Disc1dAdvectionSpaceTime<'a> {
    pub basis: LagrangeBasis1DLobatto,
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
        let mut residuals: Array3<f64> = Array3::zeros((nelem, cell_ngp, 1));
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
                let transformed_f: Array1<f64> =
                    elem.jacob_det * f.dot(&elem.jacob_inv_t.slice(s![kgp, igp, .., ..]));
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
}
