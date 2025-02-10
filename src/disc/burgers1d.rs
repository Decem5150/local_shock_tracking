mod flux;
mod precompute_matrix;
mod riemann_solver;
mod cell_shock_detector;
mod shock_tracking;
use super::gauss_points::GaussPoints1d;
use super::mesh::{mesh1d::Mesh1d, BoundaryType};
use crate::disc::basis::lagrange1d::LagrangeBasis1D;
use crate::solver::{FlowParameters, MeshParameters, SolverParameters};
use flux::flux1d;
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayViewMut2};
use riemann_solver::rusanov::rusanov;

pub struct Disc1dBurgers<'a> {
    pub residuals: Array3<f64>, // (nelem, idof, neq)
    pub current_time: f64,
    pub current_step: usize,
    pub gauss_points: GaussPoints1d,
    pub basis: LagrangeBasis1D,
    pub mesh: Mesh1d,
    pub flow_param: &'a FlowParameters,
    pub mesh_param: &'a MeshParameters,
    pub solver_param: &'a SolverParameters,
    ss_m_mat: Array2<f64>,   // mass matrix of the two space polynomials
    ss_im_mat: Array2<f64>,  // inverse mass matrix of the two space polynomials
    sst_kxi_mat: Array2<f64>, // spatial stiffness matrix of the space polynomial and the space-time polynomial
    stst_m_mat: Array2<f64>,   // mass matrix of the two space-time polynomials
    stst_im_mat: Array2<f64>,  // inverse mass matrix of the two space-time polynomials
    stst_kxi_mat: Array2<f64>, // spatial stiffness matrix of the two space-time polynomials
    stst_ik1_mat: Array2<f64>, // inverse temporal stiffness matrix of the two space-time polynomials
    stst_f0_mat: Array2<f64>,  // mass matrix at relative time 0 for the two space-time polynomials
}
impl<'a> Disc1dBurgers<'a> {
    fn new(
        gauss_points: GaussPoints1d,
        basis: LagrangeBasis1D,
        mesh: Mesh1d,
        flow_param: &'a FlowParameters,
        mesh_param: &'a MeshParameters,
        solver_param: &'a SolverParameters,
    ) -> Disc1dBurgers<'a> {
        let nelem = mesh_param.elem_num;
        let cell_ngp = solver_param.cell_gp_num;
        let neq = solver_param.equation_num;
        let residuals: Array3<f64> = Array3::zeros((nelem, cell_ngp, neq));
        let ss_m_mat: Array2<f64> = Array2::zeros((cell_ngp, cell_ngp));
        let ss_im_mat: Array2<f64> = Array2::zeros((cell_ngp, cell_ngp));
        let sst_kxi_mat: Array2<f64> = Array2::zeros((cell_ngp, cell_ngp * cell_ngp));
        let stst_m_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let stst_im_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let stst_kxi_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let stst_ik1_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let stst_f0_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let mut disc = Disc1dBurgers {
            residuals,
            current_time: 0.0,
            current_step: 0,
            gauss_points,
            basis,
            mesh,
            flow_param,
            mesh_param,
            solver_param,
            ss_m_mat,
            ss_im_mat,
            sst_kxi_mat,
            stst_m_mat,
            stst_im_mat,
            stst_kxi_mat,
            stst_ik1_mat,
            stst_f0_mat,
        };
        disc.compute_m_mat();
        disc.compute_kxi_mat();
        disc.compute_ik1_mat();
        disc.compute_f0_mat();

        disc
    }
    fn solve(&mut self, solutions: &Array3<f64>) {
        let nelem = self.mesh_param.elem_num;
        let nnode = self.mesh_param.node_num;
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut old_solutions: Array3<f64> = solutions.clone();
        let bnd_lqh: Array4<f64> = Array4::zeros((nelem, 2, cell_ngp, 1));
        while self.current_step < self.solver_param.final_step
            && self.current_time < self.solver_param.final_time
        {
            old_solutions.assign(solutions);
            let mut dt = self.compute_time_step(solutions);
            if self.current_time + dt > self.solver_param.final_time {
                dt = self.solver_param.final_time - self.current_time;
            }
            for ielem in 0..nelem {
                let solutions_slice: ArrayView2<f64> = solutions.slice(s![ielem, .., ..]);
                let residuals_slice: ArrayViewMut2<f64> = self.residuals.slice_mut(s![ielem, .., ..]);
                let bnd_lqh_slice: ArrayView3<f64> = bnd_lqh.slice(s![ielem, .., .., ..]);
                let lqh: Array3<f64> = self.local_space_time_predictor(solutions_slice, bnd_lqh_slice, dt);
                self.volume_integral(lqh, residuals_slice);
            }
            for inode in 0..nnode {
                let node = &self.mesh.nodes[inode];
                let ilelem = node.parent_elements[0];
                let irelem = node.parent_elements[1];
                let left_bnd_sol: ArrayView2<f64> = solutions.slice(s![ilelem, 1, .., ..]);
                let right_bnd_sol: ArrayView2<f64> = solutions.slice(s![irelem, 0, .., ..]);
                let left_res: ArrayViewMut2<f64> = self.residuals.slice_mut(s![ilelem, .., ..]);
                let right_res: ArrayViewMut2<f64> = self.residuals.slice_mut(s![irelem, .., ..]);
                self.edge_integral(left_bnd_sol, right_bnd_sol, left_res, right_res);
            }
            // apply bc

            // multiply inverse mass matrix, accounting for physical domain scaling
            for ielem in 0..nelem {
                let jacob_det = self.mesh.elements[ielem].jacob_det;
                self.residuals.slice_mut(s![ielem, .., ..]).assign(
                    &((1.0 / jacob_det) * self.ss_im_mat.dot(&self.residuals.slice(s![ielem, .., ..]))),
                );
            }
            // update solution
            solutions.assign(&(solutions + dt * self.residuals));
            // detect shock
            for ielem in 0..nelem {
                let old_sol: ArrayView3<f64> = {
                    let neighbor_num = self.mesh.elements[ielem].ineighbors.len();
                    let mut old_sol: Array3<f64> = Array3::zeros((neighbor_num + 1, cell_ngp, 1));
                    old_sol.slice_mut(s![0, .., ..]).assign(&old_solutions.slice(s![ielem, .., ..]));
                    for (i, &ineigh) in self.mesh.elements[ielem].ineighbors.iter().enumerate() {
                        let ineigh = ineigh as usize;
                        old_sol.slice_mut(s![i + 1, .., ..]).assign(&old_solutions.slice(s![ineigh, .., ..]));
                    }
                    old_sol.view()
                };
                let candidate_sol: ArrayView2<f64> = solutions.slice(s![ielem, .., ..]);
                if self.cell_shock_detector.detect_shock(old_sol, candidate_sol) {
                    // left for shock tracking
                }
                
            }
            self.current_time += dt;
            self.current_step += 1;
        }
    }

    fn compute_time_step(&self, solutions: ArrayView3<f64>) -> f64 {
        let nbasis = self.solver_param.cell_gp_num;
        let nelem = self.mesh_param.elem_num;
        let mut time_step = 1.0e10;
        for ielem in 0..nelem {
            // Compute average velocity in element
            let mut u = 0.0;
            for ibasis in 0..nbasis {
                u += solutions[[ielem, ibasis, 0]];
            }
            u /= nbasis as f64;

            // Wave speed for Burgers equation is |u|
            let speed = u.abs();
            let dx = self.mesh.elements[ielem].jacob_det;
            let dt = self.solver_param.cfl * dx / speed;

            time_step = time_step.min(dt);
        }

        time_step
    }
    fn local_space_time_predictor(
        &self,
        sol: ArrayView2<f64>, // (ndof, neq)
        bnd_lqh: ArrayView3<f64>, // (2, cell_ngp, neq)
        dt: f64,
    ) -> Array3<f64> {
        let cell_ngp = self.solver_param.cell_gp_num;
        let neq = self.solver_param.equation_num;
        // Dimensions: (time, x, var) for better memory access in Rust
        let mut lqh: Array3<f64> = Array3::zeros((cell_ngp, cell_ngp, neq)); // space-time DOF
        let mut lqhold: Array3<f64> = Array3::zeros((cell_ngp, cell_ngp, neq)); // old DOF
        let mut lfh: Array3<f64> = Array3::zeros((cell_ngp, cell_ngp, neq)); // flux tensor

        // Initial guess for current element
        for kgp in 0..cell_ngp {
            // time
            for igp in 0..cell_ngp {
                // x
                for ivar in 0..neq {
                    lqh[[kgp, igp, ivar]] = sol[[ivar, igp]];
                }
            }
        }

        // Picard iterations for current element
        for _iter in 0..self.solver_param.temporal_order + 1 {
            lqhold.assign(&lqh);
            // Compute fluxes
            for kgp in 0..cell_ngp {
                // time
                for igp in 0..cell_ngp {
                    // x
                    let f: Array1<f64> = flux1d(lqh.slice(s![kgp, igp, ..]));
                    lfh.slice_mut(s![kgp, igp, ..]).assign(&f);
                }
            }
            // update lqh
            for ivar in 0..neq {
                // Convert 2D views to 1D vectors for matrix multiplication
                let lqhold_slice: Array1<f64> = lqhold
                    .slice(s![.., .., ivar])
                    .into_shape(cell_ngp * cell_ngp)
                    .unwrap();
                let lfh_slice: Array1<f64> = lfh
                    .slice(s![.., .., ivar])
                    .into_shape(cell_ngp * cell_ngp)
                    .unwrap();
                // Perform matrix multiplication and store result back in lqh
                let result: Array1<f64> = self
                    .stst_ik1_mat
                    .dot(&(self.stst_f0_mat.dot(&lqhold_slice) + dt * self.stst_kxi_mat.dot(&lfh_slice)));
                lqh.slice_mut(s![.., .., ivar])
                    .assign(&result.into_shape((cell_ngp, cell_ngp)).unwrap());
            }
        }
        // Extract boundary values from space-time solution
        for kgp in 0..cell_ngp {
            // Left boundary (x=0)
            bnd_lqh.slice_mut(s![0, kgp, ..]).assign(&lqh.slice(s![kgp, 0, ..]));
            // Right boundary (x=1)
            bnd_lqh.slice_mut(s![1, kgp, ..]).assign(&lqh.slice(s![kgp, cell_ngp - 1, ..]));
        }
        lqh
    }
    fn volume_integral(
        &self, 
        lqh: ArrayView3<f64>, // (ntdof, nxdof, neq)
        res: ArrayViewMut2<f64> // (nxdof, neq)
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let nbasis = cell_ngp;
        let mut lfh: Array3<f64> = Array3::zeros((cell_ngp, cell_ngp, 1));
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let f: Array1<f64> = flux1d(lqh.slice(s![kgp, igp, ..]));
                lfh.slice_mut(s![kgp, igp, ..]).assign(&f);
            }
        }
        for idof in 0..cell_ngp {
            for ivar in 0..1 {
                let lfh_slice: Array1<f64> = lfh.slice(s![.., .., ivar]).into_shape(cell_ngp * cell_ngp).unwrap();
                res[[idof, ivar]] += self.sst_kxi_mat.dot(&lfh_slice.t());
            }
        }
    }
    fn edge_integral(
        &self,
        left_bnd_lqh: ArrayView2<f64>, // (ntgp, neq)
        right_bnd_lqh: ArrayView2<f64>, // (ntgp, neq)
        left_res: ArrayViewMut2<f64>, // (nxdof, neq)
        right_res: ArrayViewMut2<f64>, // (nxdof, neq)
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let nbasis = cell_ngp;
        for igp in 0..cell_ngp {
            let left_value: Array1<f64> = left_bnd_lqh.slice(s![igp, .., 0]);
            let right_value: Array1<f64> = right_bnd_lqh.slice(s![igp, .., 0]);
            let num_flux: Array1<f64> = rusanov(left_value, right_value);
            for ivar in 0..1 {
                left_res[[igp, ivar]] -= num_flux[ivar] * self.basis.phis_bnd_gps[[1, igp]];
                right_res[[igp, ivar]] += num_flux[ivar] * self.basis.phis_bnd_gps[[0, igp]];
            }
        }
    }
    fn apply_bc(&self, ) {
        
    }
}
