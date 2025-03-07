pub mod boundary_condition;
mod cell_shock_detector;
mod flux;
mod precompute_matrix;
mod riemann_solver;
// mod shock_tracking;
use super::{
    basis::lagrange1d::LagrangeBasis1DLobatto,
    mesh::mesh1d::{Element1d, Mesh1d},
};
use crate::{
    io::write_to_csv::write_to_csv,
    solver::{FlowParameters, SolverParameters},
};
use boundary_condition::{BoundaryQuantity1d, BoundaryType};
use flux::flux1d;
use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut2,
    ArrayViewMut3, ArrayViewMut4, s,
};
use riemann_solver::rusanov::rusanov;

pub struct Disc1dBurgers<'a> {
    pub current_time: f64,
    pub current_step: usize,
    pub basis: LagrangeBasis1DLobatto,
    mesh: &'a Mesh1d,
    flow_param: &'a FlowParameters,
    solver_param: &'a SolverParameters,
    ss_m_mat: Array2<f64>,     // mass matrix of two space polynomials
    ss_im_mat: Array2<f64>,    // inverse mass matrix of two space polynomials
    sst_kxi_mat: Array2<f64>, // spatial stiffness matrix of space polynomial and space-time polynomial
    stst_m_mat: Array2<f64>,  // mass matrix of two space-time polynomials
    stst_im_mat: Array2<f64>, // inverse mass matrix of two space-time polynomials
    stst_kxi_mat: Array2<f64>, // spatial stiffness matrix of two space-time polynomials
    stst_ik1_mat: Array2<f64>, // inverse temporal stiffness matrix of two space-time polynomials
    sts_f0_mat: Array2<f64>,  // mass matrix at relative time 0 for two space-time polynomials
}
impl<'a> Disc1dBurgers<'a> {
    pub fn new(
        basis: LagrangeBasis1DLobatto,
        mesh: &'a Mesh1d,
        flow_param: &'a FlowParameters,
        solver_param: &'a SolverParameters,
    ) -> Disc1dBurgers<'a> {
        let cell_ngp = solver_param.cell_gp_num;
        let ss_m_mat: Array2<f64> = Array2::zeros((cell_ngp, cell_ngp));
        let ss_im_mat: Array2<f64> = Array2::zeros((cell_ngp, cell_ngp));
        let sst_kxi_mat: Array2<f64> = Array2::zeros((cell_ngp, cell_ngp * cell_ngp));
        let stst_m_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let stst_im_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let stst_kxi_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let stst_ik1_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let sts_f0_mat: Array2<f64> = Array2::zeros((cell_ngp * cell_ngp, cell_ngp));
        let mut disc = Disc1dBurgers {
            current_time: 0.0,
            current_step: 0,
            basis,
            mesh,
            flow_param,
            solver_param,
            ss_m_mat,
            ss_im_mat,
            sst_kxi_mat,
            stst_m_mat,
            stst_im_mat,
            stst_kxi_mat,
            stst_ik1_mat,
            sts_f0_mat,
        };
        disc.compute_m_mat();
        disc.compute_kxi_mat();
        disc.compute_ik1_mat();
        disc.compute_f0_mat();
        /*
        dbg!(&disc.ss_m_mat);
        dbg!(&disc.ss_im_mat);
        dbg!(&disc.sst_kxi_mat);
        dbg!(&disc.stst_m_mat);
        dbg!(&disc.stst_im_mat);
        dbg!(&disc.stst_kxi_mat);
        dbg!(&disc.stst_ik1_mat);
        dbg!(&disc.sts_f0_mat);
        */
        disc
    }
    pub fn solve(&mut self, mut solutions: ArrayViewMut3<f64>) {
        let nelem = self.mesh.elem_num;
        let nnode = self.mesh.node_num;
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut residuals: Array3<f64> = Array3::zeros((nelem, cell_ngp, 1));
        let mut old_solutions: Array3<f64> = solutions.to_owned();
        // let bnd_lqh: Array4<f64> = Array4::zeros((nelem, 2, cell_ngp, 1));
        let mut lqh: Array4<f64> = Array4::zeros((nelem, cell_ngp, cell_ngp, 1));
        write_to_csv(
            solutions.view(),
            self.mesh,
            &self.basis,
            self.current_time,
            &format!("outputs/solutions_{}.csv", self.current_step),
        )
        .unwrap();
        while self.current_step < self.solver_param.final_step
            && self.current_time < self.solver_param.final_time
        {
            residuals.fill(0.0);
            old_solutions.assign(&solutions);
            let mut dt = self.compute_time_step(solutions.view());
            // let mut dt = 0.002;
            if self.current_time + dt > self.solver_param.final_time {
                dt = self.solver_param.final_time - self.current_time;
            }
            for ielem in 0..nelem {
                let elem = &self.mesh.elements[ielem];
                let solutions_slice: ArrayView2<f64> = solutions.slice(s![ielem, .., ..]);
                // let bnd_lqh_slice: ArrayViewMut3<f64> = bnd_lqh.slice_mut(s![ielem, .., .., ..]);
                self.local_space_time_predictor(
                    lqh.slice_mut(s![ielem, .., .., ..]),
                    solutions_slice,
                    elem,
                    dt,
                );
                // let residuals_slice: ArrayViewMut2<f64> = residuals.slice_mut(s![ielem, .., ..]);
                // self.volume_integral(lqh.slice(s![ielem, .., .., ..]), residuals_slice, elem);
            }
            /*
            write_to_csv(
                lqh.view().slice(s![.., cell_ngp - 1, .., ..]),
                self.mesh,
                &self.basis,
                &format!("outputs/lqh_{}.csv", self.current_step),
            )
            .unwrap();
            */
            for ielem in 0..nelem {
                let elem = &self.mesh.elements[ielem];
                let residuals_slice: ArrayViewMut2<f64> = residuals.slice_mut(s![ielem, .., ..]);
                self.volume_integral(lqh.slice(s![ielem, .., .., ..]), residuals_slice, elem);
            }
            /*
            write_to_csv(
                residuals.view(),
                self.mesh,
                &self.basis,
                &format!("outputs/residuals_{}.csv", self.current_step),
            )
            .unwrap();
            */
            for &inode in self.mesh.internal_nodes.iter() {
                let node = &self.mesh.nodes[inode];
                let ilelem = node.parent_elements[0];
                let irelem = node.parent_elements[1];
                let left_elem = &self.mesh.elements[ilelem as usize];
                let right_elem = &self.mesh.elements[irelem as usize];
                let left_bnd_lqh: ArrayView2<f64> = lqh.slice(s![ilelem, .., cell_ngp - 1, ..]);
                let right_bnd_lqh: ArrayView2<f64> = lqh.slice(s![irelem, .., 0, ..]);
                let (left_res, right_res) =
                    residuals.multi_slice_mut((s![ilelem, .., ..], s![irelem, .., ..]));
                self.edge_integral(
                    left_bnd_lqh,
                    right_bnd_lqh,
                    left_res,
                    right_res,
                    left_elem,
                    right_elem,
                );
            }
            // apply bc
            // self.apply_bc(lqh.view(), residuals.view_mut());
            // multiply inverse mass matrix, accounting for physical domain scaling
            for ielem in 0..nelem {
                let jacob_det = self.mesh.elements[ielem].jacob_det;
                let computed =
                    (1.0 / jacob_det) * self.ss_im_mat.dot(&residuals.slice(s![ielem, .., ..]));
                residuals.slice_mut(s![ielem, .., ..]).assign(&computed);
            }
            // update solution
            solutions.scaled_add(dt, &residuals.view());
            // detect shock

            for &ielem in self.mesh.internal_nodes.iter() {
                let old_sol: Array3<f64> = {
                    let neighbor_num = self.mesh.elements[ielem].ineighbors.len();
                    let mut old_sol: Array3<f64> = Array3::zeros((neighbor_num + 1, cell_ngp, 1));
                    old_sol
                        .slice_mut(s![0, .., ..])
                        .assign(&old_solutions.slice(s![ielem, .., ..]));
                    for (i, &ineigh) in self.mesh.elements[ielem].ineighbors.indexed_iter() {
                        let ineigh = ineigh as usize;
                        old_sol
                            .slice_mut(s![i + 1, .., ..])
                            .assign(&old_solutions.slice(s![ineigh, .., ..]));
                    }
                    old_sol
                };
                /*
                write_to_csv(
                    solutions.view(),
                    self.mesh,
                    &self.basis,
                    &format!("outputs/solutions_{}.csv", self.current_step + 1),
                )
                .unwrap();
                */
                let candidate_sol: ArrayView2<f64> = solutions.slice(s![ielem, .., ..]);
                if self.detect_shock(old_sol.view(), candidate_sol) {
                    println!("Index of shock: {}", ielem);
                    let final_error = write_to_csv(
                        solutions.view(),
                        self.mesh,
                        &self.basis,
                        self.current_time,
                        &format!("outputs/solutions_final.csv"),
                    )
                    .unwrap();
                    println!("Final L² error: {:.4e}", final_error);
                    println!("Final step: {}", self.current_step);
                    println!("Final time: {}", self.current_time);
                    panic!("Shock detected!");
                    // left for shock tracking
                }
            }

            self.current_time += dt;
            self.current_step += 1;
            println!("step: {}, time: {}", self.current_step, self.current_time);

            if self.current_step % 10 == 0 {
                let error = write_to_csv(
                    solutions.view(),
                    self.mesh,
                    &self.basis,
                    self.current_time,
                    &format!("outputs/solutions_{}.csv", self.current_step),
                )
                .unwrap();
                println!("Step {} L² error: {:.4e}", self.current_step, error);
            }
        }

        // Final error calculation
        let final_error = write_to_csv(
            solutions.view(),
            self.mesh,
            &self.basis,
            self.current_time,
            &format!("outputs/solutions_final.csv"),
        )
        .unwrap();
        println!("Final L² error: {:.4e}", final_error);
        println!("Final step: {}", self.current_step);
        println!("Final time: {}", self.current_time);
    }
    pub fn initialize_solution(
        &mut self,
        mut solutions: ArrayViewMut3<f64>,
        init_func: &dyn Fn(f64) -> f64,
    ) {
        let nelem = self.mesh.elem_num;
        let cell_ngp = self.solver_param.cell_gp_num;
        for ielem in 0..nelem {
            let elem = &self.mesh.elements[ielem];
            let x_left = self.mesh.nodes[elem.inodes[0]].x;
            let jacob_det = elem.jacob_det;
            for igp in 0..cell_ngp {
                let xi = self.basis.cell_gauss_points[igp];
                let x = x_left + xi * jacob_det;
                solutions[[ielem, igp, 0]] = init_func(x);
            }
        }
    }
    fn compute_time_step(&self, solutions: ArrayView3<f64>) -> f64 {
        let nbasis = self.solver_param.cell_gp_num;
        let nelem = self.mesh.elem_num;
        let mut time_step: f64 = 1.0e10;
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
            let dt = self.solver_param.cfl * dx
                / ((self.solver_param.polynomial_order as f64 * 2.0 + 1.0) * speed);

            time_step = time_step.min(dt);
        }

        time_step
    }
    fn local_space_time_predictor(
        &mut self,
        mut lqh: ArrayViewMut3<f64>, // (cell_ngp, cell_ngp, neq)
        sol: ArrayView2<f64>,        // (ndof, neq)
        elem: &Element1d,
        dt: f64,
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let jacob_det = elem.jacob_det;
        // Dimensions: (time, x, var) for better memory access in Rust
        let mut lfh: Array3<f64> = Array3::zeros((cell_ngp, cell_ngp, 1)); // flux tensor

        // Initial guess for current element
        for kgp in 0..cell_ngp {
            // time
            for igp in 0..cell_ngp {
                // x
                for ivar in 0..1 {
                    lqh[[kgp, igp, ivar]] = sol[[igp, ivar]];
                }
            }
        }

        // Picard iterations for current element
        for _iter in 0..self.solver_param.polynomial_order + 1 {
            // Compute fluxes
            for kgp in 0..cell_ngp {
                // time
                for igp in 0..cell_ngp {
                    // x
                    let f: Array1<f64> = flux1d(lqh.slice(s![kgp, igp, ..]));
                    lfh.slice_mut(s![kgp, igp, ..])
                        .assign(&(dt * &f / jacob_det));
                }
            }
            // update lqh
            for ivar in 0..1 {
                // Convert 2D views to 1D vectors for matrix multiplication
                let lfh_slice: ArrayView1<f64> = lfh
                    .slice(s![.., .., ivar])
                    .into_shape(cell_ngp * cell_ngp)
                    .unwrap();
                // Perform matrix multiplication and store result back in lqh
                let result: Array1<f64> = self.stst_ik1_mat.dot(
                    &(self.sts_f0_mat.dot(&sol.slice(s![.., ivar]))
                        - self.stst_kxi_mat.dot(&lfh_slice)),
                );
                lqh.slice_mut(s![.., .., ivar])
                    .assign(&result.into_shape((cell_ngp, cell_ngp)).unwrap());
            }
        }
    }
    fn volume_integral(
        &mut self,
        lqh: ArrayView3<f64>,        // (ntdof, nxdof, neq)
        mut res: ArrayViewMut2<f64>, // (nxdof, neq)
        elem: &Element1d,
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut lfh: Array3<f64> = Array3::zeros((cell_ngp, cell_ngp, 1));
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let f: Array1<f64> = flux1d(lqh.slice(s![kgp, igp, ..]));
                lfh.slice_mut(s![kgp, igp, ..]).assign(&f);
            }
        }
        for ivar in 0..1 {
            let lfh_slice: ArrayView1<f64> = lfh
                .slice(s![.., .., ivar])
                .into_shape(cell_ngp * cell_ngp)
                .unwrap();
            res.slice_mut(s![.., ivar])
                .scaled_add(1.0, &self.sst_kxi_mat.dot(&lfh_slice));
        }
    }
    fn edge_integral(
        &self,
        left_bnd_lqh: ArrayView2<f64>,     // (ntgp, neq)
        right_bnd_lqh: ArrayView2<f64>,    // (ntgp, neq)
        mut left_res: ArrayViewMut2<f64>,  // (nxdof, neq)
        mut right_res: ArrayViewMut2<f64>, // (nxdof, neq)
        left_elem: &Element1d,
        right_elem: &Element1d,
    ) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let nbasis = cell_ngp;
        let weights = &self.basis.cell_gauss_weights;
        for ibasis in 0..nbasis {
            for kgp in 0..cell_ngp {
                // time
                let left_value: ArrayView1<f64> = left_bnd_lqh.slice(s![kgp, ..]);
                let right_value: ArrayView1<f64> = right_bnd_lqh.slice(s![kgp, ..]);
                let num_flux: Array1<f64> = rusanov(left_value, right_value);
                for ivar in 0..1 {
                    left_res[[ibasis, ivar]] -= weights[kgp]
                        * num_flux[ivar]
                        * self.basis.phis_cell_gps[[cell_ngp - 1, ibasis]];
                    right_res[[ibasis, ivar]] +=
                        weights[kgp] * num_flux[ivar] * self.basis.phis_cell_gps[[0, ibasis]];
                }
            }
        }
    }
    fn apply_bc(&self, lqh: ArrayView4<f64>, mut residuals: ArrayViewMut3<f64>) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let weights = &self.basis.cell_gauss_weights;
        for ipatch in 0..2 {
            let boundary_type = &self.mesh.boundary_patches[ipatch].boundary_type;
            let inode = self.mesh.boundary_patches[ipatch].inode;
            let node = &self.mesh.nodes[inode];
            for (iparent, &ielem) in node.parent_elements.indexed_iter() {
                if ielem != -1 {
                    let ielem = ielem as usize;
                    let local_id = node.local_ids[iparent];
                    let bnd_lqh: ArrayView2<f64> = match local_id {
                        0 => lqh.slice(s![ielem, 0, .., ..]),
                        1 => lqh.slice(s![ielem, cell_ngp - 1, .., ..]),
                        _ => unreachable!(),
                    };
                    match boundary_type {
                        BoundaryType::Dirichlet => {
                            let boundary_quantity = &self.mesh.boundary_patches[ipatch]
                                .boundary_quantity
                                .as_ref()
                                .unwrap();
                            for ibasis in 0..cell_ngp {
                                for kgp in 0..cell_ngp {
                                    let interior_value: ArrayView1<f64> =
                                        bnd_lqh.slice(s![kgp, ..]);
                                    let boundary_flux: Array1<f64> = rusanov(
                                        interior_value,
                                        Array1::from_vec(vec![boundary_quantity.u; cell_ngp])
                                            .view(),
                                    );
                                    match local_id {
                                        0 => {
                                            residuals[[ielem, ibasis, 0]] -= 0.5
                                                * weights[kgp]
                                                * boundary_flux[0]
                                                * self.basis.phis_cell_gps[[cell_ngp - 1, ibasis]]
                                        }
                                        1 => {
                                            residuals[[ielem, ibasis, 0]] -= 0.5
                                                * weights[kgp]
                                                * boundary_flux[0]
                                                * self.basis.phis_cell_gps[[0, ibasis]]
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                            }
                        }
                        BoundaryType::Neumann => {}
                    }
                } else {
                    continue;
                }
            }
        }
    }
}
