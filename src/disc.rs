pub mod basis;
pub mod boundary_conditions;
pub mod flux;
pub mod gauss_points;
pub mod mesh;
// pub mod riemann_solver;
pub mod burgers1d;
// pub mod euler1d;


/*
pub struct Disc1dEuler<'a> {
    pub current_time: f64,
    pub current_step: usize,
    pub gauss_points: GaussPoints1d,
    pub basis: LagrangeBasis1D,
    pub mesh: Mesh1d,
    pub flow_param: &'a FlowParameters,
    pub mesh_param: &'a MeshParameters,
    pub solver_param: &'a SolverParameters,
    m_mat: Array<f64, Ix2>,   // mass matrix
    kx_mat: Array<f64, Ix2>,  // spatial stiffness matrix
    ik1_mat: Array<f64, Ix2>, // inverse temporal stiffness matrix
    f0_mat: Array<f64, Ix2>,  // mass matrix at relative time 0
}
impl<'a> Disc1dEuler<'a> {
    fn new(
        gauss_points: GaussPoints1d,
        basis: LagrangeBasis1D,
        mesh: Mesh1d,
        flow_param: &'a FlowParameters,
        mesh_param: &'a MeshParameters,
        solver_param: &'a SolverParameters,
    ) -> Disc1dEuler<'a> {
        let cell_ngp = solver_param.cell_gp_num;
        let m_mat = Array::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let kx_mat = Array::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let ik1_mat = Array::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let f0_mat = Array::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        let mut disc = Disc1dEuler {
            current_time: 0.0,
            current_step: 0,
            gauss_points,
            basis,
            mesh,
            flow_param,
            mesh_param,
            solver_param,
            m_mat,
            kx_mat,
            ik1_mat,
            f0_mat,
        };
        disc.compute_m_mat();
        disc.compute_kx_mat();
        disc.compute_ik1_mat();
        disc.compute_f0_mat();

        disc
    }
    fn solve(&self, solutions: &Array<f64, Ix3>, residuals: &mut Array<f64, Ix3>) {
        let nelem = self.mesh_param.elem_num;
        let nedge = self.mesh_param.edge_num;
        while self.current_step < self.solver_param.final_step && self.current_time < self.solver_param.final_time {
            let mut dt = self.compute_time_step(solutions);
            if self.current_time + dt > self.solver_param.final_time {
                dt = self.solver_param.final_time - self.current_time;
            }
            for ielem in 0..nelem {
                let solutions_slice = solutions.slice(s![ielem, .., ..]);
                let residuals_slice = residuals.slice(s![ielem, .., ..]);
                let lqh = self.local_space_time_predictor(solutions_slice, dt);
                self.volume_integral(lqh, residuals_slice);
            }
            for iedge in 0..nedge {
                let edge = &self.mesh.edges[iedge];
                let ilelem = edge.parent_elements[0];
                let irelem = edge.parent_elements[1];
                let left_sol = solutions.slice(s![ilelem, .., ..]);
                let right_sol = solutions.slice(s![irelem, .., ..]);
                let left_res = residuals.slice(s![ilelem, .., ..]);
                let right_res = residuals.slice(s![irelem, .., ..]);
                self.edge_integral(solutions_slice, residuals_slice, iedge);
            }
            self.current_time += dt;
            self.current_step += 1;
        }

        self.integrate_over_edges(residuals, solutions);
        self.apply_bc(residuals, solutions);
    }
    fn compute_m_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        for i in 0..cell_ngp * cell_ngp {
            // ith space-time DOF
            for j in 0..cell_ngp * cell_ngp {
                // jth space-time DOF
                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    // spatial quadrature point
                    for k_t in 0..cell_ngp {
                        // temporal quadrature point
                        let ix = i % cell_ngp; // spatial index
                        let it = i / cell_ngp; // temporal index
                        let jx = j % cell_ngp; // spatial index
                        let jt = j / cell_ngp; // temporal index

                        sum += self.basis.cell_gauss_weights[k_x]
                            * self.basis.cell_gauss_weights[k_t]
                            * self.basis.phis_cell_gps[(k_x, ix)]
                            * self.basis.phis_cell_gps[(k_t, it)]
                            * self.basis.phis_cell_gps[(k_t, jt)]
                            * self.basis.phis_cell_gps[(k_x, jx)];
                    }
                }
                self.m_mat[[i, j]] = sum;
            }
        }
    }
    fn compute_kx_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        for i in 0..cell_ngp * cell_ngp {
            // ith space-time DOF
            for j in 0..cell_ngp * cell_ngp {
                // jth space-time DOF
                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    // spatial quadrature point
                    for k_t in 0..cell_ngp {
                        // temporal quadrature point
                        let ix = i % cell_ngp; // spatial index
                        let it = i / cell_ngp; // temporal index
                        let jx = j % cell_ngp; // spatial index
                        let jt = j / cell_ngp; // temporal index

                        sum += self.basis.cell_gauss_weights[k_x]
                            * self.basis.cell_gauss_weights[k_t]
                            * self.basis.dphis_cell_gps[(k_x, ix)]
                            * self.basis.phis_cell_gps[(k_t, it)]
                            * self.basis.phis_cell_gps[(k_t, jt)]
                            * self.basis.phis_cell_gps[(k_x, jx)];
                    }
                }
                self.kx_mat[[i, j]] = sum;
            }
        }
    }
    fn compute_ik1_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut f1_mat = Array::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    // spatial quadrature
                    let ix = i % cell_ngp; // spatial index
                    let it = i / cell_ngp; // temporal index
                    let jx = j % cell_ngp;
                    let jt = j / cell_ngp;

                    sum += self.basis.cell_gauss_weights[k_x]
                        * self.basis.phis_cell_gps[(k_x, ix)]
                        * self.basis.phis_cell_gps[(1, it)]
                        * self.basis.phis_cell_gps[(k_x, jx)]
                        * self.basis.phis_cell_gps[(1, jt)];
                }
                f1_mat[[i, j]] = sum;
            }
        }
        // Compute temporal stiffness matrix for space-time basis
        let mut t_mat = Array::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    // spatial quadrature
                    for k_t in 0..cell_ngp {
                        // temporal quadrature
                        let ix = i % cell_ngp; // spatial index
                        let it = i / cell_ngp; // temporal index
                        let jx = j % cell_ngp;
                        let jt = j / cell_ngp;

                        sum += self.basis.cell_gauss_weights[k_x]
                            * self.basis.cell_gauss_weights[k_t]
                            * self.basis.phis_cell_gps[(k_x, ix)]
                            * self.basis.dphis_cell_gps[(k_t, it)]
                            * self.basis.phis_cell_gps[(k_x, jx)]
                            * self.basis.phis_cell_gps[(k_t, jt)];
                    }
                }
                t_mat[[i, j]] = sum;
            }
        }
        let mut k1_mat = Array::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        // Subtract temporal stiffness matrix from f1_matrix
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                k1_mat[[i, j]] = f1_mat[[i, j]] - t_mat[[i, j]];
            }
        }
        self.ik1_mat = k1_mat.inv().unwrap();
    }
    fn compute_f0_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    // spatial quadrature
                    let ix = i % cell_ngp; // spatial index
                    let it = i / cell_ngp; // temporal index
                    let jx = j % cell_ngp;
                    let jt = j / cell_ngp;

                    sum += self.basis.cell_gauss_weights[k_x] * 
                           self.basis.phis_cell_gps[(k_x, ix)] * 
                           self.basis.phis_cell_gps[(0, it)] *  // Evaluate temporal basis at t=0
                           self.basis.phis_cell_gps[(k_x, jx)] * 
                           self.basis.phis_cell_gps[(0, jt)]; // Evaluate temporal basis at t=0
                }
                self.f0_mat[[i, j]] = sum;
            }
        }
    }
    fn compute_time_step(&self, solutions: ArrayView3<f64>) -> f64 {
        let nbasis = self.solver_param.cell_gp_num;
        let nelem = self.mesh_param.elem_num;
        let hcr = self.flow_param.hcr;
        let mut time_step = 1.0e10;
        for ielem in 0..nelem {
            let mut rho = 0.0;
            let mut rho_u = 0.0;
            let mut rho_v = 0.0;
            let mut rho_e = 0.0;
            for ibasis in 0..nbasis {
                rho += solutions[[ielem, 0, ibasis]];
                rho_u += solutions[[ielem, 1, ibasis]];
                rho_v += solutions[[ielem, 2, ibasis]];
                rho_e += solutions[[ielem, 3, ibasis]];
            }
            let rho = rho;
            let u = rho_u / rho;
            let v = rho_v / rho;
            let p = (hcr - 1.0) * (rho_e - 0.5 * rho * (u * u + v * v));
            let c = (hcr * p / rho).sqrt();
            let speed = (u * u + v * v).sqrt() + c;
            let dx = self.mesh.elements[ielem].minimum_height;
            let dt = self.solver_param.cfl * dx / speed;
            if dt < time_step {
                time_step = dt;
            }
        }
        dbg!(&time_step);
        time_step
    }
    fn local_space_time_predictor(
        &self,
        solutions_slice: ArrayView2<f64>,
        dt: f64,
    ) -> Array<f64, Ix3> {
        let cell_ngp = self.solver_param.cell_gp_num;
        let neq = self.solver_param.equation_num;
        // Dimensions: (time, x, var) for better memory access in Rust
        let mut lqh = Array3::zeros((cell_ngp, cell_ngp, neq)); // space-time DOF
        let mut lqhold = Array3::zeros((cell_ngp, cell_ngp, neq)); // old DOF
        let mut lfh = Array3::zeros((cell_ngp, cell_ngp, neq)); // flux tensor

        // Initial guess for current element
        for kgp in 0..cell_ngp {
            // time
            for igp in 0..cell_ngp {
                // x
                for ivar in 0..neq {
                    lqh[[kgp, igp, ivar]] = solutions_slice[[ivar, igp]];
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
                    let f = flux1d(lqh.slice(s![kgp, igp, ..]), self.flow_param.hcr);
                    lfh.slice_mut(s![kgp, igp, ..]).assign(&f);
                }
            }
            // update solution
            for ivar in 0..neq {
                // Convert 2D views to 1D vectors for matrix multiplication
                let lqhold_slice = lqhold.slice(s![.., .., ivar]).into_shape(cell_ngp * cell_ngp).unwrap();
                let lfh_slice = lfh.slice(s![.., .., ivar]).into_shape(cell_ngp * cell_ngp).unwrap();
                // Perform matrix multiplication and store result back in lqh
                let result = self.ik1_mat.dot(&(self.f0_mat.dot(&lqhold_slice) + dt * self.kx_mat.dot(&lfh_slice)));
                lqh.slice_mut(s![.., .., ivar]).assign(&result.into_shape((cell_ngp, cell_ngp)).unwrap());
            }
        }
        lqh
    }
    fn volume_integral(&self, lqh: ArrayView3<f64>, residuals_slice: ArrayView2<f64>) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let nbasis = cell_ngp;
        let neq = self.solver_param.equation_num;
        let mut lfh = Array3::zeros((cell_ngp, cell_ngp, neq));
        for kgp in 0..cell_ngp {
            for igp in 0..cell_ngp {
                let f = flux1d(lqh.slice(s![kgp, igp, ..]), self.flow_param.hcr);
                lfh.slice_mut(s![kgp, igp, ..]).assign(&f);
            }
        }
        for igp in 0..cell_ngp {
            for ivar in 0..neq {
                let lfh_slice = lfh.slice(s![.., .., ivar]).into_shape(cell_ngp * cell_ngp).unwrap();
                residuals_slice[[igp, ivar]] += self.kx_mat.dot(&lfh_slice);
            }
        }
    }
    fn integrate_over_edges(&self, residuals: &mut Array<f64, Ix2>, solutions: &Array<f64, Ix2>) {
        let nelem = self.mesh_param.number_of_elements;
        let cell_ngp = self.solver_param.number_of_cell_gp;
        let nbasis = cell_ngp;
        let neq = self.solver_param.number_of_equations;
        for ivertex in self.mesh.internal_vertex_indices.iter() {
            let vertex = &self.mesh.vertices[*ivertex];
            let ilelem = vertex.iedges[0] as usize;
            let irelem = vertex.iedges[1] as usize;
            let left_dofs = solutions.slice(s![ilelem, .., ..]);
            let right_dofs = solutions.slice(s![irelem, .., ..]);
            let left_values: Array<f64, Ix1> = left_dofs.slice(s![.., -1]).to_owned();
            let right_values: Array<f64, Ix1> = right_dofs.slice(s![.., 0]).to_owned();
            let num_flux = match hllc1d(&left_values, &right_values, &self.flow_param.hcr) {
                Ok(flux) => flux,
                Err(e) => {
                    println!("{}", e);
                    println!("ivertex: {:?}", ivertex);
                    panic!("Error in HLLC flux computation!");
                }
            };
            for ivar in 0..residuals.shape()[1] {
                for ibasis in 0..residuals.shape()[2] {
                    residuals[[ilelem, ivar, ibasis]] -=
                        num_flux[ivar] * self.basis.phis_cell_gps[[cell_ngp, ibasis]];
                    residuals[[irelem, ivar, ibasis]] +=
                        num_flux[ivar] * self.basis.phis_cell_gps[[0, ibasis]];
                }
            }
        }
    }
    fn apply_bc(&self, residuals: &mut Array<f64, Ix2>, solutions: &Array<f64, Ix2>) {
        let cell_ngp = self.solver_param.number_of_cell_gp;
        let nelem = self.mesh.elements.len();
        let neq = self.solver_param.number_of_equations;
        let nbasis = cell_ngp;
        let cell_weights = &self.basis.cell_gauss_weights;
        for boundary_patch in self.mesh.boundary_patches.iter() {
            let boundary_vertex = &self.mesh.vertices[boundary_patch.ivertex];
            let iinrelem = boundary_vertex.iedges[0];
            let boundary_type = &boundary_patch.boundary_type;
            let (left_values, iphi, normal) = {
                if boundary_vertex.in_edge_indices[0] == 0 {
                    (solutions.slice(s![iinrelem, .., 0]), 0, -1.0)
                } else {
                    (solutions.slice(s![iinrelem, .., -1]), nbasis - 1, 1.0)
                }
            };
            match boundary_type {
                BoundaryType::Wall => {
                    let pressure = (self.flow_param.hcr - 1.0)
                        * (left_values[2]
                            - 0.5 * (left_values[1] * left_values[1]) / left_values[0]);
                    let boundary_inviscid_flux = [0.0, pressure * normal, 0.0];
                    for ivar in 0..neq {
                        for ibasis in 0..nbasis {
                            residuals[[iinrelem, ivar, ibasis]] -= cell_weights[ibasis]
                                * boundary_inviscid_flux[ivar]
                                * self.basis.phis_cell_gps[[iphi, ibasis]];
                        }
                    }
                }
                BoundaryType::FarField => {}
            }
        }
    }
}
*/
