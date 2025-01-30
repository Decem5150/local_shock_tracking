use ndarray::Array2;
use ndarray_linalg::Inverse;
use super::Disc1dEuler;

impl<'a> Disc1dEuler<'a> {
    pub fn compute_m_mat(&mut self) {
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
    pub fn compute_kx_mat(&mut self) {
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
    pub fn compute_ik1_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut f1_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
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
        let mut t_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
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
        let mut k1_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        // Subtract temporal stiffness matrix from f1_matrix
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                k1_mat[[i, j]] = f1_mat[[i, j]] - t_mat[[i, j]];
            }
        }
        self.ik1_mat = k1_mat.inv().unwrap();
    }
    pub fn compute_f0_mat(&mut self) {
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
}