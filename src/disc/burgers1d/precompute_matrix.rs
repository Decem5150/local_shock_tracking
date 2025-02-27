use super::Disc1dBurgers;
use ndarray::Array2;
use ndarray_linalg::Inverse;

impl<'a> Disc1dBurgers<'a> {
    pub fn compute_m_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        // compute mass matrix of the two space polynomials
        for i in 0..cell_ngp {
            for j in 0..cell_ngp {
                if i == j {
                    self.ss_m_mat[[i, j]] = self.basis.cell_gauss_weights[i];
                } else {
                    self.ss_m_mat[[i, j]] = 0.0;
                }
            }
        }
        self.ss_im_mat = self.ss_m_mat.inv().unwrap();
        // compute mass matrix of the two space-time polynomials
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                let ix = i % cell_ngp; // spatial index
                let it = i / cell_ngp; // temporal index
                let jx = j % cell_ngp; // spatial index
                let jt = j / cell_ngp; // temporal index

                // Only compute when indices match (otherwise result is 0)
                if ix == jx && it == jt {
                    self.stst_m_mat[[i, j]] =
                        self.basis.cell_gauss_weights[ix] * self.basis.cell_gauss_weights[it];
                } else {
                    self.stst_m_mat[[i, j]] = 0.0;
                }
            }
        }
        self.stst_im_mat = self.stst_m_mat.inv().unwrap();
    }
    pub fn compute_kxi_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        // compute spatial stiffness matrix of the space polynomial and the space-time polynomial
        for i in 0..cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                let jx = j % cell_ngp;
                let jt = j / cell_ngp;

                // Due to Lagrange polynomial properties, only compute at matching indices
                self.sst_kxi_mat[[i, j]] = self.basis.cell_gauss_weights[jx]
                    * self.basis.cell_gauss_weights[jt]
                    * self.basis.dphis_cell_gps[(jx, i)];
            }
        }

        // compute spatial stiffness matrix of the two space-time polynomials
        for i in 0..cell_ngp * cell_ngp {
            // ith space-time DOF
            let ix = i % cell_ngp; // spatial index
            let it = i / cell_ngp; // temporal index
            for j in 0..cell_ngp * cell_ngp {
                // jth space-time DOF
                let jx = j % cell_ngp; // spatial index
                let jt = j / cell_ngp; // temporal index
                // Spatial integral
                let mut spatial = 0.0;
                for k_x in 0..cell_ngp {
                    spatial += self.basis.cell_gauss_weights[k_x]
                        * self.basis.dphis_cell_gps[(k_x, jx)]
                        * self.basis.phis_cell_gps[(k_x, ix)];
                }

                // Temporal integral
                let temporal =
                    self.basis.cell_gauss_weights[it] * self.basis.phis_cell_gps[(it, jt)];

                self.stst_kxi_mat[[i, j]] = spatial * temporal;
            }
        }
    }
    pub fn compute_ik1_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        let mut f1_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        for i in 0..cell_ngp * cell_ngp {
            let ix = i % cell_ngp; // Spatial component
            let it = i / cell_ngp; // Temporal component
            for j in 0..cell_ngp * cell_ngp {
                let jx = j % cell_ngp;
                let jt = j / cell_ngp;

                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    sum += self.basis.cell_gauss_weights[k_x]
                        * self.basis.phis_cell_gps[(k_x, ix)]  // φ_i^x(x_k)
                        * self.basis.phis_cell_gps[(k_x, jx)]  // φ_j^x(x_k)
                        * self.basis.phis_cell_gps[(cell_ngp - 1, it)] // φ_i^t(x_{cell_ngp-1})
                        * self.basis.phis_cell_gps[(cell_ngp - 1, jt)]; // φ_j^t(x_{cell_ngp-1})
                }
                f1_mat[[i, j]] = sum;
            }
        }
        // Compute temporal stiffness matrix for space-time basis
        let mut t_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        for i in 0..cell_ngp * cell_ngp {
            // ith space-time DOF
            let ix = i % cell_ngp; // spatial index
            let it = i / cell_ngp; // temporal index
            for j in 0..cell_ngp * cell_ngp {
                // jth space-time DOF
                let jx = j % cell_ngp; // spatial index
                let jt = j / cell_ngp; // temporal index
                // Spatial integral
                let spatial =
                    self.basis.cell_gauss_weights[ix] * self.basis.phis_cell_gps[(ix, jx)];
                // Temporal integral
                let mut temporal = 0.0;
                for k_t in 0..cell_ngp {
                    temporal += self.basis.cell_gauss_weights[k_t]
                        * self.basis.dphis_cell_gps[(k_t, it)]
                        * self.basis.phis_cell_gps[(k_t, jt)];
                }
                t_mat[[i, j]] = spatial * temporal;
            }
        }
        let mut k1_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        // Subtract temporal stiffness matrix from f1_matrix
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                k1_mat[[i, j]] = f1_mat[[i, j]] - t_mat[[i, j]];
            }
        }
        self.stst_ik1_mat = k1_mat.inv().unwrap();
    }
    pub fn compute_f0_mat(&mut self) {
        let cell_ngp = self.solver_param.cell_gp_num;
        for i in 0..cell_ngp * cell_ngp {
            let ix = i % cell_ngp; // spatial index
            let it = i / cell_ngp; // temporal index
            for j in 0..cell_ngp {
                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    sum += self.basis.cell_gauss_weights[k_x]
                        * self.basis.phis_cell_gps[(k_x, ix)]  // φ_i^x(x_k)
                        * self.basis.phis_cell_gps[(0, it)] // φ_i^t(x_{0})
                        * self.basis.phis_cell_gps[(k_x, j)]; // ψ_j(x_k)
                }
                self.sts_f0_mat[[i, j]] = sum;
            }
        }
    }
}
