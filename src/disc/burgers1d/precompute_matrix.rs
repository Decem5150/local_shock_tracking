use super::Disc1dBurgers;
use ndarray::Array2;
use ndarray_linalg::Inverse;

impl<'a> Disc1dBurgers<'a> {
    pub fn compute_m_mat_ref(&self) -> (Array2<f64>, Array2<f64>) {
        let cell_ngp = self.solver_param.polynomial_order + 1;
        let mut ss_m_mat = Array2::zeros((cell_ngp, cell_ngp));
        // compute mass matrix of the two space polynomials
        for i in 0..cell_ngp {
            for j in 0..cell_ngp {
                if i == j {
                    ss_m_mat[[i, j]] = self.space_basis.weights[i];
                } else {
                    ss_m_mat[[i, j]] = 0.0;
                }
            }
        }
        // self.ss_im_mat = self.ss_m_mat.inv().unwrap();
        let mut stst_m_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        // compute mass matrix of the two space-time polynomials
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                let ix = i % cell_ngp; // spatial index
                let it = i / cell_ngp; // temporal index
                let jx = j % cell_ngp; // spatial index
                let jt = j / cell_ngp; // temporal index

                // Only compute when indices match (otherwise result is 0)
                if ix == jx && it == jt {
                    stst_m_mat[[i, j]] =
                        self.space_basis.weights[ix] * self.space_basis.weights[it];
                } else {
                    stst_m_mat[[i, j]] = 0.0;
                }
            }
        }
        // self.stst_im_mat = self.stst_m_mat.inv().unwrap();
        (ss_m_mat, stst_m_mat)
    }
    pub fn compute_kxi_mat_ref(&self) -> Array2<f64> {
        let cell_ngp = self.solver_param.polynomial_order + 1;

        /*
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
        */
        let mut kxi_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
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
                    let phi = if k_x == ix { 1.0 } else { 0.0 };
                    spatial +=
                        self.space_basis.weights[k_x] * self.space_basis.dxi[(k_x, jx)] * phi;
                }

                // Temporal integral
                let phi = if it == jt { 1.0 } else { 0.0 };
                let temporal = self.space_basis.weights[it] * phi;

                kxi_mat[[i, j]] = spatial * temporal;
            }
        }
        kxi_mat
    }
    pub fn compute_ik1_mat_ref(&self) -> Array2<f64> {
        let cell_ngp = self.solver_param.polynomial_order + 1;
        let mut f1_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        for i in 0..cell_ngp * cell_ngp {
            let ix = i % cell_ngp; // Spatial component
            let it = i / cell_ngp; // Temporal component
            for j in 0..cell_ngp * cell_ngp {
                let jx = j % cell_ngp;
                let jt = j / cell_ngp;

                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    let phi_kx_ix = if k_x == ix { 1.0 } else { 0.0 };
                    let phi_kx_jx = if k_x == jx { 1.0 } else { 0.0 };
                    let phi_end_it = if cell_ngp - 1 == it { 1.0 } else { 0.0 };
                    let phi_end_jt = if cell_ngp - 1 == jt { 1.0 } else { 0.0 };
                    sum += self.space_basis.weights[k_x]
                        * phi_kx_ix  // φ_i^x(x_k)
                        * phi_kx_jx  // φ_j^x(x_k)
                        * phi_end_it // φ_i^t(x_{cell_ngp-1})
                        * phi_end_jt; // φ_j^t(x_{cell_ngp-1})
                }
                f1_mat[[i, j]] = sum;
            }
        }
        println!("f1_mat_ref: {:?}", f1_mat);
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
                let phi_ix_jx = if ix == jx { 1.0 } else { 0.0 };
                let spatial = self.space_basis.weights[ix] * phi_ix_jx;
                // Temporal integral
                let mut temporal = 0.0;
                for k_t in 0..cell_ngp {
                    let phi = if k_t == jt { 1.0 } else { 0.0 };
                    temporal +=
                        self.space_basis.weights[k_t] * self.space_basis.dxi[(k_t, it)] * phi;
                }
                t_mat[[i, j]] = spatial * temporal;
            }
        }
        println!("t_mat_ref: {:?}", t_mat);
        let mut k1_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp * cell_ngp));
        // Subtract temporal stiffness matrix from f1_matrix
        for i in 0..cell_ngp * cell_ngp {
            for j in 0..cell_ngp * cell_ngp {
                k1_mat[[i, j]] = f1_mat[[i, j]] - t_mat[[i, j]];
            }
        }
        println!("k1_mat_ref: {:?}", k1_mat);
        k1_mat.inv().unwrap()
    }
    pub fn compute_f0_mat_ref(&self) -> Array2<f64> {
        let cell_ngp = self.solver_param.polynomial_order + 1;
        let mut f0_mat = Array2::zeros((cell_ngp * cell_ngp, cell_ngp));
        for i in 0..cell_ngp * cell_ngp {
            let ix = i % cell_ngp; // spatial index
            let it = i / cell_ngp; // temporal index
            for j in 0..cell_ngp {
                let mut sum = 0.0;
                for k_x in 0..cell_ngp {
                    let phi_k_x_ix = if k_x == ix { 1.0 } else { 0.0 };
                    let phi_0_it = if it == 0 { 1.0 } else { 0.0 };
                    let phi_k_x_j = if k_x == j { 1.0 } else { 0.0 };
                    sum += self.space_basis.weights[k_x]
                        * phi_k_x_ix  // φ_i^x(x_k)
                        * phi_0_it // φ_i^t(x_{0})
                        * phi_k_x_j; // ψ_j(x_k)
                }
                f0_mat[[i, j]] = sum;
            }
        }
        f0_mat
    }
}
