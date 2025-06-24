use ndarray::{Array2, ArrayView2};
use ndarray_linalg::Inverse;

use crate::disc::basis::quadrilateral::QuadrilateralBasis;

pub trait ADER1DScalar {
    fn physical_flux(&self, u: f64) -> f64;
}
pub trait ADER1DMatrices {
    fn compute_space_mass_mat(basis: &QuadrilateralBasis) -> Array2<f64> {
        let n_pts_2d = basis.xi.len();
        let n_pts_1d = (n_pts_2d as f64).sqrt() as usize;
        let mut m_1d = Array2::zeros((n_pts_1d, n_pts_1d));
        for i in 0..n_pts_1d {
            m_1d[(i, i)] = basis.quad_w[i];
        }
        m_1d
    }
    fn compute_space_time_mass_mat(basis: &QuadrilateralBasis) -> Array2<f64> {
        basis.inv_vandermonde.t().dot(&basis.inv_vandermonde)
    }
    fn compute_kxi_mat(basis: &QuadrilateralBasis, m_mat: ArrayView2<f64>) -> Array2<f64> {
        m_mat.dot(&basis.dxi)
    }
    fn compute_ik1_mat(basis: &QuadrilateralBasis, m_mat: ArrayView2<f64>) -> Array2<f64> {
        let n_pts_2d = basis.xi.len();
        let n_pts_1d = (n_pts_2d as f64).sqrt() as usize;
        let n = n_pts_1d - 1;
        let mut m_1d = Array2::zeros((n_pts_1d, n_pts_1d));
        for i in 0..n_pts_1d {
            m_1d[(i, i)] = basis.quad_w[i];
        }

        let mut f1_mat = Array2::zeros((n_pts_2d, n_pts_2d));
        for i_xi in 0..n_pts_1d {
            for j_xi in 0..n_pts_1d {
                let i_dof = n * n_pts_1d + i_xi;
                let j_dof = n * n_pts_1d + j_xi;
                f1_mat[(i_dof, j_dof)] = m_1d[(i_xi, j_xi)];
            }
        }

        let t_mat = m_mat.dot(&basis.deta);

        let k1_mat = &f1_mat - &t_mat;
        k1_mat.inv().unwrap()
    }
    fn compute_f0_mat(basis: &QuadrilateralBasis) -> Array2<f64> {
        let n_pts_2d = basis.xi.len();
        let n_pts_1d = n_pts_2d.isqrt();

        let mut m_1d = Array2::zeros((n_pts_1d, n_pts_1d));
        for i in 0..n_pts_1d {
            m_1d[(i, i)] = basis.quad_w[i];
        }

        let mut f0_mat = Array2::zeros((n_pts_2d, n_pts_1d));
        for i in 0..n_pts_2d {
            let ix = i % n_pts_1d;
            let it = i / n_pts_1d;
            for j in 0..n_pts_1d {
                let kronecker_delta = if it == 0 { 1.0 } else { 0.0 };
                f0_mat[(i, j)] = kronecker_delta * m_1d[(ix, j)];
            }
        }
        f0_mat
    }
}
