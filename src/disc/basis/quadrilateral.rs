use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array, s};
use ndarray_linalg::{Eigh, Inverse, Solve, UPLO};
use statrs::function::gamma::gamma;

use crate::disc::basis::Basis;
use crate::disc::basis::lagrange1d::LagrangeBasis1DLobatto;
use crate::disc::gauss_points::lobatto_points::get_lobatto_points_interval;

pub struct QuadrilateralBasis {
    pub r: Array1<f64>,
    pub s: Array1<f64>,
    pub vandermonde: Array2<f64>,
    pub inv_vandermonde: Array2<f64>,
    pub dr: Array2<f64>,
    pub ds: Array2<f64>,
    pub nodes_along_edges: Array2<usize>,
    pub quad_p: Array1<f64>,
    pub quad_w: Array1<f64>,
}

impl QuadrilateralBasis {
    pub fn new(n: usize) -> Self {
        let (r, s) = Self::nodes2d(n);
        let vandermonde = Self::vandermonde2d(n, r.view(), s.view());
        let inv_vandermonde = vandermonde.inv().unwrap();
        let (dr, ds) = Self::dmatrices_2d(n, r.view(), s.view(), vandermonde.view());
        let nodes_along_edges = Self::set_nodes_along_edges(n);
        let quad_p = Self::jacobi_gauss_lobatto(0.0, 0.0, n);
        let (_, quad_w_vec) = get_lobatto_points_interval(n + 1);
        let quad_w = Array1::from(quad_w_vec);
        Self {
            r,
            s,
            vandermonde,
            inv_vandermonde,
            dr,
            ds,
            nodes_along_edges,
            quad_p,
            quad_w,
        }
    }

    fn ortho_basis_ij(r: ArrayView1<f64>, s: ArrayView1<f64>, i: i32, j: i32) -> Array1<f64> {
        let p_i_r = Self::jacobi_polynomial(r, 0.0, 0.0, i);
        let p_j_s = Self::jacobi_polynomial(s, 0.0, 0.0, j);
        &p_i_r * &p_j_s
    }
    fn set_nodes_along_edges(n: usize) -> Array2<usize> {
        let nfp = n + 1;
        let mut nodes_along_edges = Array2::<usize>::zeros((4, nfp));

        // Edge 1 (bottom, s=-1)
        let fmask1: Vec<usize> = (0..nfp).collect();

        // Edge 2 (right, r=1)
        let fmask2: Vec<usize> = (0..nfp).map(|i| (i + 1) * nfp - 1).collect();

        // Edge 3 (top, s=1)
        let fmask3: Vec<usize> = (n * nfp..(n + 1) * nfp).rev().collect();

        // Edge 4 (left, r=-1)
        let fmask4: Vec<usize> = (0..nfp).map(|i| (n - i) * nfp).collect();

        nodes_along_edges
            .slice_mut(s![0, ..])
            .assign(&Array1::from_vec(fmask1));
        nodes_along_edges
            .slice_mut(s![1, ..])
            .assign(&Array1::from_vec(fmask2));
        nodes_along_edges
            .slice_mut(s![2, ..])
            .assign(&Array1::from_vec(fmask3));
        nodes_along_edges
            .slice_mut(s![3, ..])
            .assign(&Array1::from_vec(fmask4));

        nodes_along_edges
    }
}
impl Basis for QuadrilateralBasis {
    fn vandermonde2d(n: usize, r: ArrayView1<f64>, s: ArrayView1<f64>) -> Array2<f64> {
        let n_basis_1d = n + 1;
        let num_points = r.len();
        let n_basis_2d = n_basis_1d * n_basis_1d;
        let mut v = Array2::zeros((num_points, n_basis_2d));

        let mut sk = 0;
        for i in 0..n_basis_1d {
            for j in 0..n_basis_1d {
                v.column_mut(sk)
                    .assign(&Self::ortho_basis_ij(r, s, i as i32, j as i32));
                sk += 1;
            }
        }
        v
    }
    fn nodes2d(n: usize) -> (Array1<f64>, Array1<f64>) {
        let n_pts_1d = n + 1;
        let n_pts_2d = n_pts_1d * n_pts_1d;
        let mut r = Array1::<f64>::zeros(n_pts_2d);
        let mut s = Array1::<f64>::zeros(n_pts_2d);

        let zeta = Self::jacobi_gauss_lobatto(0.0, 0.0, n);

        let mut sk = 0;
        for i in 0..n_pts_1d {
            for j in 0..n_pts_1d {
                r[sk] = zeta[j];
                s[sk] = zeta[i];
                sk += 1;
            }
        }
        (r, s)
    }
    fn grad_vandermonde_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let n_basis_1d = n + 1;
        let num_points = r.len();
        let n_basis_2d = n_basis_1d * n_basis_1d;
        let mut vr = Array2::zeros((num_points, n_basis_2d));
        let mut vs = Array2::zeros((num_points, n_basis_2d));

        let mut sk = 0;
        for i in 0..n_basis_1d {
            for j in 0..n_basis_1d {
                let p_i_r = Self::jacobi_polynomial(r, 0.0, 0.0, i as i32);
                let dp_i_r = Self::grad_jacobi_polynomial(r, 0.0, 0.0, i as i32);

                let p_j_s = Self::jacobi_polynomial(s, 0.0, 0.0, j as i32);
                let dp_j_s = Self::grad_jacobi_polynomial(s, 0.0, 0.0, j as i32);

                vr.column_mut(sk).assign(&(&dp_i_r * &p_j_s));
                vs.column_mut(sk).assign(&(&p_i_r * &dp_j_s));
                sk += 1;
            }
        }
        (vr, vs)
    }
}
