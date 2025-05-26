use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array, s};
use ndarray_linalg::{Eigh, Inverse, Solve, UPLO};
use statrs::function::gamma::gamma;
pub struct TriangleBasis {
    pub r: Array1<f64>,
    pub s: Array1<f64>,
    pub l: Array2<f64>,
    pub vandermonde: Array2<f64>,
    pub dr: Array2<f64>,
    pub ds: Array2<f64>,
    pub nodes_along_edges: Array2<usize>,
}
impl TriangleBasis {
    pub fn new(n: usize) -> Self {
        let (x, y) = Self::nodes2d(n);
        let (r, s) = Self::xy_to_rs(x.view(), y.view());
        let vandermonde = Self::vandermonde2d(n, r.view(), s.view());
        let l = Self::compute_l(vandermonde.view());
        let (dr, ds) = Self::dmatrices_2d(n, r.view(), s.view(), vandermonde.view());
        let nodes_along_edges = Self::find_nodes_along_edges(n, r.view(), s.view());

        Self {
            r,
            s,
            l,
            vandermonde,
            dr,
            ds,
            nodes_along_edges,
        }
    }
    fn compute_l(v: ArrayView2<f64>) -> Array2<f64> {
        let mut l = Array2::<f64>::zeros((v.shape()[0], v.shape()[0]));
        let v_t_inv = v.t().inv().unwrap();
        for i in 0..l.shape()[1] {
            let p = v.row(i);
            l.column_mut(i).assign(&v_t_inv.dot(&p));
        }
        l
    }
    fn find_nodes_along_edges(n: usize, r: ArrayView1<f64>, s: ArrayView1<f64>) -> Array2<usize> {
        let node_tol = 1.0e-10;
        let nfp = n + 1;
        let fmask1 = r
            .iter()
            .enumerate()
            .filter_map(|(i, _)| {
                if (s[i] + 1.0).abs() < node_tol {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();
        let fmask2 = r
            .iter()
            .enumerate()
            .filter_map(|(i, &r_val)| {
                if (r_val + s[i]).abs() < node_tol {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();
        let fmask3 = r
            .iter()
            .enumerate()
            .filter_map(|(i, &r_val)| {
                if (r_val + 1.0).abs() < node_tol {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();
        assert_eq!(fmask1.len(), nfp);
        assert_eq!(fmask2.len(), nfp);
        assert_eq!(fmask3.len(), nfp);
        let mut nodes_along_edges = Array2::<usize>::zeros((3, nfp));
        nodes_along_edges
            .slice_mut(s![0, ..])
            .assign(&Array1::from_iter(fmask1));
        nodes_along_edges
            .slice_mut(s![1, ..])
            .assign(&Array1::from_iter(fmask2));
        nodes_along_edges
            .slice_mut(s![2, ..])
            .assign(&Array1::from_iter(fmask3));
        nodes_along_edges
    }
    fn rs_to_ab(r: ArrayView1<f64>, s: ArrayView1<f64>) -> (Array1<f64>, Array1<f64>) {
        let a = r
            .iter()
            .zip(s.iter())
            .map(|(&r_val, &s_val)| {
                if s_val != 1.0 {
                    2.0 * (1.0 + r_val) / (1.0 - s_val) - 1.0
                } else {
                    -1.0
                }
            })
            .collect::<Array1<f64>>();
        let b = s.to_owned();
        (a, b)
    }
    fn jacobi_gauss_quadrature(alpha: f64, beta: f64, n: usize) -> (Array1<f64>, Array1<f64>) {
        match n {
            0 => {
                let x0 = (alpha - beta) / (alpha + beta + 2.0);
                (array![x0], array![2.0])
            }
            n_order => {
                let dim = n_order + 1;
                let mut j = Array2::<f64>::zeros((dim, dim));
                let h1 = Array1::from_iter((0..dim).map(|k| 2.0 * k as f64 + alpha + beta));
                let mut main_diag = Array1::from_iter((0..dim).map(|k| {
                    let denominator = h1[k] * (h1[k] + 2.0);
                    -0.5 * (alpha.powi(2) - beta.powi(2)) / denominator
                }));
                if (alpha + beta).abs() < 10.0 * f64::EPSILON {
                    main_diag[0] = 0.0;
                }
                j.diag_mut().assign(&main_diag);

                // off-diagonal
                for k in 0..(dim - 1) {
                    let l = k as f64 + 1.0;
                    let numerator = l * (l + alpha + beta) * (l + alpha) * (l + beta);
                    let denominator = (h1[k] + 1.0) * (h1[k] + 3.0);
                    let off_diag_val = (2.0 / (h1[k] + 2.0)) * (numerator / denominator).sqrt();
                    j[[k, k + 1]] = off_diag_val;
                    j[[k + 1, k]] = off_diag_val;
                }
                let (eigenvalues, eigenvectors) = j
                    .eigh(UPLO::Lower)
                    .expect("Eigenvalue decomposition failed");
                let mu_0_factor =
                    2.0_f64.powf(alpha + beta + 1.0) * gamma(alpha + 1.0) * gamma(beta + 1.0)
                        / gamma(alpha + beta + 2.0);
                let weights = eigenvectors
                    .row(0)
                    .mapv(|v_comp| v_comp.powi(2) * mu_0_factor);

                (eigenvalues, weights)
            }
        }
    }
    fn jacobi_gauss_lobatto(alpha: f64, beta: f64, n: usize) -> Array1<f64> {
        match n {
            0 => {
                panic!("n must be at least 1");
            }
            1 => {
                array![0.0, 1.0]
            }
            n_order => {
                let (x, _) = Self::jacobi_gauss_quadrature(alpha + 1.0, beta + 1.0, n_order - 2);
                let mut x_lobatto = Array1::<f64>::zeros(n_order + 1);
                x_lobatto[0] = -1.0;
                x_lobatto[n_order] = 1.0;
                x_lobatto.slice_mut(s![1..n_order]).assign(&x);
                x_lobatto
            }
        }
    }
    fn jacobi_polynomial(x: ArrayView1<f64>, alpha: f64, beta: f64, n: i32) -> Array1<f64> {
        match n {
            0 => {
                let p0 = (2.0_f64.powf(-alpha - beta - 1.0) * gamma(alpha + beta + 2.0)
                    / (gamma(alpha + 1.0) * gamma(beta + 1.0)))
                .sqrt();
                Array1::from_elem(x.len(), p0)
            }
            1 => {
                0.5 * Self::jacobi_polynomial(x.clone(), alpha, beta, 0)
                    * ((alpha + beta + 2.0) * &x + (alpha - beta))
                    * ((alpha + beta + 3.0) / ((alpha + 1.0) * (beta + 1.0))).sqrt()
            }
            _ => {
                let n = n as f64;
                let aold = 2.0 / (2.0 + alpha + beta)
                    * ((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0)).sqrt();
                let h1 = 2.0 * (n - 1.0) + alpha + beta;
                let anew = 2.0 / (h1 + 2.0)
                    * (n * (n + alpha + beta) * (n + alpha) * (n + beta) / (h1 + 1.0) / (h1 + 3.0))
                        .sqrt();
                let bnew = -(alpha.powi(2) - beta.powi(2)) / h1 / (h1 + 2.0);
                let n = n as i32;
                let pn_1 = Self::jacobi_polynomial(x.clone(), alpha, beta, n - 1);
                let pn_2 = Self::jacobi_polynomial(x.clone(), alpha, beta, n - 2);

                (-aold * pn_2 + (&x - bnew) * pn_1) / anew
            }
        }
    }
    fn grad_jacobi_polynomial(r: ArrayView1<f64>, alpha: f64, beta: f64, n: i32) -> Array1<f64> {
        let mut dp = Array1::<f64>::zeros(r.len());
        match n {
            0 => {
                dp.fill(0.0);
            }
            _ => {
                let pn = Self::jacobi_polynomial(r, alpha + 1.0, beta + 1.0, n - 1);
                let n = n as f64;
                dp.assign(&(n * (n + alpha + beta + 1.0).sqrt() * pn));
            }
        }
        dp
    }
    fn dubiner_basis(a: ArrayView1<f64>, b: ArrayView1<f64>, i: i32, j: i32) -> Array1<f64> {
        2.0_f64.sqrt()
            * Self::jacobi_polynomial(a, 0.0, 0.0, i)
            * Self::jacobi_polynomial(b, 2.0 * i as f64 + 1.0, 0.0, j)
    }
    fn vandermonde_matrix(n: usize, r: ArrayView1<f64>) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((r.len(), n + 1));
        for j in 0..=n {
            v.column_mut(j)
                .assign(&Self::jacobi_polynomial(r, 0.0, 0.0, j as i32));
        }
        v
    }
    fn warp_factor(n: usize, r: ArrayView1<f64>) -> Array1<f64> {
        let lglr = Self::jacobi_gauss_lobatto(0.0, 0.0, n);
        let req = Array1::linspace(-1.0, 1.0, n + 1);
        let veq = Self::vandermonde_matrix(n, req.view());
        let nr = r.len();
        let mut pmat = Array2::<f64>::zeros((n + 1, nr));
        for i in 0..n + 1 {
            pmat.row_mut(i)
                .assign(&Self::jacobi_polynomial(r.clone(), 0.0, 0.0, i as i32));
        }
        let mut lmat = Array2::zeros((veq.shape()[1], pmat.shape()[1]));
        for i in 0..pmat.shape()[1] {
            let col = pmat.column(i);
            let sol = veq.t().solve(&col).unwrap();
            lmat.column_mut(i).assign(&sol);
        }
        assert_eq!(lglr.len(), req.len());
        let warp = lmat.t().dot(&(&lglr - &req));
        let zerof = r.mapv(|x| if x.abs() < 1.0 - 1.0e-10 { 1.0 } else { 0.0 });
        let sf = 1.0 - (&zerof * &r).mapv(|x| x.powi(2));
        let warp = &warp / &sf + &warp * &(zerof - 1.0);
        warp
    }
    fn nodes2d(n: usize) -> (Array1<f64>, Array1<f64>) {
        let alpopt = [
            0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832, 1.3648, 1.4773, 1.4959,
            1.5743, 1.5770, 1.6223, 1.6258,
        ];
        let alpha = alpopt[n];
        let np = (n + 1) * (n + 2) / 2;

        let mut l1 = Array1::<f64>::zeros(np);
        let mut l2 = Array1::<f64>::zeros(np);
        let mut l3 = Array1::<f64>::zeros(np);
        let mut sk: usize = 0;
        for i in 0..n + 1 {
            for j in 0..n + 1 - i {
                l1[sk] = i as f64 / n as f64;
                l3[sk] = j as f64 / n as f64;
                l2[sk] = 1.0 - l1[sk] - l3[sk];
                sk += 1;
            }
        }
        let mut x = -&l2 + &l3;
        let mut y = (-&l2 - &l3 + 2.0 * &l1) / 3.0_f64.sqrt();

        let blend1 = 4.0 * &l2 * &l3;
        let blend2 = 4.0 * &l1 * &l3;
        let blend3 = 4.0 * &l1 * &l2;

        let warpf1 = Self::warp_factor(n, (&l3 - &l2).view());
        let warpf2 = Self::warp_factor(n, (&l1 - &l3).view());
        let warpf3 = Self::warp_factor(n, (&l2 - &l1).view());

        let warp1 = blend1 * warpf1 * (1.0 + (alpha * l1).powi(2));
        let warp2 = blend2 * warpf2 * (1.0 + (alpha * l2).powi(2));
        let warp3 = blend3 * warpf3 * (1.0 + (alpha * l3).powi(2));

        x = x + &warp1 + (2.0 * PI / 3.0).cos() * &warp2 + (4.0 * PI / 3.0).cos() * &warp3;
        y = y + (2.0 * PI / 3.0).sin() * &warp2 + (4.0 * PI / 3.0).sin() * &warp3;
        (x, y)
    }
    fn xy_to_rs(x: ArrayView1<f64>, y: ArrayView1<f64>) -> (Array1<f64>, Array1<f64>) {
        let l1 = (3.0_f64.sqrt() * &y + 1.0) / 3.0;
        let l2 = (-3.0 * &x - 3.0_f64.sqrt() * &y + 2.0) / 6.0;
        let l3 = (3.0 * &x - 3.0_f64.sqrt() * &y + 2.0) / 6.0;
        let r = -&l2 + &l3 - &l1;
        let s = -&l2 - &l3 + &l1;
        (r, s)
    }
    fn vandermonde1d(n: usize, r: ArrayView1<f64>) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((r.len(), n + 1));
        for j in 0..n + 1 {
            v.column_mut(j)
                .assign(&Self::jacobi_polynomial(r, 0.0, 0.0, j as i32));
        }
        v
    }
    fn vandermonde2d(n: usize, r: ArrayView1<f64>, s: ArrayView1<f64>) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((r.len(), (n + 1) * (n + 2) / 2));
        let (a, b) = Self::rs_to_ab(r, s);
        let mut sk: usize = 0;
        for i in 0..n + 1 {
            for j in 0..n + 1 - i {
                v.column_mut(sk).assign(&Self::dubiner_basis(
                    a.view(),
                    b.view(),
                    i as i32,
                    j as i32,
                ));
                sk += 1;
            }
        }
        v
    }
    fn grad_simplex_2d(
        a: ArrayView1<f64>,
        b: ArrayView1<f64>,
        id: i32,
        jd: i32,
    ) -> (Array1<f64>, Array1<f64>) {
        let fa = Self::jacobi_polynomial(a, 0.0, 0.0, id);
        let gb = Self::jacobi_polynomial(b, 2.0 * id as f64 + 1.0, 0.0, jd);
        let dfa = Self::grad_jacobi_polynomial(a, 0.0, 0.0, id);
        let dgb = Self::grad_jacobi_polynomial(b, 2.0 * id as f64 + 1.0, 0.0, jd);
        let mut dmode_dr = &dfa * &gb;
        if id > 0 {
            dmode_dr = (0.5 * (1.0 - &b)).powi(id - 1) * &dmode_dr;
        }
        let mut dmode_ds = &dfa * (&gb * (0.5 * (1.0 + &a)));
        if id > 0 {
            dmode_ds = (0.5 * (1.0 + &a)).powi(id - 1) * &dmode_ds;
        }
        let mut tmp = &dgb * ((0.5 * (1.0 - &b)).powi(id));
        if id > 0 {
            tmp = tmp - 0.5 * id as f64 * &gb * ((0.5 * (1.0 - &b)).powi(id - 1));
        }
        dmode_ds = dmode_ds + &fa * &tmp;
        dmode_dr = dmode_dr * 2.0_f64.powf(id as f64 + 0.5);
        dmode_ds = dmode_ds * 2.0_f64.powf(id as f64 + 0.5);
        (dmode_dr, dmode_ds)
    }
    fn grad_vandermonde_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut v2dr = Array2::<f64>::zeros((r.len(), (n + 1) * (n + 2) / 2));
        let mut v2ds = Array2::<f64>::zeros((r.len(), (n + 1) * (n + 2) / 2));
        let (a, b) = Self::rs_to_ab(r, s);
        let mut sk: usize = 0;
        for i in 0..n + 1 {
            for j in 0..n + 1 - i {
                let (v2dr_row, v2ds_row) =
                    Self::grad_simplex_2d(a.view(), b.view(), i as i32, j as i32);
                v2dr.row_mut(sk).assign(&v2dr_row);
                v2ds.row_mut(sk).assign(&v2ds_row);
                sk += 1;
            }
        }
        (v2dr, v2ds)
    }
    fn dmatrices_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
        v: ArrayView2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let (vr, vs) = Self::grad_vandermonde_2d(n, r, s);
        let v_inv = v.inv().unwrap();
        let dr = vr.dot(&v_inv);
        let ds = vs.dot(&v_inv);
        (dr, ds)
    }
    fn lift_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
        nodes_along_edges: ArrayView2<usize>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let np = (n + 1) * (n + 2) / 2;
        let nfp = n + 1;
        let nfaces = 3;
        let mut emat = Array2::<f64>::zeros((np, nfaces * nfp));
        // face 1
        let facer = r.select(
            Axis(0),
            nodes_along_edges.slice(s![0, ..]).as_slice().unwrap(),
        );
        let v1d = Self::vandermonde1d(n, facer.view());
        let mass_edge1 = v1d.t().dot(&v1d).inv().unwrap();
        for &i in nodes_along_edges.slice(s![0, ..]) {
            let mut emat_row = emat.row_mut(i);
            emat_row.slice_mut(s![0..nfp]).assign(&mass_edge1.row(i));
        }
        // face 2
        let facer = r.select(
            Axis(0),
            nodes_along_edges.slice(s![1, ..]).as_slice().unwrap(),
        );
        let v1d = Self::vandermonde1d(n, facer.view());
        let mass_edge2 = v1d.t().dot(&v1d).inv().unwrap();
        for &i in nodes_along_edges.slice(s![1, ..]) {
            let mut emat_row = emat.row_mut(i);
            emat_row
                .slice_mut(s![nfp..2 * nfp])
                .assign(&mass_edge2.row(i));
        }
        // face 3
        let faces = s.select(
            Axis(0),
            nodes_along_edges.slice(s![2, ..]).as_slice().unwrap(),
        );
        let v1d = Self::vandermonde1d(n, faces.view());
        let mass_edge3 = v1d.t().dot(&v1d).inv().unwrap();
        for &i in nodes_along_edges.slice(s![2, ..]) {
            let mut emat_row = emat.row_mut(i);
            emat_row
                .slice_mut(s![2 * nfp..3 * nfp])
                .assign(&mass_edge3.row(i));
        }
        let lift = v.t().dot(&(v.dot(&emat)));
        lift
    }
}
