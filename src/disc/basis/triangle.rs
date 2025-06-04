use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array, s};
use ndarray_linalg::{Eigh, Inverse, Solve, UPLO};
use statrs::function::gamma::gamma;

use crate::disc::gauss_points::lobatto_points::get_lobatto_points_interval;
pub struct TriangleBasis {
    pub r: Array1<f64>,
    pub s: Array1<f64>,
    pub vandermonde: Array2<f64>,
    pub inv_vandermonde: Array2<f64>,
    pub dr: Array2<f64>,
    pub ds: Array2<f64>,
    pub nodes_along_edges: Array2<usize>,
    pub quad_p: Array1<f64>,
    pub quad_w: Array1<f64>,
    pub cub_r: Array1<f64>,
    pub cub_s: Array1<f64>,
    pub cub_w: Array1<f64>,
}
impl TriangleBasis {
    pub fn new(n: usize) -> Self {
        let (x, y) = Self::nodes2d(n);
        let (r, s) = Self::xy_to_rs(x.view(), y.view());
        println!("r: {:?}", r);
        println!("s: {:?}", s);
        let vandermonde = Self::vandermonde2d(n, r.view(), s.view());
        println!("vandermonde: {:?}", vandermonde);
        let inv_vandermonde = vandermonde.inv().unwrap();
        let (dr, ds) = Self::dmatrices_2d(n, r.view(), s.view(), vandermonde.view());
        println!("dr: {:?}", dr);
        println!("ds: {:?}", ds);
        let l = Self::compute_l(vandermonde.view());
        println!("l: {:?}", l);
        let nodes_along_edges = Self::find_nodes_along_edges(n, r.view(), s.view());
        let quad_p = Self::jacobi_gauss_lobatto(0.0, 0.0, n);
        let (_, quad_w_vec) = get_lobatto_points_interval(n + 1);
        let quad_w = Array1::from_iter(quad_w_vec);
        let (cub_r, cub_s, cub_w) = Self::cubature_points(n);
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
            cub_r,
            cub_s,
            cub_w,
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
        let mut fmask3 = r
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
        fmask3.reverse();
        assert_eq!(fmask1.len(), nfp);
        assert_eq!(fmask2.len(), nfp);
        assert_eq!(fmask3.len(), nfp);
        /*
        // Verification code
        println!("=== Edge Node Verification (n={}) ===", n);

        // Verify Edge 0 (bottom edge: s = -1, should be ordered by increasing r)
        println!("Edge 0 (bottom, s=-1) nodes:");
        let mut prev_r = f64::NEG_INFINITY;
        for (idx, &node_id) in fmask1.iter().enumerate() {
            let r_val = r[node_id];
            let s_val = s[node_id];
            println!("  Node {}: r={:.6}, s={:.6}", node_id, r_val, s_val);

            // Verify node is on correct edge
            assert!(
                (s_val + 1.0).abs() < node_tol,
                "Node {} not on bottom edge",
                node_id
            );

            // Verify counterclockwise ordering (r should increase)
            if idx > 0 {
                assert!(
                    r_val >= prev_r - node_tol,
                    "Edge 0 nodes not in counterclockwise order"
                );
            }
            prev_r = r_val;
        }

        // Verify Edge 1 (diagonal edge: r + s = 0, should be ordered by decreasing r)
        println!("Edge 1 (diagonal, r+s=0) nodes:");
        let mut prev_r = f64::INFINITY;
        for (idx, &node_id) in fmask2.iter().enumerate() {
            let r_val = r[node_id];
            let s_val = s[node_id];
            println!("  Node {}: r={:.6}, s={:.6}", node_id, r_val, s_val);

            // Verify node is on correct edge
            assert!(
                (r_val + s_val).abs() < node_tol,
                "Node {} not on diagonal edge",
                node_id
            );

            // Verify counterclockwise ordering (r should decrease)
            if idx > 0 {
                assert!(
                    r_val <= prev_r + node_tol,
                    "Edge 1 nodes not in counterclockwise order"
                );
            }
            prev_r = r_val;
        }

        // Verify Edge 2 (left edge: r = -1, should be ordered by decreasing s)
        println!("Edge 2 (left, r=-1) nodes:");
        let mut prev_s = f64::INFINITY;
        for (idx, &node_id) in fmask3.iter().enumerate() {
            let r_val = r[node_id];
            let s_val = s[node_id];
            println!("  Node {}: r={:.6}, s={:.6}", node_id, r_val, s_val);

            // Verify node is on correct edge
            assert!(
                (r_val + 1.0).abs() < node_tol,
                "Node {} not on left edge",
                node_id
            );

            // Verify counterclockwise ordering (s should decrease)
            if idx > 0 {
                assert!(
                    s_val <= prev_s + node_tol,
                    "Edge 2 nodes not in counterclockwise order"
                );
            }
            prev_s = s_val;
        }
        */
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
    pub fn jacobi_gauss_lobatto(alpha: f64, beta: f64, n: usize) -> Array1<f64> {
        match n {
            0 => {
                panic!("n must be at least 1");
            }
            1 => {
                array![-1.0, 1.0]
            }
            n_order => {
                let (x_interior, _) =
                    Self::jacobi_gauss_quadrature(alpha + 1.0, beta + 1.0, n_order - 2);
                let mut x_lobatto = Array1::<f64>::zeros(n_order + 1);
                let mut w_lobatto = Array1::<f64>::zeros(n_order + 1);

                x_lobatto[0] = -1.0;
                x_lobatto[n_order] = 1.0;
                x_lobatto.slice_mut(s![1..n_order]).assign(&x_interior);
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
            * (1.0 - &b).powi(i)
    }
    fn warp_factor(n: usize, r: ArrayView1<f64>) -> Array1<f64> {
        let lglr = Self::jacobi_gauss_lobatto(0.0, 0.0, n);
        let req = Array1::linspace(-1.0, 1.0, n + 1);
        let veq = Self::vandermonde1d(n, req.view());
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
        let alpha = if n < 16 { alpopt[n] } else { 5.0 / 3.0 };
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
    pub fn vandermonde1d(n: usize, r: ArrayView1<f64>) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((r.len(), n + 1));
        for j in 0..n + 1 {
            v.column_mut(j)
                .assign(&Self::jacobi_polynomial(r, 0.0, 0.0, j as i32));
        }
        v
    }
    pub fn vandermonde2d(n: usize, r: ArrayView1<f64>, s: ArrayView1<f64>) -> Array2<f64> {
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
            dmode_ds = (0.5 * (1.0 - &b)).powi(id - 1) * &dmode_ds;
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
    pub fn grad_vandermonde_2d(
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
                let (v2dr_col, v2ds_col) =
                    Self::grad_simplex_2d(a.view(), b.view(), i as i32, j as i32);
                v2dr.column_mut(sk).assign(&v2dr_col);
                v2ds.column_mut(sk).assign(&v2ds_col);
                sk += 1;
            }
        }
        (v2dr, v2ds)
    }
    pub fn dmatrices_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
        v: ArrayView2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let (vr, vs) = Self::grad_vandermonde_2d(n, r, s);
        println!("vr: {:?}", vr);
        println!("vs: {:?}", vs);
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
    pub fn cubature_points(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let (r, s, weight) = match n {
            2 => {
                let r = array![0.16666666666667, 0.16666666666667, 0.66666666666667];
                let s = array![0.16666666666667, 0.66666666666667, 0.16666666666667];
                let weight = array![0.33333333333333, 0.33333333333333, 0.33333333333333];
                (r, s, weight)
            }
            3 => {
                let r = array![
                    0.33333333333333,
                    0.20000000000000,
                    0.20000000000000,
                    0.60000000000000
                ];
                let s = array![
                    0.33333333333333,
                    0.20000000000000,
                    0.60000000000000,
                    0.20000000000000
                ];
                let weight = array![
                    -0.56250000000000,
                    0.52083333333333,
                    0.52083333333333,
                    0.52083333333333
                ];
                (r, s, weight)
            }
            4 => {
                let r = array![
                    0.44594849091597,
                    0.44594849091597,
                    0.10810301816807,
                    0.09157621350977,
                    0.09157621350977,
                    0.81684757298046
                ];
                let s = array![
                    0.44594849091597,
                    0.10810301816807,
                    0.44594849091597,
                    0.09157621350977,
                    0.81684757298046,
                    0.09157621350977
                ];
                let weight = array![
                    0.22338158967801,
                    0.22338158967801,
                    0.22338158967801,
                    0.10995174365532,
                    0.10995174365532,
                    0.10995174365532
                ];
                (r, s, weight)
            }
            _ => {
                panic!("Number of points not supported");
            }
        };
        let r_new = 2.0 * &s - 1.0;
        let s_new = 2.0 * &r - 1.0;
        let weight_scaled = &weight * 2.0;
        (r_new, s_new, weight_scaled)
    }
    pub fn validate_dmatrices(&self, n: usize) -> bool {
        println!("=== Validating Differentiation Matrices ===");
        let tolerance = 1e-12;
        let mut all_tests_passed = true;

        // Test 1: Fundamental property - Dr * V = Vr and Ds * V = Vs
        println!("Test 1: Fundamental differentiation matrix property");
        let (vr, vs) = Self::grad_vandermonde_2d(n, self.r.view(), self.s.view());

        let dr_v = self.dr.dot(&self.vandermonde);
        let ds_v = self.ds.dot(&self.vandermonde);

        let dr_error = (&dr_v - &vr).iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let ds_error = (&ds_v - &vs).iter().map(|&x| x.abs()).fold(0.0, f64::max);

        println!("  Max |Dr*V - Vr|: {:.2e}", dr_error);
        println!("  Max |Ds*V - Vs|: {:.2e}", ds_error);

        if dr_error > tolerance || ds_error > tolerance {
            println!("  ❌ FAILED: Fundamental matrix property");
            all_tests_passed = false;
        } else {
            println!("  ✅ PASSED: Fundamental matrix property");
        }

        // Test 2: Differentiation of basis functions
        println!("\nTest 2: Basis function differentiation exactness");
        let np = (n + 1) * (n + 2) / 2;
        let mut basis_test_passed = true;

        for k in 0..np {
            // Create a function that is 1 at node k and 0 at all other nodes
            let mut nodal_function = Array1::zeros(self.r.len());
            nodal_function[k] = 1.0;

            // Apply differentiation matrices
            let dr_basis = self.dr.dot(&nodal_function);
            let ds_basis = self.ds.dot(&nodal_function);

            // The result should equal the k-th column of Vr and Vs divided by the inverse Vandermonde
            let expected_dr = vr.column(k);
            let expected_ds = vs.column(k);

            let dr_basis_error = (&dr_basis - &expected_dr)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);
            let ds_basis_error = (&ds_basis - &expected_ds)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);

            if dr_basis_error > tolerance || ds_basis_error > tolerance {
                println!(
                    "  ❌ FAILED: Basis function {} differentiation, dr_error: {:.2e}, ds_error: {:.2e}",
                    k, dr_basis_error, ds_basis_error
                );
                basis_test_passed = false;
                all_tests_passed = false;
            }
        }

        if basis_test_passed {
            println!(
                "  ✅ PASSED: All {} basis functions differentiated correctly",
                np
            );
        }

        // Test 3: Polynomial exactness using the actual Dubiner basis functions
        println!("\nTest 3: Dubiner basis polynomial differentiation");
        let (a, b) = Self::rs_to_ab(self.r.view(), self.s.view());
        let mut dubiner_test_passed = true;

        let mut sk = 0;
        for i in 0..=n {
            for j in 0..=(n - i) {
                // Create the Dubiner basis function
                let basis_func = Self::dubiner_basis(a.view(), b.view(), i as i32, j as i32);

                // Apply differentiation matrices
                let dr_dubiner = self.dr.dot(&basis_func);
                let ds_dubiner = self.ds.dot(&basis_func);

                // Expected derivatives from analytical computation
                let (expected_dr, expected_ds) =
                    Self::grad_simplex_2d(a.view(), b.view(), i as i32, j as i32);

                let dr_dubiner_error = (&dr_dubiner - &expected_dr)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);
                let ds_dubiner_error = (&ds_dubiner - &expected_ds)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);

                if dr_dubiner_error > tolerance || ds_dubiner_error > tolerance {
                    println!(
                        "  ❌ FAILED: Dubiner basis P_{{{},{}}} differentiation, dr_error: {:.2e}, ds_error: {:.2e}",
                        i, j, dr_dubiner_error, ds_dubiner_error
                    );
                    dubiner_test_passed = false;
                    all_tests_passed = false;
                }
                sk += 1;
            }
        }

        if dubiner_test_passed {
            println!("  ✅ PASSED: All Dubiner basis functions differentiated exactly");
        }

        // Test 4: Simple coordinate functions (these should be exactly representable)
        println!("\nTest 4: Coordinate function derivatives");

        // Test linear function: r (this should be in the polynomial space for n >= 1)
        if n >= 1 {
            // Project r onto the polynomial space
            let r_coeffs = self.inv_vandermonde.dot(&self.r);
            let r_projected = self.vandermonde.dot(&r_coeffs);

            // Check if r is well-represented
            let r_projection_error = (&r_projected - &self.r)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);

            if r_projection_error < tolerance {
                let dr_r = self.dr.dot(&r_projected);
                let ds_r = self.ds.dot(&r_projected);

                // For the projected r, ∂r/∂r should be approximately 1, ∂r/∂s should be approximately 0
                let expected_dr_r: Array1<f64> = Array1::ones(self.r.len());
                let expected_ds_r: Array1<f64> = Array1::zeros(self.r.len());

                let dr_r_error = (&dr_r - &expected_dr_r)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);
                let ds_r_error = (&ds_r - &expected_ds_r)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);

                println!(
                    "  r-coordinate: Max |dr(r) - 1|: {:.2e}, Max |ds(r) - 0|: {:.2e}",
                    dr_r_error, ds_r_error
                );

                if dr_r_error > 1e-10 || ds_r_error > 1e-10 {
                    println!(
                        "  ❌ FAILED: r-coordinate differentiation (within projection tolerance)"
                    );
                    all_tests_passed = false;
                } else {
                    println!("  ✅ PASSED: r-coordinate differentiation");
                }
            } else {
                println!(
                    "  ⚠️ WARNING: r-coordinate not exactly representable in polynomial space (error: {:.2e})",
                    r_projection_error
                );
            }
        }

        // Test linear function: s
        if n >= 1 {
            let s_coeffs = self.inv_vandermonde.dot(&self.s);
            let s_projected = self.vandermonde.dot(&s_coeffs);

            let s_projection_error = (&s_projected - &self.s)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);

            if s_projection_error < tolerance {
                let dr_s = self.dr.dot(&s_projected);
                let ds_s = self.ds.dot(&s_projected);

                let expected_dr_s: Array1<f64> = Array1::zeros(self.s.len());
                let expected_ds_s: Array1<f64> = Array1::ones(self.s.len());

                let dr_s_error = (&dr_s - &expected_dr_s)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);
                let ds_s_error = (&ds_s - &expected_ds_s)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);

                println!(
                    "  s-coordinate: Max |dr(s) - 0|: {:.2e}, Max |ds(s) - 1|: {:.2e}",
                    dr_s_error, ds_s_error
                );

                if dr_s_error > 1e-10 || ds_s_error > 1e-10 {
                    println!(
                        "  ❌ FAILED: s-coordinate differentiation (within projection tolerance)"
                    );
                    all_tests_passed = false;
                } else {
                    println!("  ✅ PASSED: s-coordinate differentiation");
                }
            } else {
                println!(
                    "  ⚠️ WARNING: s-coordinate not exactly representable in polynomial space (error: {:.2e})",
                    s_projection_error
                );
            }
        }

        // Test 5: Matrix properties
        println!("\nTest 5: Matrix properties");
        let np = self.r.len();

        if self.dr.shape() != [np, np] || self.ds.shape() != [np, np] {
            println!("  ❌ FAILED: Matrix dimensions incorrect");
            all_tests_passed = false;
        } else {
            println!("  ✅ PASSED: Matrix dimensions correct ({} x {})", np, np);
        }

        // Test 6: Conservation property (sum of each row should be derivative of constant = 0)
        println!("\nTest 6: Conservation property");
        let dr_row_sums: Array1<f64> = self.dr.sum_axis(Axis(1));
        let ds_row_sums: Array1<f64> = self.ds.sum_axis(Axis(1));

        let dr_conservation_error = dr_row_sums.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let ds_conservation_error = ds_row_sums.iter().map(|&x| x.abs()).fold(0.0, f64::max);

        println!("  Max |sum(Dr[i,:])|: {:.2e}", dr_conservation_error);
        println!("  Max |sum(Ds[i,:])|: {:.2e}", ds_conservation_error);

        if dr_conservation_error > tolerance || ds_conservation_error > tolerance {
            println!("  ❌ FAILED: Conservation property");
            all_tests_passed = false;
        } else {
            println!("  ✅ PASSED: Conservation property");
        }

        println!("\n=== Validation Summary ===");
        if all_tests_passed {
            println!("✅ ALL TESTS PASSED: Differentiation matrices are valid");
        } else {
            println!("❌ SOME TESTS FAILED: Differentiation matrices may have issues");
        }

        all_tests_passed
    }
}
