use std::f64::consts::PI;

use ndarray::{Array, Array1, Array2, Ix1, Ix2, array, s};
use ndarray_linalg::{Eigh, Solve, UPLO};
use statrs::function::gamma::gamma;
struct TriangleBasis {
    pub cell_gauss_points: Vec<f64>,
    pub cell_gauss_weights: Vec<f64>,
    pub phis_cell_gps: Array<f64, Ix2>,  // (nbasis, ngp)
    pub dphis_cell_gps: Array<f64, Ix2>, // (nbasis, ngp)
}
fn rs_to_ab(r: f64, s: f64) -> (f64, f64) {
    let a = {
        if s != 1.0 {
            2.0 * (1.0 + r) / (1.0 - s) - 1.0
        } else {
            -1.0
        }
    };
    let b = s;
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
            let (x, _) = jacobi_gauss_quadrature(alpha + 1.0, beta + 1.0, n_order - 2);
            let mut x_lobatto = Array1::<f64>::zeros(n_order + 1);
            x_lobatto[0] = -1.0;
            x_lobatto[n_order] = 1.0;
            x_lobatto.slice_mut(s![1..n_order]).assign(&x);
            x_lobatto
        }
    }
}
fn jacobi_polynomial(x: Array1<f64>, alpha: f64, beta: f64, n: i32) -> Array1<f64> {
    match n {
        0 => {
            let p0 = (2.0_f64.powf(- alpha - beta - 1.0) * gamma(alpha + beta + 2.0) / (gamma(alpha + 1.0) * gamma (beta + 1.0))).sqrt();
            Array1::from_elem(x.len(), p0)
        }
        1 => 0.5 * jacobi_polynomial(x.clone(), alpha, beta, 0) * ((alpha + beta + 2.0) * &x + (alpha - beta)) * ((alpha + beta + 3.0) / ((alpha + 1.0) * (beta + 1.0))).sqrt(),
        _ => {
            let n = n as f64;
            let aold = 2.0 / (2.0 + alpha + beta) * ((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0)).sqrt();
            let h1 = 2.0 * (n - 1.0) + alpha + beta;
            let anew = 2.0 / (h1 + 2.0) * (n * (n + alpha + beta) * (n + alpha) * (n + beta) / (h1 + 1.0) / (h1 + 3.0)).sqrt();
            let bnew = - (alpha.powi(2) - beta.powi(2)) / h1 / (h1 + 2.0);
            let n = n as i32;
            let pn_1 = jacobi_polynomial(x.clone(), alpha, beta, n - 1);
            let pn_2 = jacobi_polynomial(x.clone(), alpha, beta, n - 2);

            (- aold * pn_2 + (&x - bnew) * pn_1) / anew
        }
    }
}
fn grad_jacobi_polynomial(r: Array1<f64>, alpha: f64, beta: f64, n: i32) -> Array1<f64> {
    let mut dp = Array1::<f64>::zeros(r.len());
    match n {
        0 => {
            dp.fill(0.0);
        }
        _ => {
            let pn = jacobi_polynomial(r, alpha + 1.0, beta + 1.0, n - 1);
            let n = n as f64;
            dp.assign(&(n * (n + alpha + beta + 1.0).sqrt() * pn));
        }
    }
    dp
}
fn dubiner_basis(a: Array1<f64>, b: Array1<f64>, i: i32, j: i32) -> Array1<f64> {
    2.0_f64.sqrt()
        * jacobi_polynomial(a, 0.0, 0.0, i)
        * jacobi_polynomial(b, 2.0 * i as f64 + 1.0, 0.0, j)
}
fn vandermonde_matrix(n: usize, r: Array1<f64>) -> Array2<f64> {
    let mut v = Array2::<f64>::zeros((r.len(), n + 1));
    for j in 0..=n {
        v.column_mut(j).assign(&jacobi_polynomial(r.clone(), 0.0, 0.0, j as i32));
    }
    v
}
fn warp_factor(n: usize, r: Array1<f64>) -> Array1<f64> {
    let lglr = jacobi_gauss_lobatto(0.0, 0.0, n);
    let req = Array1::linspace(-1.0, 1.0, n + 1);
    let veq = vandermonde_matrix(n, req);
    let nr = r.len();
    let mut pmat = Array2::<f64>::zeros((n + 1, nr));
    for i in 0..n + 1 {
        pmat.row_mut(i).assign(&jacobi_polynomial(r.clone(), 0.0, 0.0, i as i32));
    }
    let mut lmat = Array2::zeros((veq.shape()[1], pmat.shape()[1]));
    for i in 0..pmat.shape()[1] {
        let col = pmat.column(i);
        let sol = veq.t().solve(&col).unwrap();
        lmat.column_mut(i).assign(&sol);
    }
    let warp = lmat.t().dot(&(lglr - &r));
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
    let mut sk: usize = 1;
    for i in 0..n + 1 {
        for j in 0..n + 1 - i {
            l1[sk] = (i - 1) as f64 / n as f64;
            l3[sk] = (j - 1) as f64 / n as f64;
            l2[sk] = 1.0 - l1[sk] - l3[sk];
            sk += 1;
        }
    }
    let mut x = -&l2 + &l3;
    let mut y = (-&l2 - &l3 + 2.0 * &l1) / 3.0_f64.sqrt();

    let blend1 = 4.0 * &l2 * &l3;
    let blend2 = 4.0 * &l1 * &l3;
    let blend3 = 4.0 * &l1 * &l2;

    let warpf1 = warp_factor(n, &l3 - &l2);
    let warpf2 = warp_factor(n, &l1 - &l3);
    let warpf3 = warp_factor(n, &l2 - &l1);

    let warp1 = blend1 * warpf1 * (1.0 + (alpha * l1).powi(2));
    let warp2 = blend2 * warpf2 * (1.0 + (alpha * l2).powi(2));
    let warp3 = blend3 * warpf3 * (1.0 + (alpha * l3).powi(2));

    x = x + &warp1 + (2.0 * PI / 3.0).cos() * &warp2 + (4.0 * PI / 3.0).cos() * &warp3;
    y = y + &warp1 + (2.0 * PI / 3.0).sin() * &warp2 + (4.0 * PI / 3.0).sin() * &warp3;
    (x, y)
}
fn xy_to_rs(x: Array1<f64>, y: Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let l1 = (3.0_f64.sqrt() * &y + 1.0) / 3.0;
    let l2 = (-3.0 * &x - 3.0_f64.sqrt() * &y + 2.0) / 6.0;
    let l3 = (3.0 * &x - 3.0_f64.sqrt() * &y + 2.0) / 6.0;
    let r = -&l2 + &l3 - &l1;
    let s = -&l2 - &l3 + &l1;
    (r, s)
}
fn vandermonde2d(n: usize, r: Array1<f64>, s: Array1<f64>) -> Array2<f64> {
    let mut v = Array2::<f64>::zeros((r.len(), (n + 1) * (n + 2) / 2));
    let mut sk: usize = 1;
    for i in 0..n + 1 {
        for j in 0..n + 1 - i {
            for k in 0..r.len() {
                let (a, b) = rs_to_ab(r[k], s[k]);
                v[(k, sk)] = dubiner_basis(a, b, i as i32, j as i32);
            }
            sk += 1;
        }
    }
    v
}
fn grad_simplex_2d(a: f64, b: f64, id: i32, jd: i32) -> (Array2<f64>, Array2<f64>) {
    let fa = jacobi_polynomial(a, 0.0, 0.0, id);
    let 
}
