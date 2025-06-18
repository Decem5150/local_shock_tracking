use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array, s};
use ndarray_linalg::{Eigh, Inverse, UPLO};
use statrs::function::gamma::gamma;

pub mod lagrange1d;
pub mod quadrilateral;
pub mod triangle;

pub trait Basis {
    fn vandermonde1d(n: usize, r: ArrayView1<f64>) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((r.len(), n + 1));
        for j in 0..n + 1 {
            v.column_mut(j)
                .assign(&Self::jacobi_polynomial(r, 0.0, 0.0, j as i32));
        }
        v
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
                array![-1.0, 1.0]
            }
            n_order => {
                let (x_interior, _) =
                    Self::jacobi_gauss_quadrature(alpha + 1.0, beta + 1.0, n_order - 2);
                let mut x_lobatto = Array1::<f64>::zeros(n_order + 1);

                x_lobatto[0] = -1.0;
                x_lobatto[n_order] = 1.0;
                x_lobatto.slice_mut(s![1..n_order]).assign(&x_interior);
                x_lobatto
            }
        }
    }
    #[allow(non_snake_case)]
    fn calculate_Ak(k_order: f64, alpha: f64, beta: f64) -> f64 {
        if k_order < 1.0 {
            // A_0 is not typically defined via this recurrence.
            // For A_1, k_order would be 1.0.
            // This formula is generally for k >= 1.
            // The MATLAB code's `aold` (initial) is A_1.
            if k_order == 1.0 {
                // Explicitly A_1
                return 2.0 / (2.0 + alpha + beta)
                    * ((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0)).sqrt();
            }
            // Fallback or error for k_order < 1 if not A_1, though P_0 is const.
            // For safety, though P_0 is const and P_1 has its own formula in the match.
            panic!("A_k calculation is typically for k >= 1");
        }
        let h1_for_Ak = 2.0 * (k_order - 1.0) + alpha + beta;
        let ak_val = 2.0 / (h1_for_Ak + 2.0)
            * (k_order * (k_order + alpha + beta) * (k_order + alpha) * (k_order + beta)
                / (h1_for_Ak + 1.0)
                / (h1_for_Ak + 3.0))
                .sqrt();
        if ak_val.is_nan() {
            // Add a check for NaN, which can happen if terms under sqrt are negative
            panic!(
                "NaN encountered in A_k calculation for k={}, alpha={}, beta={}",
                k_order, alpha, beta
            );
        }
        ak_val
    }
    #[allow(non_snake_case)]
    fn jacobi_polynomial(x: ArrayView1<f64>, alpha: f64, beta: f64, n: i32) -> Array1<f64> {
        match n {
            0 => {
                let gamma0 = 2.0_f64.powf(alpha + beta + 1.0) / (alpha + beta + 1.0)
                    * gamma(alpha + 1.0)
                    * gamma(beta + 1.0)
                    / gamma(alpha + beta + 1.0);
                let p0 = 1.0 / gamma0.sqrt();
                Array1::from_elem(x.len(), p0)
            }
            1 => {
                let gamma0 = 2.0_f64.powf(alpha + beta + 1.0) / (alpha + beta + 1.0)
                    * gamma(alpha + 1.0)
                    * gamma(beta + 1.0)
                    / gamma(alpha + beta + 1.0);
                let gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0;

                ((alpha + beta + 2.0) * &x * 0.5 + (alpha - beta) * 0.5) / gamma1.sqrt()
            }
            _ => {
                // For n >= 2
                let n_f = n as f64;

                // Coefficient A_n
                let an = Self::calculate_Ak(n_f, alpha, beta);

                // Coefficient B_n
                let h1_for_Bn = 2.0 * (n_f - 1.0) + alpha + beta;
                let bn = -(alpha.powi(2) - beta.powi(2)) / h1_for_Bn / (h1_for_Bn + 2.0);

                // Coefficient A_{n-1}
                // This is `aold` in the MATLAB loop for the current iteration `i` (where `i+1 = n`)
                let a_n_minus_1 = Self::calculate_Ak(n_f - 1.0, alpha, beta);

                let pn_1 = Self::jacobi_polynomial(x.view(), alpha, beta, n - 1);
                let pn_2 = Self::jacobi_polynomial(x.view(), alpha, beta, n - 2);

                // P_n = ( (x - B_n)P_{n-1} - A_{n-1}P_{n-2} ) / A_n
                if an.abs() < 1e-16 {
                    // Avoid division by zero or very small A_n
                    panic!(
                        "A_n is too small in jacobi_polynomial for n={}, alpha={}, beta={}",
                        n, alpha, beta
                    );
                }
                ((&x - bn) * pn_1 - a_n_minus_1 * pn_2) / an
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
                dp.assign(&((n * (n + alpha + beta + 1.0)).sqrt() * pn));
            }
        }
        dp
    }
    fn dmatrices_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
        v: ArrayView2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let (vr, vs) = Self::grad_vandermonde_2d(n, r, s);
        println!("vr: {:?}", vr);
        println!("vs: {:?}", vs);
        // We want to compute differentiation matrices Dr and Ds such that:
        // Dr * V = Vr  =>  V.t() * Dr.t() = Vr.t()
        // Ds * V = Vs  =>  V.t() * Ds.t() = Vs.t()

        let inv_v = v.inv().unwrap();
        let dr = vr.dot(&inv_v);
        let ds = vs.dot(&inv_v);
        (dr, ds)
    }
    fn vandermonde2d(n: usize, r: ArrayView1<f64>, s: ArrayView1<f64>) -> Array2<f64>;
    fn grad_vandermonde_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>);
    fn nodes2d(n: usize) -> (Array1<f64>, Array1<f64>);
}
