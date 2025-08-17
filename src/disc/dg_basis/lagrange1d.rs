use crate::disc::{dg_basis::Basis1D, gauss_points::lobatto_points::get_lobatto_points_interval};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::Inverse;

pub struct LobattoBasis {
    pub n: usize,
    pub xi: Array1<f64>,
    pub weights: Array1<f64>,
    pub vandermonde: Array2<f64>,
    pub inv_vandermonde: Array2<f64>,
    pub dxi: Array2<f64>,
}

impl LobattoBasis {
    pub fn new(n: usize) -> LobattoBasis {
        let dofs = n + 1;
        let (xi_vec, weights_vec) = get_lobatto_points_interval(dofs);
        let xi = Array1::from(xi_vec).mapv(|v| (v + 1.0) / 2.0);
        let weights = Array1::from(weights_vec).mapv(|v| v / 2.0);
        let xi_map = xi.mapv(|x| 2. * x - 1.);
        let vandermonde = Self::vandermonde1d(n, xi_map.view());
        let inv_vandermonde = vandermonde.inv().unwrap();
        let dxi = Self::dmatrix_1d(n, xi.view(), vandermonde.view());
        println!("dxi: {dxi}");
        LobattoBasis {
            n,
            xi,
            weights,
            vandermonde,
            inv_vandermonde,
            dxi,
        }
    }

    /// # Arguments
    /// * `i` - Index of the basis function to evaluate
    /// * `x` - Point at which to evaluate the basis function
    fn grad_vandermonde_1d(n: usize, xi: ArrayView1<f64>) -> Array2<f64> {
        let n_basis_1d = n + 1;
        let num_points = xi.len();
        let mut vxi = Array2::zeros((num_points, n_basis_1d));

        let xi_map = xi.mapv(|x| 2. * x - 1.);

        for i in 0..n_basis_1d {
            let mut dp_i_xi = Self::grad_jacobi_polynomial(xi_map.view(), 0.0, 0.0, i as i32);
            dp_i_xi.mapv_inplace(|val| val * 2.0);
            vxi.column_mut(i).assign(&dp_i_xi);
        }
        vxi
    }
    fn dmatrix_1d(n: usize, xi: ArrayView1<f64>, v: ArrayView2<f64>) -> Array2<f64> {
        let vxi = Self::grad_vandermonde_1d(n, xi);
        let inv_v = v.inv().unwrap();
        vxi.dot(&inv_v)
    }
    /*
    pub fn evaluate_basis_at(&self, i: usize, x: f64) -> f64 {
        let n = self.cell_gauss_points.len();

        // Check if x is (almost) equal to one of the interpolation points
        for j in 0..n {
            if (x - self.cell_gauss_points[j]).abs() < 1e-14 {
                // Lagrange basis property: L_i(x_j) = δ_ij (Kronecker delta)
                return if j == i { 1.0 } else { 0.0 };
            }
        }

        // Otherwise, compute using the Lagrange formula
        let mut result = 1.0;
        let x_i = self.cell_gauss_points[i];

        for j in 0..n {
            if j != i {
                let x_j = self.cell_gauss_points[j];
                result *= (x - x_j) / (x_i - x_j);
            }
        }

        result
    }

    pub fn evaluate_all_basis_at(&self, x: f64) -> Vec<f64> {
        let n = self.cell_gauss_points.len();
        let mut values = vec![0.0; n];

        for i in 0..n {
            values[i] = self.evaluate_basis_at(i, x);
        }

        values
    }

    /// Evaluates a function at a point using the basis representation
    ///
    /// # Arguments
    /// * `coefficients` - Coefficients of the function in the basis
    /// * `x` - Point at which to evaluate the function
    ///
    /// # Returns
    /// Value of the function at point x
    pub fn evaluate_function_at(&self, coefficients: &[f64], x: f64) -> f64 {
        let n = self.cell_gauss_points.len();
        assert_eq!(
            coefficients.len(),
            n,
            "Coefficient vector length must match basis size"
        );

        let mut result = 0.0;
        for i in 0..n {
            result += coefficients[i] * self.evaluate_basis_at(i, x);
        }

        result
    }
    */
}
impl Basis1D for LobattoBasis {}
