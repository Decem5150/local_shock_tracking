use crate::disc::gauss_points::{
    legendre_points::get_legendre_points_interval, lobatto_points::get_lobatto_points_interval,
};
use ndarray::{Array, Ix1, Ix2};

pub struct LagrangeBasis1D {
    pub cell_gauss_points: Array<f64, Ix1>,
    pub cell_gauss_weights: Array<f64, Ix1>,
    pub phis_cell_gps: Array<f64, Ix2>,
    pub dphis_cell_gps: Array<f64, Ix2>,
    pub phis_bnd_gps: Array<f64, Ix2>,
    pub dphis_bnd_gps: Array<f64, Ix2>,
}

pub struct LagrangeBasis1DLobatto {
    pub cell_gauss_points: Vec<f64>,
    pub cell_gauss_weights: Vec<f64>,
    pub phis_cell_gps: Array<f64, Ix2>,  // (ngp, nbasis)
    pub dphis_cell_gps: Array<f64, Ix2>, // (ngp, nbasis)
}

impl LagrangeBasis1D {
    pub fn new(cell_gp_num: usize) -> LagrangeBasis1D {
        let dofs = cell_gp_num;
        let (cell_gauss_points, cell_gauss_weights) = get_legendre_points_interval(dofs);
        let mut phis_cell_gps = Array::zeros((dofs, dofs));
        let mut dphis_cell_gps = Array::zeros((dofs, dofs));
        let mut phis_bnd_gps = Array::zeros((2, dofs));
        let mut dphis_bnd_gps = Array::zeros((2, dofs));
        // compute the basis functions at the gauss points
        for i in 0..dofs {
            for j in 0..dofs {
                phis_cell_gps[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
        // compute the basis functions at the boundary gauss points (-1 and 1).
        let boundary_points = [-1.0, 1.0]; // left and right boundary points
        for i in 0..2 {
            // loop over boundary points
            for j in 0..dofs {
                // loop over the basis functions
                let mut product = 1.0;
                for m in 0..dofs {
                    // loop over the gauss points
                    if m != j {
                        // skip the current gauss point
                        product *= (boundary_points[i] - cell_gauss_points[m])
                            / (cell_gauss_points[j] - cell_gauss_points[m]);
                    }
                }
                phis_bnd_gps[(i, j)] = product;
            }
        }
        // compute the derivatives of the basis functions at the gauss points.
        for j in 0..dofs {
            // loop over the gauss points
            for i in 0..dofs {
                // loop over the basis functions
                let mut derivative = 0.0;
                for k in 0..dofs {
                    // loop over the gauss points
                    if k != j {
                        // skip the current gauss point
                        let mut product = 1.0;
                        for m in 0..dofs {
                            // loop over the gauss points
                            if m != j && m != k {
                                // skip the current gauss point and the gauss point we are computing the derivative at
                                product *= (cell_gauss_points[i] - cell_gauss_points[m])
                                    / (cell_gauss_points[j] - cell_gauss_points[m]);
                            }
                        }
                        derivative += product / (cell_gauss_points[j] - cell_gauss_points[k]);
                    }
                }
                dphis_cell_gps[(i, j)] = derivative;
            }
        }
        // compute the derivatives of the basis functions at the boundary gauss points.
        for i in 0..2 {
            // loop over boundary points
            for j in 0..dofs {
                // loop over the basis functions
                let mut derivative = 0.0;
                for k in 0..dofs {
                    // loop over the gauss points
                    if k != j {
                        // skip the current gauss point
                        let mut product = 1.0;
                        for m in 0..dofs {
                            // loop over the gauss points
                            if m != j && m != k {
                                // skip the current gauss point and the gauss point we are computing the derivative at
                                product *= (boundary_points[i] - cell_gauss_points[m])
                                    / (cell_gauss_points[j] - cell_gauss_points[m]);
                            }
                        }
                        derivative += product / (cell_gauss_points[j] - cell_gauss_points[k]);
                    }
                }
                dphis_bnd_gps[(i, j)] = derivative;
            }
        }
        LagrangeBasis1D {
            cell_gauss_points,
            cell_gauss_weights,
            phis_cell_gps,
            dphis_cell_gps,
            phis_bnd_gps,
            dphis_bnd_gps,
        }
    }
}

impl LagrangeBasis1DLobatto {
    pub fn new(cell_gp_num: usize) -> LagrangeBasis1DLobatto {
        let dofs = cell_gp_num;
        let (cell_gauss_points, cell_gauss_weights) = get_lobatto_points_interval(dofs);
        let mut phis_cell_gps = Array::zeros((dofs, dofs));
        let mut dphis_cell_gps = Array::zeros((dofs, dofs));
        // Compute the basis functions at the gauss points
        for j in 0..dofs {
            for i in 0..dofs {
                phis_cell_gps[(j, i)] = if i == j { 1.0 } else { 0.0 };
            }
        }
        // Compute the derivatives of the basis functions at the gauss points
        for j in 0..dofs {
            // loop over basis functions
            for i in 0..dofs {
                // loop over gauss points
                let mut sum = 0.0;
                for l in 0..dofs {
                    if l != j {
                        let mut product = 1.0;
                        for m in 0..dofs {
                            if m != j && m != l {
                                product *= (cell_gauss_points[i] - cell_gauss_points[m])
                                    / (cell_gauss_points[j] - cell_gauss_points[m]);
                            }
                        }
                        sum += product / (cell_gauss_points[j] - cell_gauss_points[l]);
                    }
                }

                dphis_cell_gps[(j, i)] = sum;
            }
        }
        LagrangeBasis1DLobatto {
            cell_gauss_points,
            cell_gauss_weights,
            phis_cell_gps,
            dphis_cell_gps,
        }
    }

    /// Evaluates the i-th basis function at point x
    ///
    /// # Arguments
    /// * `i` - Index of the basis function to evaluate
    /// * `x` - Point at which to evaluate the basis function
    ///
    /// # Returns
    /// Value of the i-th basis function at point x
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

    /// Evaluates all basis functions at a given point
    ///
    /// # Arguments
    /// * `x` - Point at which to evaluate the basis functions
    ///
    /// # Returns
    /// Vector containing values of all basis functions at point x
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
}
