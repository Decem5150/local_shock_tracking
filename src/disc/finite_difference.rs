use ndarray::{Array1, Array2};

use crate::disc::dg_basis::triangle::TriangleBasis;

/// Finite difference module for computing derivatives of functions
/// that were previously computed using automatic differentiation.
///
/// This module provides a generic framework for computing derivatives
/// using finite differences with adaptive step sizes.

pub struct FiniteDifference {
    epsilon: f64,
    epsilon_sqrt: f64,
}

impl Default for FiniteDifference {
    fn default() -> Self {
        let epsilon = f64::EPSILON;
        Self {
            epsilon,
            epsilon_sqrt: epsilon.sqrt(),
        }
    }
}

impl FiniteDifference {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the optimal step size for finite differences
    /// based on the magnitude of the variable
    pub fn compute_step_size(&self, x: f64) -> f64 {
        let scale = x.abs().max(1.0);
        scale * self.epsilon_sqrt
    }

    /// Compute derivative of a scalar function with respect to a scalar variable
    /// using central differences
    pub fn scalar_derivative<F>(&self, f: F, x: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let h = self.compute_step_size(x);
        let f_plus = f(x + h);
        let f_minus = f(x - h);
        (f_plus - f_minus) / (2.0 * h)
    }

    /// Compute derivative of a scalar function with respect to a vector variable
    /// using central differences
    pub fn vector_derivative<F>(&self, f: F, x: &[f64], grad: &mut [f64])
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x.len();
        let mut x_perturbed = x.to_vec();

        for i in 0..n {
            let h = self.compute_step_size(x[i]);
            let x_orig = x[i];

            // Forward perturbation
            x_perturbed[i] = x_orig + h;
            let f_plus = f(&x_perturbed);

            // Backward perturbation
            x_perturbed[i] = x_orig - h;
            let f_minus = f(&x_perturbed);

            // Restore original value
            x_perturbed[i] = x_orig;

            // Central difference
            grad[i] = (f_plus - f_minus) / (2.0 * h);
        }
    }

    /// Compute Jacobian of a vector function with respect to a vector variable
    /// using central differences
    pub fn jacobian<F>(&self, f: F, x: &[f64], jac: &mut Array2<f64>)
    where
        F: Fn(&[f64], &mut [f64]),
    {
        let n = x.len();
        let m = jac.nrows();
        let mut x_perturbed = x.to_vec();
        let mut f_plus = vec![0.0; m];
        let mut f_minus = vec![0.0; m];

        for j in 0..n {
            let h = self.compute_step_size(x[j]);
            let x_orig = x[j];

            // Forward perturbation
            x_perturbed[j] = x_orig + h;
            f(&x_perturbed, &mut f_plus);

            // Backward perturbation
            x_perturbed[j] = x_orig - h;
            f(&x_perturbed, &mut f_minus);

            // Restore original value
            x_perturbed[j] = x_orig;

            // Central difference for column j
            for i in 0..m {
                jac[(i, j)] = (f_plus[i] - f_minus[i]) / (2.0 * h);
            }
        }
    }
}
pub fn compute_distortion_derivatives<F>(
    fd: &FiniteDifference,
    distortion_fn: F,
    x: &[f64],
    y: &[f64],
    d_x: &mut [f64],
    d_y: &mut [f64],
    basis: &TriangleBasis,
) -> f64
where
    F: Fn(&[f64], &[f64], &TriangleBasis) -> f64,
{
    let base_value = distortion_fn(x, y, basis);

    let mut x_perturbed = x.to_vec();
    let mut y_perturbed = y.to_vec();

    // Compute derivatives with respect to x coordinates
    for i in 0..x.len() {
        let h = fd.compute_step_size(x[i]);
        let x_orig = x[i];

        x_perturbed[i] = x_orig + h;
        let f_plus = distortion_fn(&x_perturbed, y, basis);

        x_perturbed[i] = x_orig - h;
        let f_minus = distortion_fn(&x_perturbed, y, basis);

        x_perturbed[i] = x_orig;

        d_x[i] = (f_plus - f_minus) / (2.0 * h);
    }

    // Compute derivatives with respect to y coordinates
    for i in 0..y.len() {
        let h = fd.compute_step_size(y[i]);
        let y_orig = y[i];

        y_perturbed[i] = y_orig + h;
        let f_plus = distortion_fn(x, &y_perturbed, basis);

        y_perturbed[i] = y_orig - h;
        let f_minus = distortion_fn(x, &y_perturbed, basis);

        y_perturbed[i] = y_orig;

        d_y[i] = (f_plus - f_minus) / (2.0 * h);
    }

    base_value
}
/// Example implementation for the volume integral derivative computation
/// This replaces the autodiff-generated dvolume function
#[allow(clippy::too_many_arguments)]
pub fn compute_volume_derivatives<F>(
    fd: &FiniteDifference,
    volume_fn: F,
    sol: &[f64],
    x: &[f64],
    y: &[f64],
    d_sol: &mut [f64],
    d_x: &mut [f64],
    d_y: &mut [f64],
) -> f64
where
    F: Fn(&[f64], &[f64], &[f64]) -> f64,
{
    // Compute the base value
    let base_value = volume_fn(sol, x, y);

    // Compute derivatives with respect to solution
    let mut sol_perturbed = sol.to_vec();
    for i in 0..sol.len() {
        let h = fd.compute_step_size(sol[i]);
        let sol_orig = sol[i];

        sol_perturbed[i] = sol_orig + h;
        let f_plus = volume_fn(&sol_perturbed, x, y);

        sol_perturbed[i] = sol_orig - h;
        let f_minus = volume_fn(&sol_perturbed, x, y);

        sol_perturbed[i] = sol_orig;

        d_sol[i] = (f_plus - f_minus) / (2.0 * h);
    }

    // Compute derivatives with respect to x coordinates
    let mut x_perturbed = x.to_vec();
    for i in 0..x.len() {
        let h = fd.compute_step_size(x[i]);
        let x_orig = x[i];

        x_perturbed[i] = x_orig + h;
        let f_plus = volume_fn(sol, &x_perturbed, y);

        x_perturbed[i] = x_orig - h;
        let f_minus = volume_fn(sol, &x_perturbed, y);

        x_perturbed[i] = x_orig;

        d_x[i] = (f_plus - f_minus) / (2.0 * h);
    }

    // Compute derivatives with respect to y coordinates
    let mut y_perturbed = y.to_vec();
    for i in 0..y.len() {
        let h = fd.compute_step_size(y[i]);
        let y_orig = y[i];

        y_perturbed[i] = y_orig + h;
        let f_plus = volume_fn(sol, x, &y_perturbed);

        y_perturbed[i] = y_orig - h;
        let f_minus = volume_fn(sol, x, &y_perturbed);

        y_perturbed[i] = y_orig;

        d_y[i] = (f_plus - f_minus) / (2.0 * h);
    }

    base_value
}

/// Example implementation for numerical flux derivative computation
/// This replaces the autodiff-generated dnum_flux function
#[allow(clippy::too_many_arguments)]
pub fn compute_numerical_flux_derivatives<F>(
    fd: &FiniteDifference,
    flux_fn: F,
    ul: f64,
    ur: f64,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
) -> (f64, f64, f64, f64, f64, f64, f64)
where
    F: Fn(f64, f64, f64, f64, f64, f64) -> f64,
{
    // Compute the base flux value
    let flux = flux_fn(ul, ur, x0, x1, y0, y1);

    // Compute derivatives using finite differences
    let dflux_dul = fd.scalar_derivative(|u| flux_fn(u, ur, x0, x1, y0, y1), ul);
    let dflux_dur = fd.scalar_derivative(|u| flux_fn(ul, u, x0, x1, y0, y1), ur);
    let dflux_dx0 = fd.scalar_derivative(|x| flux_fn(ul, ur, x, x1, y0, y1), x0);
    let dflux_dx1 = fd.scalar_derivative(|x| flux_fn(ul, ur, x0, x, y0, y1), x1);
    let dflux_dy0 = fd.scalar_derivative(|y| flux_fn(ul, ur, x0, x1, y, y1), y0);
    let dflux_dy1 = fd.scalar_derivative(|y| flux_fn(ul, ur, x0, x1, y0, y), y1);

    (
        flux, dflux_dul, dflux_dur, dflux_dx0, dflux_dx1, dflux_dy0, dflux_dy1,
    )
}

/// Example implementation for flux scaling derivative computation
/// This replaces the autodiff-generated dscaling function
#[allow(clippy::too_many_arguments)]
pub fn compute_flux_scaling_derivatives<F>(
    fd: &FiniteDifference,
    scaling_fn: F,
    xi: f64,
    eta: f64,
    ref_normal: [f64; 2],
    x: &[f64],
    y: &[f64],
    d_x: &mut [f64],
    d_y: &mut [f64],
) -> f64
where
    F: Fn(f64, f64, [f64; 2], &[f64], &[f64]) -> f64,
{
    // Compute the base scaling value
    let scaling = scaling_fn(xi, eta, ref_normal, x, y);

    // Compute derivatives with respect to x coordinates
    let mut x_perturbed = x.to_vec();
    for i in 0..x.len() {
        let h = fd.compute_step_size(x[i]);
        let x_orig = x[i];

        x_perturbed[i] = x_orig + h;
        let f_plus = scaling_fn(xi, eta, ref_normal, &x_perturbed, y);

        x_perturbed[i] = x_orig - h;
        let f_minus = scaling_fn(xi, eta, ref_normal, &x_perturbed, y);

        x_perturbed[i] = x_orig;

        d_x[i] = (f_plus - f_minus) / (2.0 * h);
    }

    // Compute derivatives with respect to y coordinates
    let mut y_perturbed = y.to_vec();
    for i in 0..y.len() {
        let h = fd.compute_step_size(y[i]);
        let y_orig = y[i];

        y_perturbed[i] = y_orig + h;
        let f_plus = scaling_fn(xi, eta, ref_normal, x, &y_perturbed);

        y_perturbed[i] = y_orig - h;
        let f_minus = scaling_fn(xi, eta, ref_normal, x, &y_perturbed);

        y_perturbed[i] = y_orig;

        d_y[i] = (f_plus - f_minus) / (2.0 * h);
    }

    scaling
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_derivative() {
        let fd = FiniteDifference::new();

        // Test derivative of x^2 at x=2.0 (should be 4.0)
        let f = |x: f64| x * x;
        let df_dx = fd.scalar_derivative(f, 2.0);
        assert!((df_dx - 4.0).abs() < 1e-6);

        // Test derivative of sin(x) at x=0.0 (should be 1.0)
        let f = |x: f64| x.sin();
        let df_dx = fd.scalar_derivative(f, 0.0);
        assert!((df_dx - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_derivative() {
        let fd = FiniteDifference::new();

        // Test gradient of f(x,y) = x^2 + y^2 at (1,2)
        // Gradient should be (2x, 2y) = (2, 4)
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = vec![1.0, 2.0];
        let mut grad = vec![0.0; 2];

        fd.vector_derivative(f, &x, &mut grad);

        assert!((grad[0] - 2.0).abs() < 1e-6);
        assert!((grad[1] - 4.0).abs() < 1e-6);
    }
}
