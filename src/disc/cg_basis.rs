use ndarray::{Array1, Array2, ArrayView1};

pub mod triangle;

/// Trait for Continuous Galerkin basis functions on reference elements
/// These ensure C0 continuity across element boundaries
pub trait CGBasis2D {
    /// Evaluate Lagrange shape functions at given reference coordinates
    fn shape_functions(
        n: usize,
        l1: ArrayView1<f64>,
        l2: ArrayView1<f64>,
        l3: ArrayView1<f64>,
    ) -> Array2<f64>;

    /// Evaluate gradients of shape functions at given reference coordinates
    fn grad_shape_functions(
        n: usize,
        l1: ArrayView1<f64>,
        l2: ArrayView1<f64>,
        l3: ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>);

    /// Get the nodal coordinates on the reference element (optional, for interpolation)
    fn cubature_points(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>);

    /// Get the number of nodes for polynomial order n
    fn num_nodes(n: usize) -> usize {
        (n + 1) * (n + 2) / 2
    }
}
