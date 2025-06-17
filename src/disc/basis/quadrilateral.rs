use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array, s};
use ndarray_linalg::{Eigh, Inverse, Solve, UPLO};
use statrs::function::gamma::gamma;

pub struct QuadrilateralBasis {
    pub xi: Array1<f64>,
    pub eta: Array1<f64>,
    pub vandermonde: Array2<f64>,
    pub inv_vandermonde: Array2<f64>,
    pub dxi: Array2<f64>,
    pub deta: Array2<f64>,
    pub nodes_along_edges: Array2<usize>,
    pub quad_p: Array1<f64>,
    pub quad_w: Array1<f64>,
}

impl QuadrilateralBasis {
    pub fn new(n: usize) -> Self {}
    pub fn vandermonde2d(n: usize, xi: ArrayView1<f64>, eta: ArrayView1<f64>) -> Array2<f64> {}
}
