use ndarray::Array1;

use crate::disc::boundary::BoundaryPosition;

#[derive(Clone, Debug)]
pub struct ConstantBoundary {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
    pub value: f64,
    pub position: BoundaryPosition,
}
#[derive(Clone)]
pub struct FunctionBoundary {
    pub iedges: Vec<usize>,
    pub func: fn(f64, f64) -> f64,
    pub position: BoundaryPosition,
}
#[derive(Clone, Debug)]
pub struct PolynomialBoundary {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
    pub nodal_coeffs: Array1<f64>,
    pub position: BoundaryPosition,
}
#[derive(Clone, Debug)]
pub struct OpenBoundary {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
    pub position: BoundaryPosition,
}
