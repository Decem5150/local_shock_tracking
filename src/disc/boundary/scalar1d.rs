use ndarray::Array1;

use crate::disc::boundary::BoundaryPosition;

#[derive(Clone, Debug)]
pub struct ConstantBoundary {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
    pub value: f64,
    pub position: BoundaryPosition,
}
#[derive(Clone, Debug)]
pub struct FunctionBoundary {
    pub inodes: Vec<usize>,
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

pub fn burgers_bnd_condition(x: f64, _t: f64) -> f64 {
    if x < 0.25 {
        2.0
    } else if x < 0.75 {
        4.0 - 8.0 * x
    } else {
        -2.0
    }
}

pub fn burgers_bnd_condition_2(x: f64, _t: f64) -> f64 {
    2.0 * (x + 1.0).powf(2.0) * (1.0 - heaviside(x))
}
fn heaviside(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 }
}
