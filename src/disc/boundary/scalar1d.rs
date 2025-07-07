use ndarray::Array1;

#[derive(Clone)]
pub struct ConstantBoundary {
    pub iedges: Vec<usize>,
    pub value: f64,
}
#[derive(Clone)]
pub struct FunctionBoundary {
    pub iedges: Vec<usize>,
    pub func: fn(f64, f64) -> f64,
}
#[derive(Clone)]
pub struct PolynomialBoundary {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
    pub nodal_coeffs: Array1<f64>,
    pub normal: [f64; 2],
}
