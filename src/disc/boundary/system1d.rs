use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct ConstantBoundary {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
    pub value: Array1<f64>,
}
#[derive(Clone, Debug)]
pub struct FunctionBoundary {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
    pub func: fn(f64, f64) -> Array1<f64>,
}
#[derive(Clone, Debug)]
pub struct OpenBoundary {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
}
