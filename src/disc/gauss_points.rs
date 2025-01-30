use ndarray::{Array, Ix1};

pub mod legendre_points;
pub mod lobatto_points;

pub struct GaussPoints1d {
    pub points: Array<f64, Ix1>,
    pub weights: Array<f64, Ix1>,
}
impl GaussPoints1d {
    pub fn new(points_num: usize) -> Self {
        let (points, weights) = legendre_points::get_legendre_points_interval(points_num);
        Self { points, weights }
    }
}
