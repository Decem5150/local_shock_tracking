use ndarray::{array, Array, Ix1};

pub fn get_lobatto_points_interval(points_num: usize) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
    let (gauss_points, gauss_weights) = match points_num {
        3 => {
            let points = array![- 1.0, 0.0, 1.0];
            let weights = array![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0];
            (points, weights)
        }
        4 => {
            let points = array![- 1.0, - 1.0 / 5.0_f64.sqrt(), 1.0 / 5.0_f64.sqrt(), 1.0];
            let weights = array![1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0];
            (points, weights)
        }
        5 => {
            let points = array![- 1.0, - (3.0 / 7.0_f64).sqrt(), 0.0, (3.0 / 7.0_f64).sqrt(), 1.0];
            let weights = array![1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0];
            (points, weights)
        }
        _ => panic!("Number of points not supported"),
    };
    (gauss_points, gauss_weights)
}