use ndarray::{Array, Ix1, array};

pub fn get_lobatto_points_interval(points_num: usize) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
    let (gauss_points, gauss_weights) = match points_num {
        3 => {
            let points = array![0.0, 0.5, 1.0];
            let weights = array![1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
            (points, weights)
        }
        4 => {
            let sqrt5 = 5.0_f64.sqrt();
            let points = array![
                0.0,
                0.5 * (1.0 - 1.0 / sqrt5),
                0.5 * (1.0 + 1.0 / sqrt5),
                1.0
            ];
            let weights = array![1.0 / 12.0, 5.0 / 12.0, 5.0 / 12.0, 1.0 / 12.0];
            (points, weights)
        }
        5 => {
            let sqrt3_7 = (3.0_f64 / 7.0_f64).sqrt();
            let points = array![0.0, 0.5 * (1.0 - sqrt3_7), 0.5, 0.5 * (1.0 + sqrt3_7), 1.0];
            let weights = array![
                1.0 / 20.0,
                49.0 / 180.0,
                16.0 / 45.0,
                49.0 / 180.0,
                1.0 / 20.0
            ];
            (points, weights)
        }
        _ => panic!("Number of points not supported"),
    };
    (gauss_points, gauss_weights)
}
