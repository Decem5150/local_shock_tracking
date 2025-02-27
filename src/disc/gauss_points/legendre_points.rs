use ndarray::{Array, Ix1, array};

pub fn get_legendre_points_interval(points_num: usize) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
    let (gauss_points, gauss_weights) = match points_num {
        3 => {
            let points = array![-0.7745966692414834, 0.0, 0.7745966692414834];
            let weights = array![0.5555555555555556, 0.8888888888888888, 0.5555555555555556];
            (points, weights)
        }
        4 => {
            let points = array![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526
            ];
            let weights = array![
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538
            ];
            (points, weights)
        }
        5 => {
            let points = array![
                -0.9061798459386640,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.9061798459386640
            ];
            let weights = array![
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891
            ];
            (points, weights)
        }
        6 => {
            let points = array![
                -0.9324695142031521,
                -0.6612093864662645,
                -0.2386191860831969,
                0.2386191860831969,
                0.6612093864662645,
                0.9324695142031521
            ];
            let weights = array![
                0.1713244923791704,
                0.3607615730481386,
                0.4679139345726910,
                0.4679139345726910,
                0.3607615730481386,
                0.1713244923791704
            ];
            (points, weights)
        }
        _ => panic!("Number of points not supported"),
    };
    (gauss_points, gauss_weights)
}
