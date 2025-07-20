pub fn get_lobatto_points_interval(points_num: usize) -> (Vec<f64>, Vec<f64>) {
    let (gauss_points, gauss_weights) = match points_num {
        1 => {
            let points = vec![0.0];
            let weights = vec![2.0];
            (points, weights)
        }
        2 => {
            let points = vec![-1.0, 1.0];
            let weights = vec![1.0, 1.0];
            (points, weights)
        }
        3 => {
            let points = vec![-1.0, 0.0, 1.0];
            let weights = vec![1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0];
            (points, weights)
        }
        4 => {
            let sqrt5 = 5.0_f64.sqrt();
            let points = vec![-1.0, -1.0 / 5.0 * sqrt5, 1.0 / 5.0 * sqrt5, 1.0];
            let weights = vec![1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0];
            (points, weights)
        }
        5 => {
            let sqrt21 = (21.0_f64).sqrt();
            let points = vec![-1.0, -1.0 / 7.0 * sqrt21, 0.0, 1.0 / 7.0 * sqrt21, 1.0];
            let weights = vec![
                1.0 / 10.0,
                49.0 / 90.0,
                32.0 / 45.0,
                49.0 / 90.0,
                1.0 / 10.0,
            ];
            (points, weights)
        }
        _ => panic!("Number of points not supported"),
    };
    (gauss_points, gauss_weights)
}
