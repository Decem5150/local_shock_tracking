use ndarray::{Array1, ArrayView1};

pub fn smoothed_upwind(
    ul: ArrayView1<f64>,
    ur: ArrayView1<f64>,
    normal: [f64; 2],
    advection_speed: f64,
) -> Array1<f64> {
    let beta = Array1::from_vec(vec![advection_speed, 1.0]);
    let normal_array = Array1::from(normal.to_vec());
    let result = 0.5
        * (beta.dot(&normal_array) * (&ul + &ur)
            + (beta.dot(&normal_array) * (100.0 * beta.dot(&normal_array)).tanh()) * (&ul - &ur));
    result
}
