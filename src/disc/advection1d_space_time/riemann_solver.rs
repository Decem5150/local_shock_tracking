use ndarray::Array1;

pub fn smoothed_upwind(ul: f64, ur: f64, normal: [f64; 2], advection_speed: f64) -> f64 {
    let beta = Array1::from_vec(vec![advection_speed, 1.0]);
    let normal_array = Array1::from(normal.to_vec());
    let result = 0.5
        * (beta.dot(&normal_array) * (ul + ur)
            + (beta.dot(&normal_array) * (100.0 * beta.dot(&normal_array)).tanh()) * (ul - ur));
    result
}
pub fn dflux_dul(normal: [f64; 2], advection_speed: f64) -> f64 {
    let beta = Array1::from_vec(vec![advection_speed, 1.0]);
    let normal_array = Array1::from(normal.to_vec());
    let result = 0.5
        * (beta.dot(&normal_array)
            + (beta.dot(&normal_array) * (100.0 * beta.dot(&normal_array)).tanh()));
    result
}
pub fn dflux_dur(normal: [f64; 2], advection_speed: f64) -> f64 {
    let beta = Array1::from_vec(vec![advection_speed, 1.0]);
    let normal_array = Array1::from(normal.to_vec());
    let result = 0.5
        * (beta.dot(&normal_array)
            - (beta.dot(&normal_array) * (100.0 * beta.dot(&normal_array)).tanh()));
    result
}
pub fn dflux_dnormal(ul: f64, ur: f64, normal: [f64; 2], advection_speed: f64) -> (f64, f64) {
    let beta = Array1::from_vec(vec![advection_speed, 1.0]);
    let normal_array = Array1::from(normal.to_vec());
    let dflux_dnx = advection_speed
        * (0.5
            * (ul
                + ur
                + 100.0 * (ul - ur) * (beta.dot(&normal_array))
                    / (100.0 * beta.dot(&normal_array)).cosh().powf(2.0)
                + (ul - ur) * (100.0 * beta.dot(&normal_array)).tanh()));
    let dflux_dny = 0.5
        * (ul
            + ur
            + 100.0 * (ul - ur) * (beta.dot(&normal_array))
                / (100.0 * beta.dot(&normal_array)).cosh().powf(2.0)
            + (ul - ur) * (100.0 * beta.dot(&normal_array)).tanh());
    (dflux_dnx, dflux_dny)
}
