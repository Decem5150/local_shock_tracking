pub fn smoothed_upwind(ul: f64, ur: f64, normal: [f64; 2], advection_speed: f64) -> f64 {
    let beta = [advection_speed, 1.0];
    let beta_dot_normal = beta[0] * normal[0] + beta[1] * normal[1];
    let result = 0.5
        * (beta_dot_normal * (ul + ur)
            + (beta_dot_normal * (100.0 * beta_dot_normal).tanh()) * (ul - ur));
    result
}
pub fn dflux_dul(normal: [f64; 2], advection_speed: f64) -> f64 {
    let beta = [advection_speed, 1.0];
    let beta_dot_normal = beta[0] * normal[0] + beta[1] * normal[1];
    let result = 0.5 * (beta_dot_normal + (beta_dot_normal * (100.0 * beta_dot_normal).tanh()));
    result
}
pub fn dflux_dur(normal: [f64; 2], advection_speed: f64) -> f64 {
    let beta = [advection_speed, 1.0];
    let beta_dot_normal = beta[0] * normal[0] + beta[1] * normal[1];
    let result = 0.5 * (beta_dot_normal - (beta_dot_normal * (100.0 * beta_dot_normal).tanh()));
    result
}
pub fn dflux_dnormal(ul: f64, ur: f64, normal: [f64; 2], advection_speed: f64) -> (f64, f64) {
    let beta = [advection_speed, 1.0];
    let beta_dot_normal = beta[0] * normal[0] + beta[1] * normal[1];
    let dflux_dnx = advection_speed
        * (0.5
            * (ul
                + ur
                + 100.0 * (ul - ur) * beta_dot_normal
                    / (100.0 * beta_dot_normal).cosh().powf(2.0)
                + (ul - ur) * (100.0 * beta_dot_normal).tanh()));
    let dflux_dny = 0.5
        * (ul
            + ur
            + 100.0 * (ul - ur) * beta_dot_normal / (100.0 * beta_dot_normal).cosh().powf(2.0)
            + (ul - ur) * (100.0 * beta_dot_normal).tanh());
    (dflux_dnx, dflux_dny)
}
