pub fn burgers1d_space_time_flux(u: f64) -> [f64; 2] {
    [0.5 * u * u, u]
}
