pub fn space_time_flux1d(u: f64, advection_speed: f64) -> [f64; 2] {
    [advection_speed * u, u]
}
