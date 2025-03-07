use ndarray::Array1;

pub fn space_time_flux1d(u: f64, advection_speed: f64) -> Array1<f64> {
    Array1::from_vec(vec![u, advection_speed * u])
}
