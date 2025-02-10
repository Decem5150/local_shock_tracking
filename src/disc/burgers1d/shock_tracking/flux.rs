use ndarray::{Array1, ArrayView1};

pub fn space_time_flux(q: ArrayView1<f64>) -> Array1<f64> {
    let mut f = Array1::zeros(2);
    f[0] = q[0];
    f[1] = 0.5 * q[0] * q[0];
    f
}