use ndarray::{Array1, ArrayView1};

pub fn flux1d(q: ArrayView1<f64>) -> Array1<f64> {
    let mut f = Array1::zeros(1);
    f[0] = 0.5 * q[0] * q[0]; 
    f
}
