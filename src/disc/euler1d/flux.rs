use ndarray::{Array1, ArrayView1};

pub fn flux1d(q: ArrayView1<'_, f64>, hcr: f64) -> Array1<f64> {
    let mut f = Array1::zeros(3);
    let u = q[1] / q[0];
    let p = (hcr - 1.0) * (q[2] - 0.5 * (q[1] * q[1]) / q[0]);
    f[0] = q[1];
    f[1] = q[1] * u + p;
    f[2] = u * (q[2] + p);
    f
}
