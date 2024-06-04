use ndarray::{Array, Ix1};

pub fn flux1d(q: &Array<f64, Ix1>, hcr: f64) -> [f64; 3] {
    let mut f = [0.0f64; 3];
    let u = q[1] / q[0];
    let p = (hcr - 1.0) * (q[2] - 0.5 * (q[1] * q[1]) / q[0]);
    f[0] = q[1];
    f[1] = q[1] * u + p;
    f[2] = u * (q[2] + p);
    f
}