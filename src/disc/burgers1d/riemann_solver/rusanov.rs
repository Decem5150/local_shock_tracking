use ndarray::{Array1, ArrayView1};

pub fn rusanov(ul: ArrayView1<f64>, ur: ArrayView1<f64>) -> Array1<f64> {
    // Physical flux function for Burgers equation: f(u) = u²/2
    let fl = 0.5 * ul[0] * ul[0];
    let fr = 0.5 * ur[0] * ur[0];

    // Maximum wave speed (characteristic speed)
    // For Burgers equation, it's max(|ul|, |ur|)
    let wave_speed = ul[0].abs().max(ur[0].abs());

    // Rusanov flux: 0.5 * (f(ul) + f(ur) - α(ur - ul))
    // where α is the maximum wave speed
    let mut flux = Array1::zeros(1);
    flux[0] = 0.5 * (fl + fr - wave_speed * (ur[0] - ul[0]));
    flux
}
