pub fn rusanov(ul: f64, ur: f64) -> f64 {
    // Physical flux function for Burgers equation: f(u) = u²/2
    let fl = 0.5 * ul * ul;
    let fr = 0.5 * ur * ur;

    // Maximum wave speed (characteristic speed)
    // For Burgers equation, it's max(|ul|, |ur|)
    let wave_speed = ul.abs().max(ur.abs());

    // Rusanov flux: 0.5 * (f(ul) + f(ur) - α(ur - ul))
    // where α is the maximum wave speed
    0.5 * (fl + fr - wave_speed * (ur - ul))
}
