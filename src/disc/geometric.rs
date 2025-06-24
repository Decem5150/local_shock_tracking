pub trait Geometric1D {
    fn compute_ref_normal(local_id: usize) -> [f64; 2] {
        match local_id {
            0 => [-1.0, 0.0],
            1 => [1.0, 0.0],
            _ => {
                panic!("Invalid edge ID");
            }
        }
    }
    fn compute_interval_length(x: &[f64]) -> f64 {
        x[1] - x[0]
    }
}
pub trait Geometric2D {
    fn compute_normal(x0: f64, y0: f64, x1: f64, y1: f64) -> [f64; 2] {
        // normalized normal vector
        let normal = [y1 - y0, x0 - x1];
        let normal_magnitude = (normal[0].powi(2) + normal[1].powi(2)).sqrt();
        [normal[0] / normal_magnitude, normal[1] / normal_magnitude]
    }
    fn compute_ref_normal(local_id: usize) -> [f64; 2] {
        match local_id {
            0 => {
                // Bottom edge: from (0,0) to (1,0)
                // Outward normal points downward
                [0.0, -1.0]
            }
            1 => {
                // Hypotenuse edge: from (1,0) to (0,1)
                // Edge vector: (-1, 1), normal: (1, 1) normalized
                let sqrt2_inv = 1.0 / (2.0_f64.sqrt());
                [sqrt2_inv, sqrt2_inv]
            }
            2 => {
                // Left edge: from (0,1) to (0,0)
                // Outward normal points leftward
                [-1.0, 0.0]
            }
            _ => {
                panic!("Invalid edge ID");
            }
        }
    }
    fn compute_ref_edge_length(local_id: usize) -> f64 {
        match local_id {
            0 => 2.0,
            1 => 2.0 * 2.0_f64.sqrt(),
            2 => 2.0,
            _ => panic!("Invalid edge ID"),
        }
    }
    fn evaluate_jacob(_xi: f64, _eta: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]) {
        // For triangular elements with reference triangle vertices at:
        // Node 0: (-1, -1)
        // Node 1: (1, -1)
        // Node 2: (-1, 1)
        // Shape functions for linear triangle:
        // N0 = -(xi + eta)/2     (node 0)
        // N1 = (1 + xi)/2        (node 1)
        // N2 = (1 + eta)/2       (node 2)

        let dn_dxi = [
            -0.5, // dN0/dξ
            0.5,  // dN1/dξ
            0.0,  // dN2/dξ
        ];
        let dn_deta = [
            -0.5, // dN0/dη
            0.0,  // dN1/dη
            0.5,  // dN2/dη
        ];

        let mut dx_dxi = 0.0;
        let mut dx_deta = 0.0;
        let mut dy_dxi = 0.0;
        let mut dy_deta = 0.0;

        for k in 0..3 {
            dx_dxi += dn_dxi[k] * x[k];
            dx_deta += dn_deta[k] * x[k];
            dy_dxi += dn_dxi[k] * y[k];
            dy_deta += dn_deta[k] * y[k];
        }

        let jacob_det = dx_dxi * dy_deta - dx_deta * dy_dxi;
        let jacob_inv_t = [
            dy_deta / jacob_det,
            -dy_dxi / jacob_det,
            -dx_deta / jacob_det,
            dx_dxi / jacob_det,
        ];

        (jacob_det, jacob_inv_t)
    }
    fn compute_element_area(x: &[f64], y: &[f64]) -> f64 {
        // For triangular elements, assumes x and y are slices of length 3
        0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])).abs()
    }
}
