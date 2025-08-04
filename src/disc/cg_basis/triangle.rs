use ndarray::{Array1, Array2, ArrayView1};

use crate::disc::cg_basis::CGBasis2D;

/// Continuous Galerkin basis for triangular elements
/// Uses Lagrange shape functions to ensure continuity
pub struct TriangleCGBasis {
    pub dxi_cub: Array2<f64>,
    pub deta_cub: Array2<f64>,
}

impl TriangleCGBasis {
    pub fn new(n: usize) -> Self {
        let (cub_l1, cub_l2, cub_w) = Self::cubature_points(2 * n - 1);
        let _phi = Self::shape_functions(n, cub_l1.view(), cub_l2.view(), cub_w.view());
        let (dxi_cub, deta_cub) =
            Self::grad_shape_functions(n, cub_l1.view(), cub_l2.view(), cub_w.view());
        Self { dxi_cub, deta_cub }
    }
}

impl CGBasis2D for TriangleCGBasis {
    fn shape_functions(
        n: usize,
        l1: ArrayView1<f64>,
        l2: ArrayView1<f64>,
        l3: ArrayView1<f64>,
    ) -> Array2<f64> {
        let np = Self::num_nodes(n);
        let npts = l1.len();
        let mut phi = Array2::<f64>::zeros((npts, np));

        match n {
            0 => {
                // P0: constant element
                for pt in 0..npts {
                    phi[[pt, 0]] = 1.0;
                }
                return phi;
            }
            1 => {
                // Generate the shape functions directly using the barycentric formulation
                // For triangular elements with vertices at (0,0), (1,0), (0,1) in (r,s) coordinates
                // Barycentric coordinates are L1 = 1-r-s, L2 = r, L3 = s
                for pt in 0..npts {
                    phi[(pt, 0)] = l1[pt];
                    phi[(pt, 1)] = l2[pt];
                    phi[(pt, 2)] = l3[pt];
                }
                return phi;
            }
            2 => {
                for pt in 0..npts {
                    // Corner nodes
                    phi[(pt, 0)] = l1[pt] * (2.0 * l1[pt] - 1.0);
                    phi[(pt, 1)] = l2[pt] * (2.0 * l2[pt] - 1.0);
                    phi[(pt, 2)] = l3[pt] * (2.0 * l3[pt] - 1.0);
                    // Edge nodes
                    phi[(pt, 3)] = 4.0 * l1[pt] * l2[pt];
                    phi[(pt, 4)] = 4.0 * l1[pt] * l3[pt];
                    phi[(pt, 5)] = 4.0 * l2[pt] * l3[pt];
                }
                return phi;
            }
            _ => {
                unimplemented!("Higher order shape functions not implemented for triangles");
            }
        }
    }

    fn grad_shape_functions(
        n: usize,
        l1: ArrayView1<f64>,
        l2: ArrayView1<f64>,
        l3: ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let np = Self::num_nodes(n);
        let npts = l1.len();
        let mut dphi_dxi = Array2::<f64>::zeros((npts, np));
        let mut dphi_deta = Array2::<f64>::zeros((npts, np));

        match n {
            0 => {
                // P0: constant element has zero derivatives
                return (dphi_dxi, dphi_deta);
            }
            1 => {
                // P1 linear elements
                // Derivatives of barycentric coordinates:
                // dL1/dr = -1, dL1/ds = -1
                // dL2/dr = 1,  dL2/ds = 0
                // dL3/dr = 0,  dL3/ds = 1
                for pt in 0..npts {
                    dphi_dxi[(pt, 0)] = -1.0;
                    dphi_deta[(pt, 0)] = -1.0;
                    dphi_dxi[(pt, 1)] = 1.0;
                    dphi_deta[(pt, 1)] = 0.0;
                    dphi_dxi[(pt, 2)] = 0.0;
                    dphi_deta[(pt, 2)] = 1.0;
                }
                return (dphi_dxi, dphi_deta);
            }
            2 => {
                // P2 quadratic elements
                for pt in 0..npts {
                    // Corner nodes
                    // d/dr[L1*(2*L1-1)] = (2*L1-1)*(-1) + L1*2*(-1) = -(4*L1-1)
                    dphi_dxi[(pt, 0)] = -(4.0 * l1[pt] - 1.0);
                    dphi_deta[(pt, 0)] = -(4.0 * l1[pt] - 1.0);

                    // d/dr[L2*(2*L2-1)] = (2*L2-1)*1 + L2*2*1 = 4*L2-1
                    dphi_dxi[(pt, 1)] = 4.0 * l2[pt] - 1.0;
                    dphi_deta[(pt, 1)] = 0.0;

                    // d/dr[L3*(2*L3-1)] = 0, d/ds[L3*(2*L3-1)] = 4*L3-1
                    dphi_dxi[(pt, 2)] = 0.0;
                    dphi_deta[(pt, 2)] = 4.0 * l3[pt] - 1.0;

                    // Edge nodes
                    // d/dr[4*L1*L2] = 4*(-1)*L2 + 4*L1*1 = 4*(L1-L2)
                    dphi_dxi[(pt, 3)] = 4.0 * (l1[pt] - l2[pt]);
                    dphi_deta[(pt, 3)] = -4.0 * l2[pt];

                    // d/dr[4*L1*L3] = 4*(-1)*L3 = -4*L3
                    dphi_dxi[(pt, 4)] = -4.0 * l3[pt];
                    dphi_deta[(pt, 4)] = 4.0 * (l1[pt] - l3[pt]);

                    // d/dr[4*L2*L3] = 4*1*L3 = 4*L3
                    dphi_dxi[(pt, 5)] = 4.0 * l3[pt];
                    dphi_deta[(pt, 5)] = 4.0 * l2[pt];
                }
                return (dphi_dxi, dphi_deta);
            }
            _ => {
                unimplemented!(
                    "Higher order gradient shape functions not implemented for triangles"
                );
            }
        }
    }

    fn cubature_points(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        match n {
            _ => {
                let l1 = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                let l2 = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                let l3 = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                let w = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                todo!();
                (l1, l2, l3, w)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_partition_of_unity() {
        // Test that shape functions sum to 1 at any point
        // Test at several points
        let test_r = Array1::from(vec![0.0, 0.5, 0.3, 1.0, 0.0, 0.25]);
        let test_s = Array1::from(vec![0.0, 0.0, 0.3, 0.0, 1.0, 0.25]);

        let phi = TriangleCGBasis::shape_functions(2, test_r.view(), test_s.view());

        for i in 0..test_r.len() {
            let sum: f64 = phi.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        }
    }
}
