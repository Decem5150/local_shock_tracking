use ndarray::{Array1, Array2, ArrayView1};

use crate::disc::cg_basis::CGBasis2D;

/// Continuous Galerkin basis for triangular elements
/// Uses Lagrange shape functions to ensure continuity
pub struct TriangleCGBasis {
    pub cub_l1: Array1<f64>,
    pub cub_l2: Array1<f64>,
    pub cub_w: Array1<f64>,
    pub dxi_cub: Array2<f64>,
    pub deta_cub: Array2<f64>,
}

impl TriangleCGBasis {
    pub fn new(n: usize) -> Self {
        let (cub_l1, cub_l2, cub_w) = Self::cubature_points(2 * n - 1);
        let _phi = Self::shape_functions(n, cub_l1.view(), cub_l2.view(), cub_w.view());
        let (dxi_cub, deta_cub) =
            Self::grad_shape_functions(n, cub_l1.view(), cub_l2.view(), cub_w.view());
        Self {
            cub_l1,
            cub_l2,
            cub_w,
            dxi_cub,
            deta_cub,
        }
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
                phi
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
                phi
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
                phi
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
                (dphi_dxi, dphi_deta)
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
                (dphi_dxi, dphi_deta)
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
                (dphi_dxi, dphi_deta)
            }
            _ => {
                unimplemented!(
                    "Higher order gradient shape functions not implemented for triangles"
                );
            }
        }
    }

    fn cubature_points(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        match n {
            1 => {
                // 1-point rule (centroid)
                let l1 = Array1::from(vec![1.0 / 3.0]);
                let l2 = Array1::from(vec![1.0 / 3.0]);
                let w = Array1::from(vec![0.5]);
                (l1, l2, w)
            }
            2 => {
                // 3-point rule (edge midpoints)
                let l1 = Array1::from(vec![0.5, 0.0, 0.5]);
                let l2 = Array1::from(vec![0.5, 0.5, 0.0]);
                let w = Array1::from(vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]);
                (l1, l2, w)
            }
            3 => {
                // 4-point rule
                let l1 = Array1::from(vec![1.0 / 3.0, 0.6, 0.2, 0.2]);
                let l2 = Array1::from(vec![1.0 / 3.0, 0.2, 0.6, 0.2]);
                let w = Array1::from(vec![-9.0 / 32.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0]);
                (l1, l2, w)
            }
            _ => {
                // For higher orders, use a 6-point rule as default
                let a = 0.445948490915965;
                let b = 0.091576213509771;
                let l1 = Array1::from(vec![a, 1.0 - 2.0 * a, a, b, 1.0 - 2.0 * b, b]);
                let l2 = Array1::from(vec![a, a, 1.0 - 2.0 * a, b, b, 1.0 - 2.0 * b]);
                let w1 = 0.111690794839005;
                let w2 = 0.054975871827661;
                let w = Array1::from(vec![w1, w1, w1, w2, w2, w2]);
                (l1, l2, w)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array;

    #[test]
    fn test_partition_of_unity() {
        // Test that shape functions sum to 1 at any point
        // Test at several points in (r,s) coordinates
        let test_r = Array1::from(vec![0.0, 0.5, 0.3, 1.0, 0.0, 0.25]);
        let test_s = Array1::from(vec![0.0, 0.0, 0.3, 0.0, 1.0, 0.25]);

        // Convert (r,s) to barycentric coordinates (l1, l2, l3)
        let test_l1 = &Array::ones(test_r.len()) - &test_r - &test_s;
        let test_l2 = test_r.clone();
        let test_l3 = test_s.clone();

        let phi =
            TriangleCGBasis::shape_functions(2, test_l1.view(), test_l2.view(), test_l3.view());

        for i in 0..test_r.len() {
            let sum: f64 = phi.row(i).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_grad_shape_functions_p0() {
        // Test P0 gradient shape functions (should be zero)
        let test_l1 = Array1::from(vec![0.3, 0.5, 0.2]);
        let test_l2 = Array1::from(vec![0.3, 0.3, 0.4]);
        let test_l3 = Array1::from(vec![0.4, 0.2, 0.4]);

        let (dphi_dxi, dphi_deta) = TriangleCGBasis::grad_shape_functions(
            0,
            test_l1.view(),
            test_l2.view(),
            test_l3.view(),
        );

        // All derivatives should be zero for constant elements
        assert_eq!(dphi_dxi.shape(), &[3, 1]);
        assert_eq!(dphi_deta.shape(), &[3, 1]);

        for i in 0..3 {
            assert_relative_eq!(dphi_dxi[[i, 0]], 0.0, epsilon = 1e-12);
            assert_relative_eq!(dphi_deta[[i, 0]], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_grad_shape_functions_p1() {
        // Test P1 gradient shape functions
        let test_l1 = Array1::from(vec![0.3, 0.5, 0.2]);
        let test_l2 = Array1::from(vec![0.3, 0.3, 0.4]);
        let test_l3 = Array1::from(vec![0.4, 0.2, 0.4]);

        let (dphi_dxi, dphi_deta) = TriangleCGBasis::grad_shape_functions(
            1,
            test_l1.view(),
            test_l2.view(),
            test_l3.view(),
        );

        // Check shape
        assert_eq!(dphi_dxi.shape(), &[3, 3]);
        assert_eq!(dphi_deta.shape(), &[3, 3]);

        // For P1, derivatives are constants:
        // dphi1/dxi = -1, dphi1/deta = -1
        // dphi2/dxi = 1,  dphi2/deta = 0
        // dphi3/dxi = 0,  dphi3/deta = 1
        for pt in 0..3 {
            assert_relative_eq!(dphi_dxi[[pt, 0]], -1.0, epsilon = 1e-12);
            assert_relative_eq!(dphi_deta[[pt, 0]], -1.0, epsilon = 1e-12);
            assert_relative_eq!(dphi_dxi[[pt, 1]], 1.0, epsilon = 1e-12);
            assert_relative_eq!(dphi_deta[[pt, 1]], 0.0, epsilon = 1e-12);
            assert_relative_eq!(dphi_dxi[[pt, 2]], 0.0, epsilon = 1e-12);
            assert_relative_eq!(dphi_deta[[pt, 2]], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_grad_shape_functions_p2() {
        // Test P2 gradient shape functions at specific points
        // Test at vertex (1,0,0) in barycentric coordinates
        let test_l1 = Array1::from(vec![1.0]);
        let test_l2 = Array1::from(vec![0.0]);
        let test_l3 = Array1::from(vec![0.0]);

        let (dphi_dxi, dphi_deta) = TriangleCGBasis::grad_shape_functions(
            2,
            test_l1.view(),
            test_l2.view(),
            test_l3.view(),
        );

        // At vertex (1,0,0), check specific values
        // Node 0 (corner): dphi/dxi = -(4*1-1) = -3
        assert_relative_eq!(dphi_dxi[[0, 0]], -3.0, epsilon = 1e-12);
        assert_relative_eq!(dphi_deta[[0, 0]], -3.0, epsilon = 1e-12);

        // Node 1 (corner): dphi/dxi = 4*0-1 = -1
        assert_relative_eq!(dphi_dxi[[0, 1]], -1.0, epsilon = 1e-12);
        assert_relative_eq!(dphi_deta[[0, 1]], 0.0, epsilon = 1e-12);

        // Node 2 (corner): dphi/dxi = 0, dphi/deta = 4*0-1 = -1
        assert_relative_eq!(dphi_dxi[[0, 2]], 0.0, epsilon = 1e-12);
        assert_relative_eq!(dphi_deta[[0, 2]], -1.0, epsilon = 1e-12);

        // Test at centroid (1/3, 1/3, 1/3)
        let test_l1 = Array1::from(vec![1.0 / 3.0]);
        let test_l2 = Array1::from(vec![1.0 / 3.0]);
        let test_l3 = Array1::from(vec![1.0 / 3.0]);

        let (dphi_dxi, dphi_deta) = TriangleCGBasis::grad_shape_functions(
            2,
            test_l1.view(),
            test_l2.view(),
            test_l3.view(),
        );

        // Verify sum of derivatives
        // The sum of all shape function derivatives should relate to partition of unity
        let sum_dxi: f64 = dphi_dxi.row(0).sum();
        let sum_deta: f64 = dphi_deta.row(0).sum();

        // For shape functions that sum to 1, derivative of sum is 0
        assert_relative_eq!(sum_dxi, 0.0, epsilon = 1e-12);
        assert_relative_eq!(sum_deta, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_grad_shape_functions_consistency() {
        // Test that gradients are consistent across different evaluation points
        let n_points = 5;
        let test_l1 = Array1::from(vec![0.2, 0.5, 0.3, 0.1, 0.4]);
        let test_l2 = Array1::from(vec![0.3, 0.2, 0.4, 0.6, 0.3]);
        let test_l3 = Array1::from(vec![0.5, 0.3, 0.3, 0.3, 0.3]);

        for order in 0..=2 {
            let (dphi_dxi, dphi_deta) = TriangleCGBasis::grad_shape_functions(
                order,
                test_l1.view(),
                test_l2.view(),
                test_l3.view(),
            );

            let expected_nodes = TriangleCGBasis::num_nodes(order);
            assert_eq!(dphi_dxi.shape(), &[n_points, expected_nodes]);
            assert_eq!(dphi_deta.shape(), &[n_points, expected_nodes]);

            // Sum of derivatives should be zero (partition of unity property)
            for pt in 0..n_points {
                let sum_dxi: f64 = dphi_dxi.row(pt).sum();
                let sum_deta: f64 = dphi_deta.row(pt).sum();

                if order > 0 {
                    assert_relative_eq!(sum_dxi, 0.0, epsilon = 1e-10);
                    assert_relative_eq!(sum_deta, 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_cubature_points_order_1() {
        // Test 1-point rule (centroid)
        let (l1, l2, w) = TriangleCGBasis::cubature_points(1);

        assert_eq!(l1.len(), 1);
        assert_eq!(l2.len(), 1);
        assert_eq!(w.len(), 1);

        // Check centroid coordinates
        assert_relative_eq!(l1[0], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(l2[0], 1.0 / 3.0, epsilon = 1e-12);

        // Check weight (area of reference triangle is 0.5)
        assert_relative_eq!(w[0], 0.5, epsilon = 1e-12);

        // Verify l3 = 1 - l1 - l2
        let l3 = 1.0 - l1[0] - l2[0];
        assert_relative_eq!(l3, 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cubature_points_order_2() {
        // Test 3-point rule (edge midpoints)
        let (l1, l2, w) = TriangleCGBasis::cubature_points(2);

        assert_eq!(l1.len(), 3);
        assert_eq!(l2.len(), 3);
        assert_eq!(w.len(), 3);

        // Check edge midpoint coordinates
        assert_relative_eq!(l1[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(l2[0], 0.5, epsilon = 1e-12);

        assert_relative_eq!(l1[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(l2[1], 0.5, epsilon = 1e-12);

        assert_relative_eq!(l1[2], 0.5, epsilon = 1e-12);
        assert_relative_eq!(l2[2], 0.0, epsilon = 1e-12);

        // Check weights sum to triangle area
        let weight_sum: f64 = w.sum();
        assert_relative_eq!(weight_sum, 0.5, epsilon = 1e-12);

        // Check individual weights
        for i in 0..3 {
            assert_relative_eq!(w[i], 1.0 / 6.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_cubature_points_order_3() {
        // Test 4-point rule
        let (l1, l2, w) = TriangleCGBasis::cubature_points(3);

        assert_eq!(l1.len(), 4);
        assert_eq!(l2.len(), 4);
        assert_eq!(w.len(), 4);

        // Check first point (centroid)
        assert_relative_eq!(l1[0], 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(l2[0], 1.0 / 3.0, epsilon = 1e-12);

        // Check other points
        assert_relative_eq!(l1[1], 0.6, epsilon = 1e-12);
        assert_relative_eq!(l2[1], 0.2, epsilon = 1e-12);

        assert_relative_eq!(l1[2], 0.2, epsilon = 1e-12);
        assert_relative_eq!(l2[2], 0.6, epsilon = 1e-12);

        assert_relative_eq!(l1[3], 0.2, epsilon = 1e-12);
        assert_relative_eq!(l2[3], 0.2, epsilon = 1e-12);

        // Check weights sum to triangle area
        let weight_sum: f64 = w.sum();
        assert_relative_eq!(weight_sum, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_cubature_points_higher_order() {
        // Test higher order (default 6-point rule)
        let (l1, l2, w) = TriangleCGBasis::cubature_points(4);

        assert_eq!(l1.len(), 6);
        assert_eq!(l2.len(), 6);
        assert_eq!(w.len(), 6);

        // Check that all barycentric coordinates are valid (between 0 and 1)
        for i in 0..6 {
            assert!(l1[i] >= 0.0 && l1[i] <= 1.0);
            assert!(l2[i] >= 0.0 && l2[i] <= 1.0);
            let l3 = 1.0 - l1[i] - l2[i];
            assert!((-1e-12..=1.0 + 1e-12).contains(&l3)); // Allow small numerical error
        }

        // Check weights sum to triangle area
        let weight_sum: f64 = w.sum();
        assert_relative_eq!(weight_sum, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_cubature_integration_constant() {
        // Test that integrating a constant function gives correct area
        for order in 1..=5 {
            let (_l1, _l2, w) = TriangleCGBasis::cubature_points(order);

            // Integrate f(x,y) = 1 over the reference triangle
            let mut integral = 0.0;
            for i in 0..w.len() {
                integral += w[i] * 1.0; // f = 1
            }

            // Should equal the area of reference triangle
            assert_relative_eq!(integral, 0.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_cubature_integration_linear() {
        // Test integration of linear functions
        // For order >= 2, should integrate linear functions exactly
        for order in 2..=5 {
            let (l1, l2, w) = TriangleCGBasis::cubature_points(order);

            // Integrate f(r,s) = r over the reference triangle
            // Reference triangle has vertices at (0,0), (1,0), (0,1)
            // r corresponds to l2 (second barycentric coordinate)
            let mut integral_r = 0.0;
            for i in 0..w.len() {
                integral_r += w[i] * l2[i];
            }

            // Exact integral of r over reference triangle is 1/6
            assert_relative_eq!(integral_r, 1.0 / 6.0, epsilon = 1e-10);

            // Integrate f(r,s) = s over the reference triangle
            // s corresponds to l3 (third barycentric coordinate)
            let mut integral_s = 0.0;
            for i in 0..w.len() {
                let l3 = 1.0 - l1[i] - l2[i];
                integral_s += w[i] * l3;
            }

            // Exact integral of s over reference triangle is 1/6
            assert_relative_eq!(integral_s, 1.0 / 6.0, epsilon = 1e-10);
        }
    }
}
