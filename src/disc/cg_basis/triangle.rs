use ndarray::{Array1, Array2, ArrayView1};

use crate::disc::cg_basis::CGBasis2D;

/// Continuous Galerkin basis for triangular elements
/// Uses Lagrange shape functions to ensure continuity
pub struct TriangleCGBasis;

impl TriangleCGBasis {}

impl CGBasis2D for TriangleCGBasis {
    fn shape_functions(n: usize, r: ArrayView1<f64>, s: ArrayView1<f64>) -> Array2<f64> {
        let np = Self::num_nodes(n);
        let npts = r.len();
        let mut phi = Array2::<f64>::zeros((npts, np));
        
        if n == 0 {
            // P0: constant element
            for pt in 0..npts {
                phi[[pt, 0]] = 1.0;
            }
            return phi;
        }
        
        // Generate the shape functions directly using the barycentric formulation
        // For triangular elements with vertices at (0,0), (1,0), (0,1) in (r,s) coordinates
        // Barycentric coordinates are L1 = 1-r-s, L2 = r, L3 = s
        
        let mut node_idx = 0;
        for i in 0..=n {
            for j in 0..=(n - i) {
                let k = n - i - j;
                
                for pt in 0..npts {
                    // Barycentric coordinates at evaluation point
                    let l1 = 1.0 - r[pt] - s[pt];  // vertex (0,0)
                    let l2 = r[pt];                // vertex (1,0) 
                    let l3 = s[pt];                // vertex (0,1)
                    
                    // Shape function is product of 1D Lagrange polynomials in barycentric coords
                    let mut phi_val = 1.0;
                    
                    // Build shape function using Silvester's approach
                    // N_ijk(L1,L2,L3) = C * L1^i * L2^j * L3^k * P_i(2*L1-1) * P_j(2*L2-1) * P_k(2*L3-1)
                    // For simplicity, we use the direct Lagrange form
                    
                    if n == 1 {
                        // P1 case: simple linear shape functions
                        if i == 1 && j == 0 && k == 0 {
                            phi_val = l1;  // Node at (0,0)
                        } else if i == 0 && j == 1 && k == 0 {
                            phi_val = l2;  // Node at (1,0)
                        } else if i == 0 && j == 0 && k == 1 {
                            phi_val = l3;  // Node at (0,1)
                        }
                    } else if n == 2 {
                        // P2 case: quadratic shape functions  
                        if i == 2 && j == 0 && k == 0 {
                            phi_val = l1 * (2.0 * l1 - 1.0);  // Corner node (0,0)
                        } else if i == 0 && j == 2 && k == 0 {
                            phi_val = l2 * (2.0 * l2 - 1.0);  // Corner node (1,0)
                        } else if i == 0 && j == 0 && k == 2 {
                            phi_val = l3 * (2.0 * l3 - 1.0);  // Corner node (0,1)
                        } else if i == 1 && j == 1 && k == 0 {
                            phi_val = 4.0 * l1 * l2;          // Edge node (0.5,0)
                        } else if i == 1 && j == 0 && k == 1 {
                            phi_val = 4.0 * l1 * l3;          // Edge node (0,0.5)
                        } else if i == 0 && j == 1 && k == 1 {
                            phi_val = 4.0 * l2 * l3;          // Edge node (0.5,0.5)
                        }
                    } else {
                        // General case: use recursive formula for higher orders
                        // This is a simplified version - full implementation would use 
                        // proper orthogonal polynomials
                        phi_val = 1.0;
                        
                        // L1 contribution
                        if i > 0 {
                            let mut l1_contrib = 1.0;
                            for m in 0..i {
                                l1_contrib *= (n as f64 * l1 - m as f64) / (i as f64 - m as f64);
                            }
                            phi_val *= l1_contrib;
                        }
                        
                        // L2 contribution
                        if j > 0 {
                            let mut l2_contrib = 1.0;
                            for m in 0..j {
                                l2_contrib *= (n as f64 * l2 - m as f64) / (j as f64 - m as f64);
                            }
                            phi_val *= l2_contrib;
                        }
                        
                        // L3 contribution
                        if k > 0 {
                            let mut l3_contrib = 1.0;
                            for m in 0..k {
                                l3_contrib *= (n as f64 * l3 - m as f64) / (k as f64 - m as f64);
                            }
                            phi_val *= l3_contrib;
                        }
                    }
                    
                    phi[[pt, node_idx]] = phi_val;
                }
                
                node_idx += 1;
            }
        }
        
        phi
    }
    
    fn grad_shape_functions(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let np = Self::num_nodes(n);
        let npts = r.len();
        let mut dphi_dr = Array2::<f64>::zeros((npts, np));
        let mut dphi_ds = Array2::<f64>::zeros((npts, np));
        
        if n == 0 {
            // P0: constant element has zero derivatives
            return (dphi_dr, dphi_ds);
        }
        
        let mut node_idx = 0;
        for i in 0..=n {
            for j in 0..=(n - i) {
                let k = n - i - j;
                
                for pt in 0..npts {
                    // Barycentric coordinates at evaluation point
                    let l1 = 1.0 - r[pt] - s[pt];
                    let l2 = r[pt];
                    let l3 = s[pt];
                    
                    // Derivatives of barycentric coordinates
                    // dL1/dr = -1, dL1/ds = -1
                    // dL2/dr = 1,  dL2/ds = 0  
                    // dL3/dr = 0,  dL3/ds = 1
                    
                    let mut dphi_dr_val = 0.0;
                    let mut dphi_ds_val = 0.0;
                    
                    if n == 1 {
                        // P1 linear elements
                        if i == 1 && j == 0 && k == 0 {
                            dphi_dr_val = -1.0;  // dL1/dr
                            dphi_ds_val = -1.0;  // dL1/ds
                        } else if i == 0 && j == 1 && k == 0 {
                            dphi_dr_val = 1.0;   // dL2/dr
                            dphi_ds_val = 0.0;   // dL2/ds
                        } else if i == 0 && j == 0 && k == 1 {
                            dphi_dr_val = 0.0;   // dL3/dr
                            dphi_ds_val = 1.0;   // dL3/ds
                        }
                    } else if n == 2 {
                        // P2 quadratic elements
                        if i == 2 && j == 0 && k == 0 {
                            // d/dr[L1*(2*L1-1)] = (2*L1-1)*(-1) + L1*2*(-1) = -(4*L1-1)
                            dphi_dr_val = -(4.0 * l1 - 1.0);
                            dphi_ds_val = -(4.0 * l1 - 1.0);
                        } else if i == 0 && j == 2 && k == 0 {
                            // d/dr[L2*(2*L2-1)] = (2*L2-1)*1 + L2*2*1 = 4*L2-1
                            dphi_dr_val = 4.0 * l2 - 1.0;
                            dphi_ds_val = 0.0;
                        } else if i == 0 && j == 0 && k == 2 {
                            // d/dr[L3*(2*L3-1)] = 0, d/ds[L3*(2*L3-1)] = 4*L3-1
                            dphi_dr_val = 0.0;
                            dphi_ds_val = 4.0 * l3 - 1.0;
                        } else if i == 1 && j == 1 && k == 0 {
                            // d/dr[4*L1*L2] = 4*(-1)*L2 + 4*L1*1 = 4*(L1-L2)
                            dphi_dr_val = 4.0 * (l1 - l2);
                            dphi_ds_val = 4.0 * (-l2);
                        } else if i == 1 && j == 0 && k == 1 {
                            // d/dr[4*L1*L3] = 4*(-1)*L3 = -4*L3
                            dphi_dr_val = -4.0 * l3;
                            dphi_ds_val = 4.0 * (l1 - l3);
                        } else if i == 0 && j == 1 && k == 1 {
                            // d/dr[4*L2*L3] = 4*1*L3 = 4*L3
                            dphi_dr_val = 4.0 * l3;
                            dphi_ds_val = 4.0 * l2;
                        }
                    } else {
                        // Higher order case - use product rule on the general form
                        // This is more complex and would require implementing the full derivative
                        // For now, fall back to a simple finite difference
                        let eps = 1e-8;
                        let r_plus = Array1::from(vec![r[pt] + eps]);
                        let s_same = Array1::from(vec![s[pt]]);
                        let phi_r_plus = Self::shape_functions(n, r_plus.view(), s_same.view());
                        
                        let r_base = Array1::from(vec![r[pt]]);
                        let s_base = Array1::from(vec![s[pt]]);
                        let phi_base = Self::shape_functions(n, r_base.view(), s_base.view());
                        
                        dphi_dr_val = (phi_r_plus[[0, node_idx]] - phi_base[[0, node_idx]]) / eps;
                        
                        let r_same = Array1::from(vec![r[pt]]);
                        let s_plus = Array1::from(vec![s[pt] + eps]);
                        let phi_s_plus = Self::shape_functions(n, r_same.view(), s_plus.view());
                        
                        dphi_ds_val = (phi_s_plus[[0, node_idx]] - phi_base[[0, node_idx]]) / eps;
                    }
                    
                    dphi_dr[[pt, node_idx]] = dphi_dr_val;
                    dphi_ds[[pt, node_idx]] = dphi_ds_val;
                }
                
                node_idx += 1;
            }
        }
        
        (dphi_dr, dphi_ds)
    }
    
    fn nodes2d(n: usize) -> (Array1<f64>, Array1<f64>) {
        let np = Self::num_nodes(n);
        let mut r = Array1::<f64>::zeros(np);
        let mut s = Array1::<f64>::zeros(np);
        
        // Generate equidistant nodes on the reference triangle
        let mut idx = 0;
        for i in 0..=n {
            for j in 0..=(n - i) {
                r[idx] = j as f64 / n as f64;
                s[idx] = i as f64 / n as f64;
                idx += 1;
            }
        }
        
        (r, s)
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
    
    #[test]
    fn test_kronecker_delta() {
        // Test that shape function i is 1 at node i and 0 at other nodes
        let n = 2;
        let (r_nodes, s_nodes) = TriangleCGBasis::nodes2d(n);
        let phi = TriangleCGBasis::shape_functions(n, r_nodes.view(), s_nodes.view());
        
        let np = r_nodes.len();
        for i in 0..np {
            for j in 0..np {
                if i == j {
                    assert_relative_eq!(phi[[i, j]], 1.0, epsilon = 1e-12);
                } else {
                    assert_relative_eq!(phi[[i, j]], 0.0, epsilon = 1e-12);
                }
            }
        }
    }
    
    #[test]
    fn test_gradient_consistency() {
        // Test that gradients are consistent with finite differences
        let eps = 1e-7;
        
        // Test at an interior point
        let r = Array1::from(vec![0.3]);
        let s = Array1::from(vec![0.3]);
        
        let (dphi_dr, dphi_ds) = TriangleCGBasis::grad_shape_functions(2, r.view(), s.view());
        
        // Check with finite differences
        let r_plus = Array1::from(vec![0.3 + eps]);
        let r_minus = Array1::from(vec![0.3 - eps]);
        let s_plus = Array1::from(vec![0.3 + eps]);
        let s_minus = Array1::from(vec![0.3 - eps]);
        
        let phi_r_plus = TriangleCGBasis::shape_functions(2, r_plus.view(), s.view());
        let phi_r_minus = TriangleCGBasis::shape_functions(2, r_minus.view(), s.view());
        let phi_s_plus = TriangleCGBasis::shape_functions(2, r.view(), s_plus.view());
        let phi_s_minus = TriangleCGBasis::shape_functions(2, r.view(), s_minus.view());
        
        for j in 0..phi_r_plus.ncols() {
            let fd_dr = (phi_r_plus[[0, j]] - phi_r_minus[[0, j]]) / (2.0 * eps);
            let fd_ds = (phi_s_plus[[0, j]] - phi_s_minus[[0, j]]) / (2.0 * eps);
            
            assert_relative_eq!(dphi_dr[[0, j]], fd_dr, epsilon = 1e-5);
            assert_relative_eq!(dphi_ds[[0, j]], fd_ds, epsilon = 1e-5);
        }
    }
}