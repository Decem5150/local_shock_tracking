use ndarray::Array2;

use crate::disc::{cg_basis::triangle::TriangleCGBasis, mesh::mesh2d::{Mesh2d, Status, TriangleElement}};

pub struct LinearElliptic {
    stiffness: Array2<f64>,
}
impl LinearElliptic {
    fn compute_stiffness(&self, mesh: &Mesh2d<TriangleElement>, basis: &TriangleCGBasis) -> Array2<f64> {
        let mut stiffness = Array2::<f64>::zeros((mesh.ref_nodes.len(), mesh.ref_nodes.len()));
        let (min_id, min_area) = self.find_smallest_element(mesh);
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let x: [f64; 3] =
                    std::array::from_fn(|i| mesh.ref_nodes[elem.inodes[i]].as_ref().x);
                let y: [f64; 3] =
                    std::array::from_fn(|i| mesh.ref_nodes[elem.inodes[i]].as_ref().y);
                let area = Self::compute_element_area(&x, &y);
                let k = min_area / area;
                for igp in self.
            }
        }
        stiffness
    }
    fn find_smallest_element(&self, mesh: &Mesh2d<TriangleElement>) -> (usize, f64) {
        let mut min_area = f64::INFINITY;
        let mut min_element_id = 0;
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let x: [f64; 3] =
                    std::array::from_fn(|i| mesh.ref_nodes[elem.inodes[i]].as_ref().x);
                let y: [f64; 3] =
                    std::array::from_fn(|i| mesh.ref_nodes[elem.inodes[i]].as_ref().y);
                let area = Self::compute_element_area(&x, &y);
                if area < min_area {
                    min_area = area;
                    min_element_id = ielem;
                }
            }
        }
        (min_element_id, min_area)
    }
    fn compute_element_area(x: &[f64], y: &[f64]) -> f64 {
        // For triangular elements, assumes x and y are slices of length 3
        0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])).abs()
    }
    fn evaluate_jacob(order: usize, l1: f64, l2: f64, l3: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]) {
        // Note: l1, l2, l3 are barycentric coordinates
        // Conversion from (xi, eta) to (l1, l2, l3):
        // l1 = 1 - xi - eta
        // l2 = xi  
        // l3 = eta
        // Therefore: dL1/dxi = -1, dL1/deta = -1
        //           dL2/dxi = 1,  dL2/deta = 0
        //           dL3/dxi = 0,  dL3/deta = 1
        
        match order {
            1 => {
                // Linear element (P1) with 3 nodes
                // Shape functions: N0 = L1, N1 = L2, N2 = L3
                // Derivatives w.r.t. reference coordinates (xi, eta):
                let dn_dxi = [
                    -1.0, // dN0/dξ = dL1/dξ
                    1.0,  // dN1/dξ = dL2/dξ
                    0.0,  // dN2/dξ = dL3/dξ
                ];
                let dn_deta = [
                    -1.0, // dN0/dη = dL1/dη
                    0.0,  // dN1/dη = dL2/dη
                    1.0,  // dN2/dη = dL3/dη
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
            2 => {
                // Quadratic element (P2) with 6 nodes
                // Node ordering for P2 triangle:
                // Vertices: 0, 1, 2
                // Edge midpoints: 3 (between 0-1), 4 (between 1-2), 5 (between 2-0)
                
                // Shape function derivatives for P2 elements
                // Vertex nodes use: Ni = Li * (2*Li - 1)
                // Edge nodes use: Nij = 4 * Li * Lj
                // Derivatives w.r.t. xi and eta using chain rule:
                let dn_dxi = [
                    -(4.0 * l1 - 1.0), // dN0/dxi (vertex 0)
                    4.0 * l2 - 1.0,    // dN1/dxi (vertex 1)
                    0.0,               // dN2/dxi (vertex 2)
                    4.0 * (l1 - l2),   // dN3/dxi (edge 0-1)
                    4.0 * l3,          // dN4/dxi (edge 1-2)
                    -4.0 * l3,         // dN5/dxi (edge 2-0)
                ];

                let dn_deta = [
                    -(4.0 * l1 - 1.0), // dN0/deta (vertex 0)
                    0.0,               // dN1/deta (vertex 1)
                    4.0 * l3 - 1.0,    // dN2/deta (vertex 2)
                    -4.0 * l2,         // dN3/deta (edge 0-1)
                    4.0 * l2,          // dN4/deta (edge 1-2)
                    4.0 * (l1 - l3),   // dN5/deta (edge 2-0)
                ];

                let mut dx_dxi = 0.0;
                let mut dx_deta = 0.0;
                let mut dy_dxi = 0.0;
                let mut dy_deta = 0.0;

                for i in 0..6 {
                    dx_dxi += dn_dxi[i] * x[i];
                    dx_deta += dn_deta[i] * x[i];
                    dy_dxi += dn_dxi[i] * y[i];
                    dy_deta += dn_deta[i] * y[i];
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
            _ => {
                panic!(
                    "evaluate_jacob: Only linear (order=1) and quadratic (order=2) elements are supported"
                );
            }
        }
    }
}
