use super::mesh1d::Node;
use crate::disc::advection1d_space_time::boundary_condition::{BoundaryQuantity1d, BoundaryType};
use crate::disc::basis::lagrange1d::{LagrangeBasis1D, LagrangeBasis1DLobatto};
use ndarray::{Array, ArrayView1, Ix1, Ix2, Ix3, Ix4};

pub struct BoundaryPatch2d {
    pub iedges: Array<usize, Ix1>,
    pub boundary_type: BoundaryType,
    pub boundary_quantity: Option<BoundaryQuantity1d>,
}
pub struct Edge {
    pub inodes: Array<usize, Ix1>,
    pub parent_elements: Array<isize, Ix1>,
    pub local_ids: Array<isize, Ix1>,
    pub normal: [f64; 2],
}
pub struct Element2d {
    pub inodes: Array<usize, Ix1>,
    pub iedges: Array<usize, Ix1>,
    pub ineighbors: Array<isize, Ix1>,
    pub jacob_det: Array<f64, Ix2>,
    pub jacob_inv_t: Array<f64, Ix4>,
}
pub struct SubMesh2d {
    pub nodes: Array<Node, Ix1>,
    pub edges: Array<Edge, Ix1>,
    pub elements: Array<Element2d, Ix1>,
}
pub struct Mesh2d {
    pub nodes: Array<Node, Ix1>,
    pub edges: Array<Edge, Ix1>,
    pub elements: Array<Element2d, Ix1>,
    pub internal_edges: Array<usize, Ix1>,
    // pub internal_elements: Array<usize, Ix1>, // index of internal elements
    // pub boundary_elements: Array<usize, Ix1>, // index of boundary elements
    pub boundary_patches: Array<BoundaryPatch2d, Ix1>,
    pub elem_num: usize,
    pub node_num: usize,
}
impl Mesh2d {
    fn compute_jacob(&mut self, basis: &LagrangeBasis1DLobatto) {
        let ngp = basis.cell_gauss_points.len();

        for elem in self.elements.iter_mut() {
            let inodes: ArrayView1<usize> = elem.inodes.view();
            let iedges: ArrayView1<usize> = elem.iedges.view();

            match iedges.len() {
                // Triangle case commented out as requested
                4 => {
                    // Get node coordinates
                    let x = [
                        self.nodes[inodes[0]].x,
                        self.nodes[inodes[1]].x,
                        self.nodes[inodes[2]].x,
                        self.nodes[inodes[3]].x,
                    ];
                    let y = [
                        self.nodes[inodes[0]].y,
                        self.nodes[inodes[1]].y,
                        self.nodes[inodes[2]].y,
                        self.nodes[inodes[3]].y,
                    ];

                    // Calculate determinant (area) - can be used as a check
                    /*
                    elem.jacob_det = 0.5
                        * ((x[0] * y[1] - x[1] * y[0])
                            + (x[1] * y[2] - x[2] * y[1])
                            + (x[2] * y[3] - x[3] * y[2])
                            + (x[3] * y[0] - x[0] * y[3]))
                            .abs();
                    */
                    // Initialize 4D array for inverse transpose Jacobian at each quadrature point
                    // Dimensions: [xi_points, eta_points, 2, 2] for explicit matrix representation
                    elem.jacob_det = Array::zeros((ngp, ngp));
                    elem.jacob_inv_t = Array::zeros((ngp, ngp, 2, 2));

                    // Evaluate at each quadrature point
                    for i in 0..ngp {
                        let xi = basis.cell_gauss_points[i];

                        for j in 0..ngp {
                            let eta = basis.cell_gauss_points[j];

                            // Shape function derivatives at this quadrature point
                            // dN/dξ values
                            let dn_dxi = [
                                -0.25 * (1.0 - eta), // dN1/dξ
                                0.25 * (1.0 - eta),  // dN2/dξ
                                0.25 * (1.0 + eta),  // dN3/dξ
                                -0.25 * (1.0 + eta), // dN4/dξ
                            ];

                            // dN/dη values
                            let dn_deta = [
                                -0.25 * (1.0 - xi), // dN1/dη
                                -0.25 * (1.0 + xi), // dN2/dη
                                0.25 * (1.0 + xi),  // dN3/dη
                                0.25 * (1.0 - xi),  // dN4/dη
                            ];

                            // Calculate Jacobian components
                            let mut dx_dxi = 0.0;
                            let mut dx_deta = 0.0;
                            let mut dy_dxi = 0.0;
                            let mut dy_deta = 0.0;

                            for k in 0..4 {
                                dx_dxi += dn_dxi[k] * x[k];
                                dx_deta += dn_deta[k] * x[k];
                                dy_dxi += dn_dxi[k] * y[k];
                                dy_deta += dn_deta[k] * y[k];
                            }

                            // Calculate determinant at this point
                            let det_j = dx_dxi * dy_deta - dx_deta * dy_dxi;
                            elem.jacob_det[[i, j]] = det_j;
                            let inv_det = 1.0 / det_j;

                            // Compute inverse transpose Jacobian with 4D indexing
                            elem.jacob_inv_t[[i, j, 0, 0]] = dy_deta * inv_det;
                            elem.jacob_inv_t[[i, j, 0, 1]] = -dx_deta * inv_det;
                            elem.jacob_inv_t[[i, j, 1, 0]] = -dy_dxi * inv_det;
                            elem.jacob_inv_t[[i, j, 1, 1]] = dx_dxi * inv_det;
                        }
                    }
                }
                _ => {
                    panic!("Invalid number of edges for element");
                }
            }
        }
    }
}
