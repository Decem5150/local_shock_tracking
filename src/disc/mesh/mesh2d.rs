use super::mesh1d::Node;
use crate::disc::advection1d_space_time::boundary_condition::{BoundaryQuantity1d, BoundaryType};
use crate::disc::basis::lagrange1d::{LagrangeBasis1D, LagrangeBasis1DLobatto};
use ndarray::{Array, Array1, Array2, Array4, ArrayView1, Ix1, Ix2, Ix3, Ix4};

pub struct BoundaryPatch2d {
    pub iedges: Vec<usize>,
    pub boundary_type: BoundaryType,
    pub boundary_quantity: Option<BoundaryQuantity1d>,
}
#[derive(Clone)]
pub struct Edge {
    pub inodes: Vec<usize>,
    pub parents: Vec<usize>,
    pub local_ids: Vec<usize>,
}
#[derive(Clone)]
pub struct Element2d {
    pub inodes: Vec<usize>,
    pub iedges: Vec<usize>,
    pub ineighbors: Vec<usize>,
    /*
    pub jacob_det: Array<f64, Ix2>,
    pub jacob_inv_t: Array<f64, Ix4>,
    pub enriched_jacob_det: Array<f64, Ix2>,
    pub enriched_jacob_inv_t: Array<f64, Ix4>,
    */
}
pub struct SubMesh2d {
    pub nodes: Array<Node, Ix1>,
    pub edges: Array<Edge, Ix1>,
    pub elements: Array<Element2d, Ix1>,
}
#[derive(Clone)]
pub struct Mesh2d {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub elements: Vec<Element2d>,
    pub internal_edges: Vec<usize>,
    pub boundary_edges: Vec<usize>,
    pub free_x: Vec<usize>,
    pub interior_node_num: usize,
    pub elem_num: usize,
    pub node_num: usize,
}
impl Mesh2d {
    pub fn create_two_element_mesh(
        basis: &LagrangeBasis1DLobatto,
        enriched_basis: &LagrangeBasis1DLobatto,
    ) -> Mesh2d {
        let nodes = vec![
            Node {
                x: 0.0,
                y: 0.0,
                parents: vec![0],
                local_ids: vec![0],
            },
            Node {
                x: 1.0,
                y: 0.0,
                parents: vec![0, 1],
                local_ids: vec![1, 0],
            },
            Node {
                x: 2.0,
                y: 0.0,
                parents: vec![1],
                local_ids: vec![1],
            },
            Node {
                x: 2.0,
                y: 1.0,
                parents: vec![1],
                local_ids: vec![2],
            },
            Node {
                x: 0.8,
                y: 1.0,
                parents: vec![0, 1],
                local_ids: vec![2, 3],
            },
            Node {
                x: 0.0,
                y: 1.0,
                parents: vec![0],
                local_ids: vec![3],
            },
        ];
        let edges = vec![
            Edge {
                inodes: vec![0, 1],
                parents: vec![0],
                local_ids: vec![0],
            },
            Edge {
                inodes: vec![1, 2],
                parents: vec![1],
                local_ids: vec![0],
            },
            Edge {
                inodes: vec![2, 3],
                parents: vec![1],
                local_ids: vec![1],
            },
            Edge {
                inodes: vec![3, 4],
                parents: vec![1],
                local_ids: vec![2],
            },
            Edge {
                inodes: vec![4, 5],
                parents: vec![0],
                local_ids: vec![2],
            },
            Edge {
                inodes: vec![5, 0],
                parents: vec![0],
                local_ids: vec![3],
            },
            Edge {
                inodes: vec![1, 4],
                parents: vec![0, 1],
                local_ids: vec![1, 3],
            },
        ];
        let internal_edges = vec![6];
        let boundary_edges = vec![0, 1, 2, 3, 4, 5];
        let elements = vec![
            Element2d {
                inodes: vec![0, 1, 4, 5],
                iedges: vec![0, 6, 4, 5],
                ineighbors: vec![1],
            },
            Element2d {
                inodes: vec![1, 2, 3, 4],
                iedges: vec![1, 2, 3, 6],
                ineighbors: vec![0],
            },
        ];
        let free_x = vec![4];
        let interior_node_num = 0;
        let mesh = Mesh2d {
            nodes,
            edges,
            elements,
            internal_edges,
            boundary_edges,
            free_x,
            interior_node_num,
            elem_num: 2,
            node_num: 6,
        };
        mesh
    }
    /*
    fn compute_normal(&mut self) {
        for edge in self.edges.iter_mut() {
            let inodes: ArrayView1<usize> = edge.inodes.view();
            let x = [self.nodes[inodes[0]].x, self.nodes[inodes[1]].x];
            let y = [self.nodes[inodes[0]].y, self.nodes[inodes[1]].y];
            let normal = [y[1] - y[0], x[0] - x[1]];
            edge.normal = normal;
        }
    }
    pub fn evaluate_jacob(
        &self,
        basis: &LagrangeBasis1DLobatto,
        x: &[f64],
        y: &[f64],
    ) -> (Array2<f64>, Array4<f64>) {
        let ngp = basis.cell_gauss_points.len();
        let mut jacob_det = Array::zeros((ngp, ngp));
        let mut jacob_inv_t = Array::zeros((ngp, ngp, 2, 2));
        for i in 0..ngp {
            let eta = basis.cell_gauss_points[i];
            for j in 0..ngp {
                let xi = basis.cell_gauss_points[j];
                let dn_dxi = [
                    -0.25 * (1.0 - eta), // dN1/dξ
                    0.25 * (1.0 - eta),  // dN2/dξ
                    0.25 * (1.0 + eta),  // dN3/dξ
                    -0.25 * (1.0 + eta), // dN4/dξ
                ];
                let dn_deta = [
                    -0.25 * (1.0 - xi), // dN1/dη
                    -0.25 * (1.0 + xi), // dN2/dη
                    0.25 * (1.0 + xi),  // dN3/dη
                    0.25 * (1.0 - xi),  // dN4/dη
                ];
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
                jacob_det[[j, i]] = det_j;
                let inv_det = 1.0 / det_j;

                // Compute inverse transpose Jacobian with 4D indexing
                jacob_inv_t[[j, i, 0, 0]] = dy_deta * inv_det;
                jacob_inv_t[[j, i, 0, 1]] = -dy_dxi * inv_det;
                jacob_inv_t[[j, i, 1, 0]] = -dx_deta * inv_det;
                jacob_inv_t[[j, i, 1, 1]] = dx_dxi * inv_det;
            }
        }
        (jacob_det, jacob_inv_t)
    }
    fn compute_jacob(
        &mut self,
        basis: &LagrangeBasis1DLobatto,
        enriched_basis: &LagrangeBasis1DLobatto,
    ) {
        let ngp = basis.cell_gauss_points.len();
        let enriched_ngp = enriched_basis.cell_gauss_points.len();
        for elem in self.elements.iter_mut() {
            let inodes: ArrayView1<usize> = elem.inodes.view();
            let iedges: ArrayView1<usize> = elem.iedges.view();

            match iedges.len() {
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
                        let eta = basis.cell_gauss_points[i];

                        for j in 0..ngp {
                            let xi = basis.cell_gauss_points[j];

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
                            elem.jacob_inv_t[[i, j, 0, 1]] = -dy_dxi * inv_det;
                            elem.jacob_inv_t[[i, j, 1, 0]] = -dx_deta * inv_det;
                            elem.jacob_inv_t[[i, j, 1, 1]] = dx_dxi * inv_det;
                        }
                    }
                    // Evaluate at each enriched quadrature point
                    elem.enriched_jacob_det = Array::zeros((enriched_ngp, enriched_ngp));
                    elem.enriched_jacob_inv_t = Array::zeros((enriched_ngp, enriched_ngp, 2, 2));
                    for i in 0..enriched_ngp {
                        let eta = enriched_basis.cell_gauss_points[i];

                        for j in 0..enriched_ngp {
                            let xi = enriched_basis.cell_gauss_points[j];

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
                            elem.enriched_jacob_det[[i, j]] = det_j;
                            let inv_det = 1.0 / det_j;

                            // Compute inverse transpose Jacobian with 4D indexing
                            elem.enriched_jacob_inv_t[[i, j, 0, 0]] = dy_deta * inv_det;
                            elem.enriched_jacob_inv_t[[i, j, 0, 1]] = -dy_dxi * inv_det;
                            elem.enriched_jacob_inv_t[[i, j, 1, 0]] = -dx_deta * inv_det;
                            elem.enriched_jacob_inv_t[[i, j, 1, 1]] = dx_dxi * inv_det;
                        }
                    }
                }
                _ => {
                    panic!("Invalid number of edges for element");
                }
            }
        }
    }
    */
}
