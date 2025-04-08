use super::mesh1d::Node;
use crate::disc::advection1d_space_time::boundary_condition::{BoundaryQuantity1d, BoundaryType};
use crate::disc::basis::lagrange1d::{LagrangeBasis1D, LagrangeBasis1DLobatto};
use ndarray::{Array, Array1, ArrayView1, Ix1, Ix2, Ix3, Ix4};

pub struct BoundaryPatch2d {
    pub iedges: Array<usize, Ix1>,
    pub boundary_type: BoundaryType,
    pub boundary_quantity: Option<BoundaryQuantity1d>,
}
pub struct Edge {
    pub inodes: Array<usize, Ix1>,
    pub parent_elements: Array<usize, Ix1>,
    pub local_ids: Array<usize, Ix1>,
    pub normal: [f64; 2],
}
pub struct Element2d {
    pub inodes: Array<usize, Ix1>,
    pub iedges: Array<usize, Ix1>,
    pub ineighbors: Array<isize, Ix1>,
    pub jacob_det: Array<f64, Ix2>,
    pub jacob_inv_t: Array<f64, Ix4>,
    pub enriched_jacob_det: Array<f64, Ix2>,
    pub enriched_jacob_inv_t: Array<f64, Ix4>,
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
    // pub boundary_patches: Array<BoundaryPatch2d, Ix1>,
    pub elem_num: usize,
    pub node_num: usize,
}
impl Mesh2d {
    pub fn create_two_element_mesh(
        basis: &LagrangeBasis1DLobatto,
        enriched_basis: &LagrangeBasis1DLobatto,
    ) -> Mesh2d {
        let cell_gp_num = basis.cell_gauss_points.len();
        let enriched_cell_gp_num = enriched_basis.cell_gauss_points.len();
        let nodes = vec![
            Node {
                x: 0.0,
                y: 0.0,
                parent_elements: Array1::zeros(2),
                local_ids: Array1::zeros(2),
            },
            Node {
                x: 1.0,
                y: 0.0,
                parent_elements: Array1::zeros(2),
                local_ids: Array1::zeros(2),
            },
            Node {
                x: 2.0,
                y: 0.0,
                parent_elements: Array1::zeros(2),
                local_ids: Array1::zeros(2),
            },
            Node {
                x: 2.0,
                y: 1.0,
                parent_elements: Array1::zeros(2),
                local_ids: Array1::zeros(2),
            },
            Node {
                x: 1.1,
                y: 1.0,
                parent_elements: Array1::zeros(2),
                local_ids: Array1::zeros(2),
            },
            Node {
                x: 0.0,
                y: 1.0,
                parent_elements: Array1::zeros(2),
                local_ids: Array1::zeros(2),
            },
        ];
        let nodes = Array::from_vec(nodes);
        let edges = vec![
            Edge {
                inodes: Array1::from_vec(vec![0, 1]),
                parent_elements: Array1::from_vec(vec![0]),
                local_ids: Array1::zeros(1),
                normal: [0.0, 0.0],
            },
            Edge {
                inodes: Array1::from_vec(vec![1, 2]),
                parent_elements: Array1::from_vec(vec![1]),
                local_ids: Array1::zeros(1),
                normal: [0.0, 0.0],
            },
            Edge {
                inodes: Array1::from_vec(vec![2, 3]),
                parent_elements: Array1::from_vec(vec![1]),
                local_ids: Array1::zeros(1),
                normal: [0.0, 0.0],
            },
            Edge {
                inodes: Array1::from_vec(vec![3, 4]),
                parent_elements: Array1::from_vec(vec![1]),
                local_ids: Array1::zeros(1),
                normal: [0.0, 0.0],
            },
            Edge {
                inodes: Array1::from_vec(vec![4, 5]),
                parent_elements: Array1::from_vec(vec![0]),
                local_ids: Array1::zeros(1),
                normal: [0.0, 0.0],
            },
            Edge {
                inodes: Array1::from_vec(vec![5, 0]),
                parent_elements: Array1::from_vec(vec![0]),
                local_ids: Array1::zeros(1),
                normal: [0.0, 0.0],
            },
            Edge {
                inodes: Array1::from_vec(vec![1, 4]),
                parent_elements: Array1::from_vec(vec![0, 1]),
                local_ids: Array1::from_vec(vec![1, 3]),
                normal: [0.0, 0.0],
            },
        ];
        let edges = Array::from_vec(edges);
        let internal_edges = Array::from_vec(vec![6]);
        let elements = Array::from_vec(vec![
            Element2d {
                inodes: Array1::from_vec(vec![0, 1, 4, 5]),
                iedges: Array1::from_vec(vec![0, 6, 4, 5]),
                ineighbors: Array1::zeros(4),
                jacob_det: Array::zeros((cell_gp_num, cell_gp_num)),
                jacob_inv_t: Array::zeros((cell_gp_num, cell_gp_num, 2, 2)),
                enriched_jacob_det: Array::zeros((enriched_cell_gp_num, enriched_cell_gp_num)),
                enriched_jacob_inv_t: Array::zeros((
                    enriched_cell_gp_num,
                    enriched_cell_gp_num,
                    2,
                    2,
                )),
            },
            Element2d {
                inodes: Array1::from_vec(vec![1, 2, 3, 4]),
                iedges: Array1::from_vec(vec![1, 2, 3, 6]),
                ineighbors: Array1::zeros(4),
                jacob_det: Array::zeros((cell_gp_num, cell_gp_num)),
                jacob_inv_t: Array::zeros((cell_gp_num, cell_gp_num, 2, 2)),
                enriched_jacob_det: Array::zeros((enriched_cell_gp_num, enriched_cell_gp_num)),
                enriched_jacob_inv_t: Array::zeros((
                    enriched_cell_gp_num,
                    enriched_cell_gp_num,
                    2,
                    2,
                )),
            },
        ]);
        let mut mesh = Mesh2d {
            nodes,
            edges,
            elements,
            internal_edges,
            elem_num: 2,
            node_num: 6,
        };
        mesh.compute_normal();
        mesh.compute_jacob(&basis, &enriched_basis);
        mesh
    }
    fn compute_normal(&mut self) {
        for edge in self.edges.iter_mut() {
            let inodes: ArrayView1<usize> = edge.inodes.view();
            let x = [self.nodes[inodes[0]].x, self.nodes[inodes[1]].x];
            let y = [self.nodes[inodes[0]].y, self.nodes[inodes[1]].y];
            let normal = [y[1] - y[0], x[0] - x[1]];
            edge.normal = normal;
        }
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
}
