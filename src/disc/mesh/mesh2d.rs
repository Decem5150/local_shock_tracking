use crate::disc::boundary::{
    BoundaryPosition,
    scalar1d::{ConstantBoundary, OpenBoundary, PolynomialBoundary},
};

use super::mesh1d::Node;
use hashbrown::{HashMap, HashSet};
use ndarray::{Array1, ArrayView1, array};

#[derive(Clone, Debug)]
pub enum Status<T> {
    Active(T),
    Removed,
}
impl<T> Status<T> {
    pub fn into_inner(self) -> T {
        match self {
            Status::Active(v) => v,
            Status::Removed => unreachable!(),
        }
    }

    pub fn as_ref(&self) -> &T {
        match self {
            Status::Active(v) => v,
            Status::Removed => unreachable!(),
        }
    }

    pub fn as_mut(&mut self) -> &mut T {
        match self {
            Status::Active(v) => v,
            Status::Removed => unreachable!(),
        }
    }

    pub fn mark_as_removed(&mut self) {
        *self = Status::Removed;
    }
}
#[derive(Clone, Debug)]
pub struct FlowInBoundary {
    pub iedges: Vec<usize>,
    pub value: f64,
}
#[derive(Clone, Debug)]
pub struct FlowOutBoundary {
    pub iedges: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct Edge {
    pub inodes: Vec<usize>,
    pub parents: Vec<usize>,
    pub local_ids: Vec<usize>,
}

pub trait Element2d: std::fmt::Debug {
    fn inodes(&self) -> &[usize];
    fn iedges(&self) -> &[usize];
    fn ineighbors(&self) -> &Vec<usize>;
    fn evaluate_jacob(&self, eta: f64, xi: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]);
}
#[derive(Clone, Debug)]
pub struct QuadrilateralElement {
    pub inodes: [usize; 4],
    pub iedges: [usize; 4],
    pub ineighbors: Vec<usize>,
}
impl Element2d for QuadrilateralElement {
    fn inodes(&self) -> &[usize] {
        &self.inodes
    }
    fn iedges(&self) -> &[usize] {
        &self.iedges
    }
    fn ineighbors(&self) -> &Vec<usize> {
        &self.ineighbors
    }
    fn evaluate_jacob(&self, eta: f64, xi: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]) {
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
        let jacob_det = dx_dxi * dy_deta - dx_deta * dy_dxi;
        let jacob_inv_t = [
            dy_deta / jacob_det,
            -dy_dxi / jacob_det,
            -dx_deta / jacob_det,
            dx_dxi / jacob_det,
        ];
        (jacob_det, jacob_inv_t)
    }
}
#[derive(Clone, Debug)]
pub struct TriangleElement {
    pub inodes: [usize; 3],
    pub iedges: [usize; 3],
    pub ineighbors: Vec<usize>,
    pub original_area: f64,
}
impl Element2d for TriangleElement {
    fn inodes(&self) -> &[usize] {
        &self.inodes
    }
    fn iedges(&self) -> &[usize] {
        &self.iedges
    }
    fn ineighbors(&self) -> &Vec<usize> {
        &self.ineighbors
    }
    fn evaluate_jacob(&self, _eta: f64, _xi: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]) {
        let dn_dxi = [-1.0, 1.0, 0.0];
        let dn_deta = [-1.0, 0.0, 1.0];
        let mut dx_dxi = 0.0;
        let mut dx_deta = 0.0;
        let mut dy_dxi = 0.0;
        let mut dy_deta = 0.0;
        for i in 0..3 {
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
}
/*
pub struct SubMesh2d {
    pub nodes: Array<Node, Ix1>,
    pub edges: Array<Edge, Ix1>,
    pub elements: Array<Element2d, Ix1>,
}
*/
#[derive(Clone)]
pub struct Mesh2d<T: Element2d> {
    pub nodes: Vec<Status<Node>>,
    pub edges: Vec<Status<Edge>>,
    pub elements: Vec<Status<T>>,
    pub constant_bnds: Vec<ConstantBoundary>,
    pub polynomial_bnds: Vec<PolynomialBoundary>,
    pub open_bnds: Vec<OpenBoundary>,
    pub interior_edges: Vec<usize>,
    pub free_bnd_x: Vec<usize>,
    pub free_bnd_y: Vec<usize>,
    pub interior_nodes: Vec<usize>,
    pub elem_num: usize,
    pub node_num: usize,
}
impl<T: Element2d> Mesh2d<T> {
    pub fn update_node_coords(
        &mut self,
        new_to_old: &[usize],
        alpha: f64,
        delta_x: ArrayView1<f64>,
    ) {
        let n_nodes = self.node_num;

        for (new_idx, &delta_val) in delta_x.indexed_iter() {
            let old_dof_idx = new_to_old[new_idx];
            if old_dof_idx < n_nodes {
                // This is an X-DOF
                let node_idx = old_dof_idx;
                if let Status::Active(node) = &mut self.nodes[node_idx] {
                    node.x += alpha * delta_val;
                }
            } else {
                // This is a Y-DOF
                let node_idx = old_dof_idx - n_nodes;
                if let Status::Active(node) = &mut self.nodes[node_idx] {
                    node.y += alpha * delta_val;
                }
            }
        }
    }
    pub fn rearrange_node_dofs(&self) -> (Vec<usize>, Vec<usize>) {
        let n_nodes = self.node_num;
        let total_dofs = 2 * n_nodes;

        let mut boundary_free_x_nodes_sorted = self.free_bnd_x.clone();
        let mut boundary_free_y_nodes_sorted = self.free_bnd_y.clone();
        let mut interior_nodes_sorted = self.interior_nodes.clone();

        interior_nodes_sorted.sort();
        boundary_free_x_nodes_sorted.sort();
        boundary_free_y_nodes_sorted.sort();

        let num_interior = self.interior_nodes.len();
        let num_free_bnd_x = boundary_free_x_nodes_sorted.len();
        let num_free_bnd_y = boundary_free_y_nodes_sorted.len();

        let num_free_dofs = 2 * num_interior + num_free_bnd_x + num_free_bnd_y;
        let mut old_to_new = vec![num_free_dofs; total_dofs];
        let mut new_to_old = vec![0; num_free_dofs];

        let mut current_idx = 0;

        // Interior X DOFs
        for (i, &node_idx) in interior_nodes_sorted.iter().enumerate() {
            let old_idx = node_idx; // x-dof index is the node index
            old_to_new[old_idx] = current_idx + i;
            new_to_old[current_idx + i] = old_idx;
        }
        current_idx += num_interior;

        // Interior Y DOFs
        for (i, &node_idx) in interior_nodes_sorted.iter().enumerate() {
            let old_idx = n_nodes + node_idx; // y-dof index is offset by n_nodes
            old_to_new[old_idx] = current_idx + i;
            new_to_old[current_idx + i] = old_idx;
        }
        current_idx += num_interior;

        // Free Boundary X DOFs
        for (i, &node_idx) in boundary_free_x_nodes_sorted.iter().enumerate() {
            let old_idx = node_idx;
            old_to_new[old_idx] = current_idx + i;
            new_to_old[current_idx + i] = old_idx;
        }
        current_idx += num_free_bnd_x;

        // Free Boundary Y DOFs
        for (i, &node_idx) in boundary_free_y_nodes_sorted.iter().enumerate() {
            let old_idx = n_nodes + node_idx;
            old_to_new[old_idx] = current_idx + i;
            new_to_old[current_idx + i] = old_idx;
        }

        (old_to_new, new_to_old)
    }
    pub fn print_free_node_coords(&self) {
        use std::collections::BTreeSet;

        let mut free_nodes = BTreeSet::new();
        self.interior_nodes.iter().for_each(|&n| {
            free_nodes.insert(n);
        });
        self.free_bnd_x.iter().for_each(|&n| {
            free_nodes.insert(n);
        });
        self.free_bnd_y.iter().for_each(|&n| {
            free_nodes.insert(n);
        });

        println!("--- Free Node Coordinates ---");

        for &node_idx in free_nodes.iter() {
            if let Status::Active(node) = &self.nodes[node_idx] {
                let is_interior = self.interior_nodes.contains(&node_idx);
                let free_x = is_interior || self.free_bnd_x.contains(&node_idx);
                let free_y = is_interior || self.free_bnd_y.contains(&node_idx);

                let mut dof_str = Vec::new();
                if free_x {
                    dof_str.push("X");
                }
                if free_y {
                    dof_str.push("Y");
                }

                println!(
                    "Node {}: ({:.4}, {:.4}) -> Free DOFs: {}",
                    node_idx,
                    node.x,
                    node.y,
                    dof_str.join(", ")
                );
            }

            println!("-----------------------------");
        }
    }
}
/*
impl Mesh2d<QuadrilateralElement> {
    pub fn create_two_quad_mesh() -> Mesh2d<QuadrilateralElement> {
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
                x: 0.4,
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
        let interior_edges = vec![6];
        let boundary_edges = vec![0, 1, 2, 3, 4, 5];
        let elements: Vec<QuadrilateralElement> = vec![
            QuadrilateralElement {
                inodes: [0, 1, 4, 5],
                iedges: [0, 6, 4, 5],
                ineighbors: vec![1],
            },
            QuadrilateralElement {
                inodes: [1, 2, 3, 4],
                iedges: [1, 2, 3, 6],
                ineighbors: vec![0],
            },
        ];
        let free_bnd_x = vec![4];
        let free_bnd_y = vec![];
        let interior_nodes = vec![];
        let mesh = Mesh2d {
            nodes,
            edges,
            elements,
            constant_bnds: vec![],
            polynomial_bnds: vec![],
            lower_bnd: PolynomialBoundary {
                inodes: vec![],
                iedges: vec![],
                nodal_coeffs: Array1::zeros(0),

            },
            right_bnd: PolynomialBoundary {
                iedges: vec![],
                value: vec![],
            },
            upper_bnd: ConstantBoundary {
            free_bnd_x,
            free_bnd_y,
            interior_nodes,
            elem_num: 2,
            node_num: 6,
        };
        mesh
    }
}
*/
impl Mesh2d<TriangleElement> {
    pub fn create_eight_tri_mesh() -> Mesh2d<TriangleElement> {
        let nodes = vec![
            Status::Active(Node {
                x: 0.0,
                y: 0.0,
                parents: vec![0, 1],
                local_ids: vec![0, 0],
            }),
            Status::Active(Node {
                x: 0.5,
                y: 0.0,
                parents: vec![0, 2, 3],
                local_ids: vec![1, 0, 0],
            }),
            Status::Active(Node {
                x: 1.0,
                y: 0.0,
                parents: vec![2],
                local_ids: vec![1],
            }),
            Status::Active(Node {
                x: 0.0,
                y: 0.5,
                parents: vec![1, 4, 5],
                local_ids: vec![2, 0, 0],
            }),
            Status::Active(Node {
                x: 0.5,
                y: 0.5,
                parents: vec![0, 1, 3, 4, 6, 7],
                local_ids: vec![2, 1, 2, 1, 0, 0],
            }),
            Status::Active(Node {
                x: 1.0,
                y: 0.5,
                parents: vec![2, 3, 6],
                local_ids: vec![2, 1, 1],
            }),
            Status::Active(Node {
                x: 0.0,
                y: 1.0,
                parents: vec![5],
                local_ids: vec![2],
            }),
            Status::Active(Node {
                x: 0.5,
                y: 1.0,
                parents: vec![4, 5, 7],
                local_ids: vec![2, 1, 2],
            }),
            Status::Active(Node {
                x: 1.0,
                y: 1.0,
                parents: vec![6, 7],
                local_ids: vec![2, 1],
            }),
        ];

        let edges = vec![
            // Horizontal edges
            Status::Active(Edge {
                inodes: vec![0, 1],
                parents: vec![0],
                local_ids: vec![0],
            }), // 0
            Status::Active(Edge {
                inodes: vec![1, 2],
                parents: vec![2],
                local_ids: vec![0],
            }), // 1
            Status::Active(Edge {
                inodes: vec![4, 3],
                parents: vec![1, 4],
                local_ids: vec![1, 0],
            }), // 2
            Status::Active(Edge {
                inodes: vec![5, 4],
                parents: vec![3, 6],
                local_ids: vec![1, 0],
            }), // 3
            Status::Active(Edge {
                inodes: vec![7, 6],
                parents: vec![5],
                local_ids: vec![1],
            }), // 4
            Status::Active(Edge {
                inodes: vec![8, 7],
                parents: vec![7],
                local_ids: vec![1],
            }), // 5
            // Vertical edges
            Status::Active(Edge {
                inodes: vec![3, 0],
                parents: vec![1],
                local_ids: vec![2],
            }), // 6
            Status::Active(Edge {
                inodes: vec![1, 4],
                parents: vec![0, 3],
                local_ids: vec![1, 2],
            }), // 7
            Status::Active(Edge {
                inodes: vec![2, 5],
                parents: vec![2],
                local_ids: vec![1],
            }), // 8
            Status::Active(Edge {
                inodes: vec![6, 3],
                parents: vec![5],
                local_ids: vec![2],
            }), // 9
            Status::Active(Edge {
                inodes: vec![4, 7],
                parents: vec![4, 7],
                local_ids: vec![1, 2],
            }), // 10
            Status::Active(Edge {
                inodes: vec![5, 8],
                parents: vec![6],
                local_ids: vec![1],
            }), // 11
            // Diagonal edges
            Status::Active(Edge {
                inodes: vec![4, 0],
                parents: vec![0, 1],
                local_ids: vec![2, 0],
            }), // 12
            Status::Active(Edge {
                inodes: vec![5, 1],
                parents: vec![2, 3],
                local_ids: vec![2, 0],
            }), // 13
            Status::Active(Edge {
                inodes: vec![7, 3],
                parents: vec![4, 5],
                local_ids: vec![2, 0],
            }), // 14
            Status::Active(Edge {
                inodes: vec![8, 4],
                parents: vec![6, 7],
                local_ids: vec![2, 0],
            }), // 15
        ];

        let constant_bnds = vec![
            ConstantBoundary {
                inodes: vec![0, 1],
                iedges: vec![0],
                value: 2.0,
                position: BoundaryPosition::Lower,
            },
            ConstantBoundary {
                inodes: vec![1, 2],
                iedges: vec![1],
                value: 1.0,
                position: BoundaryPosition::Right,
            },
            ConstantBoundary {
                inodes: vec![2, 5, 8],
                iedges: vec![8, 11],
                value: 0.0,
                position: BoundaryPosition::Upper,
            },
            ConstantBoundary {
                inodes: vec![8, 7, 6],
                iedges: vec![5, 4],
                value: 0.0,
                position: BoundaryPosition::Left,
            },
            ConstantBoundary {
                inodes: vec![6, 3, 0],
                iedges: vec![9, 6],
                value: 2.0,
                position: BoundaryPosition::Lower,
            },
        ];

        let interior_edges = vec![2, 3, 7, 10, 12, 13, 14, 15];

        let elements = vec![
            Status::Active(TriangleElement {
                inodes: [0, 1, 4],
                iedges: [0, 7, 12],
                ineighbors: vec![3, 1],
                original_area: 0.125,
            }),
            Status::Active(TriangleElement {
                inodes: [0, 4, 3],
                iedges: [12, 2, 6],
                ineighbors: vec![0, 4],
                original_area: 0.125,
            }),
            Status::Active(TriangleElement {
                inodes: [1, 2, 5],
                iedges: [1, 8, 13],
                ineighbors: vec![3],
                original_area: 0.125,
            }),
            Status::Active(TriangleElement {
                inodes: [1, 5, 4],
                iedges: [13, 3, 7],
                ineighbors: vec![2, 6, 0],
                original_area: 0.125,
            }),
            Status::Active(TriangleElement {
                inodes: [3, 4, 7],
                iedges: [2, 10, 14],
                ineighbors: vec![1, 7, 5],
                original_area: 0.125,
            }),
            Status::Active(TriangleElement {
                inodes: [3, 7, 6],
                iedges: [14, 4, 9],
                ineighbors: vec![4],
                original_area: 0.125,
            }),
            Status::Active(TriangleElement {
                inodes: [4, 5, 8],
                iedges: [3, 11, 15],
                ineighbors: vec![3, 7],
                original_area: 0.125,
            }),
            Status::Active(TriangleElement {
                inodes: [4, 8, 7],
                iedges: [15, 5, 10],
                ineighbors: vec![6, 4],
                original_area: 0.125,
            }),
        ];
        let free_bnd_x = vec![7];
        let free_bnd_y = vec![3, 5];
        let interior_nodes = vec![4];
        let mesh = Mesh2d {
            nodes,
            edges,
            elements,
            constant_bnds,
            polynomial_bnds: vec![],
            open_bnds: vec![],
            interior_edges,
            free_bnd_x,
            free_bnd_y,
            interior_nodes,
            elem_num: 8,
            node_num: 9,
        };
        mesh
    }
    pub fn create_tri_mesh(
        x_num: usize,
        y_num: usize,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        n_order: usize,
    ) -> Mesh2d<TriangleElement> {
        let node_num = x_num * y_num;
        let mut nodes = Vec::with_capacity(node_num);
        let hx = (x1 - x0) / (x_num - 1) as f64;
        let hy = (y1 - y0) / (y_num - 1) as f64;

        for i in 0..y_num {
            for j in 0..x_num {
                nodes.push(Status::Active(Node {
                    x: x0 + j as f64 * hx,
                    y: y0 + i as f64 * hy,
                    parents: Vec::new(),
                    local_ids: Vec::new(),
                }));
            }
        }

        let n_quads_x = x_num - 1;
        let n_quads_y = y_num - 1;
        let elem_num = 2 * n_quads_x * n_quads_y;
        let mut elements = Vec::with_capacity(elem_num);

        for i in 0..n_quads_y {
            for j in 0..n_quads_x {
                let bl = i * x_num + j;
                let br = i * x_num + j + 1;
                let tl = (i + 1) * x_num + j;
                let tr = (i + 1) * x_num + j + 1;

                let p_bl = nodes[bl].as_ref();
                let p_br = nodes[br].as_ref();
                let p_tl = nodes[tl].as_ref();
                let p_tr = nodes[tr].as_ref();

                let area1 = 0.5
                    * ((p_br.x - p_bl.x) * (p_tr.y - p_bl.y)
                        - (p_tr.x - p_bl.x) * (p_br.y - p_bl.y));
                elements.push(Status::Active(TriangleElement {
                    inodes: [bl, br, tr],
                    iedges: [0; 3],
                    ineighbors: vec![],
                    original_area: area1,
                }));

                let area2 = 0.5
                    * ((p_tr.x - p_bl.x) * (p_tl.y - p_bl.y)
                        - (p_tl.x - p_bl.x) * (p_tr.y - p_bl.y));
                elements.push(Status::Active(TriangleElement {
                    inodes: [bl, tr, tl],
                    iedges: [0; 3],
                    ineighbors: vec![],
                    original_area: area2,
                }));
            }
        }

        for (elem_idx, element) in elements.iter().enumerate() {
            for (local_id, &inode) in element.as_ref().inodes.iter().enumerate() {
                nodes[inode].as_mut().parents.push(elem_idx);
                nodes[inode].as_mut().local_ids.push(local_id);
            }
        }

        let mut edges = Vec::new();
        let mut edge_map = std::collections::HashMap::new();
        for (elem_idx, element) in elements.iter_mut().enumerate() {
            let n = element.as_ref().inodes;
            let element_edges_nodes = [(n[0], n[1]), (n[1], n[2]), (n[2], n[0])];

            for (local_id, &(n1, n2)) in element_edges_nodes.iter().enumerate() {
                let key = (n1.min(n2), n1.max(n2));
                let edge_idx = *edge_map.entry(key).or_insert_with(|| {
                    let new_edge_idx = edges.len();
                    edges.push(Status::Active(Edge {
                        inodes: vec![n1, n2],
                        parents: Vec::new(),
                        local_ids: Vec::new(),
                    }));
                    new_edge_idx
                });

                element.as_mut().iedges[local_id] = edge_idx;

                edges[edge_idx].as_mut().parents.push(elem_idx);
                edges[edge_idx].as_mut().local_ids.push(local_id);
            }
        }

        for (elem_idx, element) in elements.clone().iter().enumerate() {
            let mut neighbors = Vec::new();
            for &iedge in &element.as_ref().iedges {
                for &parent_elem in &edges[iedge].as_ref().parents {
                    if parent_elem != elem_idx {
                        neighbors.push(parent_elem);
                    }
                }
            }
            elements[elem_idx].as_mut().ineighbors = neighbors;
        }

        let interior_edges: Vec<usize> = (0..edges.len())
            .filter(|&i| edges[i].as_ref().parents.len() == 2)
            .collect();

        let mut top_edges = Vec::new();
        for j in (0..x_num - 1).rev() {
            let n1 = (y_num - 1) * x_num + j;
            let n2 = (y_num - 1) * x_num + j + 1;
            top_edges.push(*edge_map.get(&(n1.min(n2), n1.max(n2))).unwrap());
        }

        let mut right_edges = Vec::new();
        for i in 0..y_num - 1 {
            let n1 = i * x_num + (x_num - 1);
            let n2 = (i + 1) * x_num + (x_num - 1);
            right_edges.push(*edge_map.get(&(n1.min(n2), n1.max(n2))).unwrap());
        }

        let mut left_edges = Vec::new();
        for i in (0..y_num - 1).rev() {
            let n1 = i * x_num;
            let n2 = (i + 1) * x_num;
            left_edges.push(*edge_map.get(&(n1.min(n2), n1.max(n2))).unwrap());
        }
        /*
        let mut bottom_edges = Vec::new();
        for j in 0..x_num - 1 {
            let n1 = j;
            let n2 = j + 1;
            bottom_edges.push(*edge_map.get(&(n1.min(n2), n1.max(n2))).unwrap());
        }
        */

        let mid_node_idx = x_num / 2;

        let mut bottom_edges_left = Vec::new();
        for j in 0..mid_node_idx {
            let n1 = j;
            let n2 = j + 1;
            bottom_edges_left.push(*edge_map.get(&(n1.min(n2), n1.max(n2))).unwrap());
        }

        let mut bottom_edges_right = Vec::new();
        for j in mid_node_idx..x_num - 1 {
            let n1 = j;
            let n2 = j + 1;
            bottom_edges_right.push(*edge_map.get(&(n1.min(n2), n1.max(n2))).unwrap());
        }

        let top_nodes: Vec<usize> = (0..x_num).rev().map(|j| (y_num - 1) * x_num + j).collect();

        let right_nodes: Vec<usize> = (0..y_num).map(|i| i * x_num + (x_num - 1)).collect();

        let left_nodes: Vec<usize> = (0..y_num).rev().map(|i| i * x_num).collect();

        // let bottom_nodes: Vec<usize> = (0..x_num).collect();

        let bottom_nodes_left: Vec<usize> = (0..=mid_node_idx).collect();
        let bottom_nodes_right: Vec<usize> = (mid_node_idx..x_num).collect();

        /*
        // test for lower boundary

        let lower_bnd_condition = PolynomialBoundary {
            inodes: bottom_nodes,
            iedges: bottom_edges,
            nodal_coeffs: array![2.0, 1.5, 1.0],
            position: BoundaryPosition::Lower,
        };
        */
        /*
        let lower_bnd_condition = PolynomialBoundary {
            inodes: bottom_nodes,
            iedges: bottom_edges,
            nodal_coeffs: Array1::zeros(n_order + 1),
            position: BoundaryPosition::Lower,
        };
        */

        let lower_left_bnd_condition = ConstantBoundary {
            inodes: bottom_nodes_left,
            iedges: bottom_edges_left,
            value: 0.6,
            position: BoundaryPosition::Lower,
        };
        let lower_right_bnd_condition = ConstantBoundary {
            inodes: bottom_nodes_right,
            iedges: bottom_edges_right,
            value: 0.4,
            position: BoundaryPosition::Lower,
        };

        let right_bnd_condition = ConstantBoundary {
            inodes: right_nodes,
            iedges: right_edges,
            value: 0.4,
            position: BoundaryPosition::Right,
        };

        let upper_bnd_condition = OpenBoundary {
            inodes: top_nodes,
            iedges: top_edges,
            position: BoundaryPosition::Upper,
        };

        let left_bnd_condition = ConstantBoundary {
            inodes: left_nodes,
            iedges: left_edges,
            value: 0.6,
            position: BoundaryPosition::Left,
        };

        let polynomial_bnds = vec![];
        let open_bnds = vec![upper_bnd_condition];
        let constant_bnds = vec![
            right_bnd_condition,
            left_bnd_condition,
            lower_left_bnd_condition,
            lower_right_bnd_condition,
        ];

        let mut interior_nodes = Vec::new();
        for i in 1..y_num - 1 {
            for j in 1..x_num - 1 {
                interior_nodes.push(i * x_num + j);
            }
        }

        let mut free_bnd_x = Vec::new();
        // Bottom boundary nodes (free in x)
        /*
        for j in 1..x_num - 1 {
            free_bnd_x.push(j);
        }
        */
        // Top boundary nodes (free in x)
        for j in 1..x_num - 1 {
            free_bnd_x.push((y_num - 1) * x_num + j);
        }

        let mut free_bnd_y = Vec::new();
        // Left boundary nodes (free in y)
        for i in 1..y_num - 1 {
            free_bnd_y.push(i * x_num);
        }
        // Right boundary nodes (free in y)
        for i in 1..y_num - 1 {
            free_bnd_y.push(i * x_num + (x_num - 1));
        }
        Mesh2d {
            nodes,
            edges,
            elements,
            constant_bnds,
            polynomial_bnds,
            open_bnds,
            interior_edges,
            free_bnd_x,
            free_bnd_y,
            interior_nodes,
            elem_num,
            node_num,
        }
    }
    pub fn collapse_small_elements(&mut self, min_area_ratio: f64, new_to_old: &mut Vec<usize>) {
        let mut node_boundary_count: HashMap<usize, usize> = HashMap::new();
        let mut boundary_nodes = HashSet::new();

        let mut count_node_on_boundary = |b_inodes: &[usize]| {
            let unique_inodes: HashSet<_> = b_inodes.iter().cloned().collect();
            for &n in &unique_inodes {
                *node_boundary_count.entry(n).or_insert(0) += 1;
                boundary_nodes.insert(n);
            }
        };

        self.constant_bnds
            .iter()
            .for_each(|b| count_node_on_boundary(&b.inodes));
        self.polynomial_bnds
            .iter()
            .for_each(|b| count_node_on_boundary(&b.inodes));
        self.open_bnds
            .iter()
            .for_each(|b| count_node_on_boundary(&b.inodes));

        loop {
            let mut element_to_collapse = None;
            let mut first_small_element_found = None;

            // Find an element to collapse
            // We scan for degenerate elements first, as they are the highest priority.
            // If none are found, we'll collapse the first "small" element we encounter.
            // This is done in a single pass to avoid re-scanning the element list.
            for (elem_idx, element) in self.elements.iter().enumerate() {
                if let Status::Active(element) = element {
                    let p = [
                        self.nodes[element.inodes[0]].as_ref(),
                        self.nodes[element.inodes[1]].as_ref(),
                        self.nodes[element.inodes[2]].as_ref(),
                    ];
                    let area = 0.5
                        * ((p[1].x - p[0].x) * (p[2].y - p[0].y)
                            - (p[2].x - p[0].x) * (p[1].y - p[0].y));
                    if elem_idx == 23 {
                        dbg!(&area);
                        dbg!(&element.original_area);
                    }
                    // A negative area indicates different winding order, but its magnitude is what matters.
                    if area <= 1e-12 {
                        println!("Found degenerate element: {}, area: {}", elem_idx, area);
                        element_to_collapse = Some(elem_idx);
                        break; // Found a degenerate element, collapse it immediately.
                    }

                    // If no small element has been found yet, and this one qualifies, store it.
                    // We continue scanning in case we find a fully degenerate element later.
                    let area_ratio = if element.original_area > 1e-12 {
                        area / element.original_area
                    } else {
                        1.0 // Avoid division by zero; treat as no change.
                    };
                    if first_small_element_found.is_none() && area_ratio < min_area_ratio {
                        println!(
                            "Found small element: {}, area ratio: {}, original area: {}, area: {}",
                            elem_idx, area_ratio, element.original_area, area
                        );
                        first_small_element_found = Some(elem_idx);
                    }
                }
            }

            // If we didn't find a degenerate element, use the first small one we found.
            if element_to_collapse.is_none() {
                element_to_collapse = first_small_element_found;
            }

            if let Some(elem_idx) = element_to_collapse {
                // Determine which edge to collapse and how
                let element = self.elements[elem_idx].as_ref();
                let p_indices = element.inodes;
                let p = [
                    self.nodes[p_indices[0]].as_ref(),
                    self.nodes[p_indices[1]].as_ref(),
                    self.nodes[p_indices[2]].as_ref(),
                ];

                // Find the shortest edge of the element. This is the edge we will collapse.
                let len_sq = [
                    (p[0].x - p[1].x).powi(2) + (p[0].y - p[1].y).powi(2),
                    (p[1].x - p[2].x).powi(2) + (p[1].y - p[2].y).powi(2),
                    (p[2].x - p[0].x).powi(2) + (p[2].y - p[0].y).powi(2),
                ];

                let (shortest_edge_local_idx, _) = len_sq
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                let (v1_idx, v2_idx, other_v_idx) = match shortest_edge_local_idx {
                    0 => (p_indices[0], p_indices[1], p_indices[2]),
                    1 => (p_indices[1], p_indices[2], p_indices[0]),
                    _ => (p_indices[2], p_indices[0], p_indices[1]),
                };

                let v1_is_bnd = boundary_nodes.contains(&v1_idx);
                let v2_is_bnd = boundary_nodes.contains(&v2_idx);

                // Decide which vertex to merge into which (`v_from_idx` -> `v_to_idx`)
                let (v_from_idx, v_to_idx) = match (v1_is_bnd, v2_is_bnd) {
                    // Case 1: One vertex is on the boundary, the other is not.
                    // Always collapse the interior vertex onto the boundary vertex.
                    (false, true) => (v1_idx, v2_idx),
                    (true, false) => (v2_idx, v1_idx),
                    // Case 2: Both vertices are on the boundary.
                    (true, true) => {
                        let c1 = node_boundary_count.get(&v1_idx).copied().unwrap_or(0);
                        let c2 = node_boundary_count.get(&v2_idx).copied().unwrap_or(0);
                        if c1 > c2 {
                            // v1 is a corner, v2 is not. Collapse v2 onto v1.
                            (v2_idx, v1_idx)
                        } else if c2 > c1 {
                            // v2 is a corner, v1 is not. Collapse v1 onto v2.
                            (v1_idx, v2_idx)
                        } else {
                            // Both are on the same number of boundaries (e.g., both on one line, or both corners).
                            // Collapse the shorter of the other two edges. This improves element quality.
                            let p_other = self.nodes[other_v_idx].as_ref();
                            let p1 = self.nodes[v1_idx].as_ref();
                            let p2 = self.nodes[v2_idx].as_ref();
                            let l_sq_1_other =
                                (p1.x - p_other.x).powi(2) + (p1.y - p_other.y).powi(2);
                            let l_sq_2_other =
                                (p2.x - p_other.x).powi(2) + (p2.y - p_other.y).powi(2);
                            if l_sq_1_other > l_sq_2_other {
                                (v2_idx, v1_idx)
                            } else {
                                (v1_idx, v2_idx)
                            }
                        }
                    }
                    // Case 3: Both vertices are in the interior.
                    // Use the "collapse to longest opposite edge" heuristic to improve quality.
                    (false, false) => {
                        let p_other = self.nodes[other_v_idx].as_ref();
                        let p1 = self.nodes[v1_idx].as_ref();
                        let p2 = self.nodes[v2_idx].as_ref();
                        let l_sq_1_other = (p1.x - p_other.x).powi(2) + (p1.y - p_other.y).powi(2);
                        let l_sq_2_other = (p2.x - p_other.x).powi(2) + (p2.y - p_other.y).powi(2);
                        if l_sq_1_other > l_sq_2_other {
                            (v2_idx, v1_idx)
                        } else {
                            (v1_idx, v2_idx)
                        }
                    }
                };

                // Perform the collapse and update tracking
                println!("Collapsing edge from {} to {}...", v_from_idx, v_to_idx);
                self.perform_edge_collapse(v_from_idx, v_to_idx, new_to_old);
                println!("Collapsed edge from {} to {}", v_from_idx, v_to_idx);
            } else {
                break;
            }
        }
    }
    fn perform_edge_collapse(
        &mut self,
        v_from_idx: usize,
        v_to_idx: usize,
        new_to_old: &mut Vec<usize>,
    ) {
        // --- 1. Identify affected and collapsed elements ---
        let parents_from: HashSet<usize> = self.nodes[v_from_idx]
            .as_ref()
            .parents
            .iter()
            .cloned()
            .collect();
        dbg!(&parents_from);
        let parents_to: HashSet<usize> = self.nodes[v_to_idx]
            .as_ref()
            .parents
            .iter()
            .cloned()
            .collect();
        dbg!(&parents_to);
        let elements_to_remove: HashSet<usize> =
            parents_from.intersection(&parents_to).cloned().collect();
        let affected_elements: HashSet<usize> = parents_from
            .symmetric_difference(&parents_to)
            .cloned()
            .collect();

        // --- 2. Find "from" and "to" edges that will be merged ---
        let mut from_edges = Vec::new();
        let mut to_edges = Vec::new();

        let mut edges_to_merge: HashMap<usize, usize> = HashMap::new();
        let mut edges_to_remove: HashSet<usize> = HashSet::new();

        for &elem_idx in &elements_to_remove {
            let element = &self.elements[elem_idx];
            let inodes = element.as_ref().inodes;
            if let (Some(local_from), Some(local_to)) = (
                inodes.iter().position(|&n| n == v_from_idx),
                inodes.iter().position(|&n| n == v_to_idx),
            ) {
                let local_other = 3 - local_from - local_to;

                // An edge connecting local nodes i and (i+1)%3 is element.iedges[i].
                // This helper finds the local edge index given two local node indices.
                let local_edge_idx = |v1: usize, v2: usize| -> usize {
                    if (v1 + 1) % 3 == v2 {
                        v1
                    } else if (v2 + 1) % 3 == v1 {
                        v2
                    } else {
                        // This case should not be reached for a valid triangle.
                        // The third vertex is needed to determine the edge.
                        // The edge opposite to vertex k is edge k.
                        3 - v1 - v2
                    }
                };

                let collapsed_edge_idx =
                    element.as_ref().iedges[local_edge_idx(local_from, local_to)];
                edges_to_remove.insert(collapsed_edge_idx);

                // The "from" edge connects the "from" vertex and the other vertex.
                let from_edge_local_idx = local_edge_idx(local_from, local_other);
                let from_edge_idx = element.as_ref().iedges[from_edge_local_idx];
                from_edges.push(from_edge_idx);

                edges_to_remove.insert(from_edge_idx);

                // The "to" edge connects the "to" vertex and the other vertex.
                let to_edge_local_idx = local_edge_idx(local_to, local_other);
                let to_edge_idx = element.as_ref().iedges[to_edge_local_idx];
                dbg!(&to_edge_idx);
                to_edges.push(to_edge_idx);

                edges_to_merge.insert(from_edge_idx, to_edge_idx);

                // Find the neighbors of the collapsed element across the "from" and "to" edges.
                let from_neighbor_opt = self.edges[from_edge_idx]
                    .as_ref()
                    .parents
                    .iter()
                    .find(|&&p| p != elem_idx)
                    .copied();
                let to_neighbor_opt = self.edges[to_edge_idx]
                    .as_ref()
                    .parents
                    .iter()
                    .find(|&&p| p != elem_idx)
                    .copied();
                // Update parents and local ids of the to edge
                if let (Some(collapsed_idx_in_to_edge), Some(kept_idx_in_from_edge)) = (
                    self.edges[to_edge_idx]
                        .as_ref()
                        .parents
                        .iter()
                        .position(|&p| p == elem_idx),
                    self.edges[from_edge_idx]
                        .as_ref()
                        .parents
                        .iter()
                        .position(|&p| p != elem_idx),
                ) {
                    self.edges[to_edge_idx].as_mut().parents[collapsed_idx_in_to_edge] =
                        self.edges[from_edge_idx].as_ref().parents[kept_idx_in_from_edge];
                    self.edges[to_edge_idx].as_mut().local_ids[collapsed_idx_in_to_edge] =
                        self.edges[from_edge_idx].as_ref().local_ids[kept_idx_in_from_edge];
                }

                // Update parents of the to node by taking symmetric difference of parents of the from and to nodes
                let mut new_parents: Vec<usize> = parents_from
                    .symmetric_difference(&parents_to)
                    .cloned()
                    .collect();
                new_parents.sort();
                self.nodes[v_to_idx].as_mut().parents = new_parents;

                // Update parents of the local_other node
                let other_node_idx = element.as_ref().inodes[local_other];
                let other_node = self.nodes[other_node_idx].as_mut();
                other_node.parents.retain(|&p| p != elem_idx);

                // Update the neighbors' connectivity.
                match (from_neighbor_opt, to_neighbor_opt) {
                    (Some(from_neighbor_idx), Some(to_neighbor_idx)) => {
                        // Case 1: Both edges are interior. The two neighbors become adjacent.
                        let merged_edge_local_idx_from = self.elements[from_neighbor_idx]
                            .as_ref()
                            .iedges
                            .iter()
                            .position(|&e| e == from_edge_idx)
                            .unwrap();

                        // Update `from_neighbor` to point to `to_neighbor` via `to_edge`.
                        self.elements[from_neighbor_idx].as_mut().iedges
                            [merged_edge_local_idx_from] = to_edge_idx;
                        if let Some(pos) = self.elements[from_neighbor_idx]
                            .as_ref()
                            .ineighbors
                            .iter()
                            .position(|&i| i == elem_idx)
                        {
                            self.elements[from_neighbor_idx].as_mut().ineighbors[pos] =
                                to_neighbor_idx;
                        }

                        // Update `to_neighbor` to point back to `from_neighbor`.
                        if let Some(pos) = self.elements[to_neighbor_idx]
                            .as_ref()
                            .ineighbors
                            .iter()
                            .position(|&i| i == elem_idx)
                        {
                            self.elements[to_neighbor_idx].as_mut().ineighbors[pos] =
                                from_neighbor_idx;
                        }
                    }
                    (Some(from_neighbor_idx), None) => {
                        // Case 2: Only from_edge is interior. `to_edge` is on the boundary.
                        // The neighbor across from_edge now borders the boundary.
                        if let Some(pos) = self.elements[from_neighbor_idx]
                            .as_ref()
                            .ineighbors
                            .iter()
                            .position(|&n| n == elem_idx)
                        {
                            let merged_edge_local_idx_from = self.elements[from_neighbor_idx]
                                .as_ref()
                                .iedges
                                .iter()
                                .position(|&e| e == from_edge_idx)
                                .unwrap();
                            self.elements[from_neighbor_idx].as_mut().iedges
                                [merged_edge_local_idx_from] = to_edge_idx;
                            self.elements[from_neighbor_idx]
                                .as_mut()
                                .ineighbors
                                .remove(pos);
                        }
                    }
                    (None, Some(_)) => {
                        // Case 3: Only to_edge is interior. `from_edge` is on the boundary.
                        // This is impossible since boundary nodes are never removed.
                        unreachable!();
                    }
                    (None, None) => {
                        // Case 4: Both are boundary edges. No neighbor connectivity to update.
                        unreachable!();
                    }
                }
            }
        }

        // --- 3. Update the nodes of the affected elements and edges. ---
        let mut affected_edges = HashSet::new();
        for &elem_idx in &affected_elements {
            let element = &mut self.elements[elem_idx];
            // We only need to update elements connected to the node we are removing.
            if let Some(local_from_pos) = element
                .as_ref()
                .inodes
                .iter()
                .position(|&n| n == v_from_idx)
            {
                // This element is connected to v_from_idx.
                // First, update its node list to point to v_to_idx.
                element.as_mut().inodes[local_from_pos] = v_to_idx;

                // The two edges connected to the old v_from_idx in this element are:
                let edge1_idx = element.as_ref().iedges[local_from_pos];
                let edge2_idx = element.as_ref().iedges[(local_from_pos + 2) % 3];

                affected_edges.insert(edge1_idx);
                affected_edges.insert(edge2_idx);
            }
        }

        // Update edges connected to v_from_idx.
        for &edge_idx in &affected_edges {
            let edge = &mut self.edges[edge_idx];
            if let Some(local_from_pos) = edge.as_ref().inodes.iter().position(|&n| n == v_from_idx)
            {
                edge.as_mut().inodes[local_from_pos] = v_to_idx;
            }
        }

        // --- 4. Remove the collapsed elements, edges, and nodes. ---
        let mut elements_to_remove_vec: Vec<usize> = elements_to_remove.iter().cloned().collect();
        elements_to_remove_vec.sort();
        elements_to_remove_vec.reverse();
        for &elem_idx in &elements_to_remove_vec {
            self.elements[elem_idx].mark_as_removed();
            new_to_old.retain(|&i| i != elem_idx);
            println!("Removed element: {}", elem_idx);
        }
        let mut edges_to_remove_vec: Vec<usize> = edges_to_remove.iter().cloned().collect();
        edges_to_remove_vec.sort();
        edges_to_remove_vec.reverse();
        for &edge_idx in &edges_to_remove_vec {
            self.interior_edges
                .retain(|edge| !edges_to_remove.contains(edge));
            self.polynomial_bnds.iter_mut().for_each(|bnd| {
                bnd.iedges.retain(|edge| !edges_to_remove.contains(edge));
            });
            self.constant_bnds.iter_mut().for_each(|bnd| {
                bnd.iedges.retain(|edge| !edges_to_remove.contains(edge));
            });
            self.open_bnds.iter_mut().for_each(|bnd| {
                bnd.iedges.retain(|edge| !edges_to_remove.contains(edge));
            });
            self.edges[edge_idx].mark_as_removed();
            println!("Removed edge: {}", edge_idx);
        }
        // Remove nodes from interior_nodes, free_bnd_x, free_bnd_y
        self.interior_nodes.retain(|&node| node != v_from_idx);
        self.free_bnd_x.retain(|&node| node != v_from_idx);
        self.free_bnd_y.retain(|&node| node != v_from_idx);

        self.polynomial_bnds.iter_mut().for_each(|bnd| {
            bnd.inodes.retain(|&node| node != v_from_idx);
        });
        self.constant_bnds.iter_mut().for_each(|bnd| {
            bnd.inodes.retain(|&node| node != v_from_idx);
        });
        self.open_bnds.iter_mut().for_each(|bnd| {
            bnd.inodes.retain(|&node| node != v_from_idx);
        });

        self.nodes[v_from_idx].mark_as_removed();
    }
}
