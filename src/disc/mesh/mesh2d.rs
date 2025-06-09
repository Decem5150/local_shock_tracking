use super::mesh1d::Node;
use crate::disc::basis::lagrange1d::{LagrangeBasis1D, LagrangeBasis1DLobatto};
use ndarray::{Array, Array1, Array2, Array4, ArrayView1, Ix1, Ix2, Ix3, Ix4};

#[derive(Clone, Debug)]
pub struct FlowInBoundary {
    pub iedges: Vec<usize>,
    pub value: f64,
}
#[derive(Clone, Debug)]
pub struct FlowOutBoundary {
    pub iedges: Vec<usize>,
}
#[derive(Clone)]
pub struct Edge {
    pub inodes: Vec<usize>,
    pub parents: Vec<usize>,
    pub local_ids: Vec<usize>,
    pub ref_normal: [f64; 2],
}
impl Edge {
    pub fn compute_ref_tri_normal(&mut self) {
        let local_edge_id = self.local_ids[0];

        match local_edge_id {
            0 => {
                // Bottom edge: from (0,0) to (1,0)
                // Outward normal points downward
                self.ref_normal = [0.0, -1.0];
            }
            1 => {
                // Hypotenuse edge: from (1,0) to (0,1)
                // Edge vector: (-1, 1), normal: (1, 1) normalized
                let sqrt2_inv = 1.0 / (2.0_f64.sqrt());
                self.ref_normal = [sqrt2_inv, sqrt2_inv];
            }
            2 => {
                // Left edge: from (0,1) to (0,0)
                // Outward normal points leftward
                self.ref_normal = [-1.0, 0.0];
            }
            _ => {
                panic!("Invalid edge ID");
            }
        }
    }
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
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub elements: Vec<T>,
    pub flow_in_bnds: Vec<FlowInBoundary>,
    pub flow_out_bnds: Vec<FlowOutBoundary>,
    pub internal_edges: Vec<usize>,
    pub boundary_edges: Vec<usize>,
    pub free_coords: Vec<usize>,
    pub interior_node_num: usize,
    pub elem_num: usize,
    pub node_num: usize,
}
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
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![1, 2],
                parents: vec![1],
                local_ids: vec![0],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![2, 3],
                parents: vec![1],
                local_ids: vec![1],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![3, 4],
                parents: vec![1],
                local_ids: vec![2],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![4, 5],
                parents: vec![0],
                local_ids: vec![2],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![5, 0],
                parents: vec![0],
                local_ids: vec![3],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![1, 4],
                parents: vec![0, 1],
                local_ids: vec![1, 3],
                ref_normal: [0.0, 0.0],
            },
        ];
        let internal_edges = vec![6];
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
        let free_coords = vec![4];
        let interior_node_num = 0;
        let mesh = Mesh2d {
            nodes,
            edges,
            elements,
            flow_in_bnds: vec![],
            flow_out_bnds: vec![],
            internal_edges,
            boundary_edges,
            free_coords,
            interior_node_num,
            elem_num: 2,
            node_num: 6,
        };
        mesh
    }
}
impl Mesh2d<TriangleElement> {
    pub fn create_four_tri_mesh() -> Mesh2d<TriangleElement> {
        let nodes = vec![
            Node {
                x: 0.0,
                y: 0.0,
                parents: vec![0, 1],
                local_ids: vec![0, 0],
            },
            Node {
                x: 1.0,
                y: 0.0,
                parents: vec![0, 2, 3],
                local_ids: vec![1, 0, 0],
            },
            Node {
                x: 2.0,
                y: 0.0,
                parents: vec![2],
                local_ids: vec![1],
            },
            Node {
                x: 2.0,
                y: 1.0,
                parents: vec![2, 3],
                local_ids: vec![2, 1],
            },
            Node {
                x: 0.4,
                y: 1.0,
                parents: vec![0, 1, 3],
                local_ids: vec![2, 1, 2],
            },
            Node {
                x: 0.0,
                y: 1.0,
                parents: vec![1],
                local_ids: vec![2],
            },
        ];
        let mut edges = vec![
            Edge {
                inodes: vec![0, 1],
                parents: vec![0],
                local_ids: vec![0],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![1, 2],
                parents: vec![2],
                local_ids: vec![0],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![2, 3],
                parents: vec![2],
                local_ids: vec![1],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![3, 4],
                parents: vec![3],
                local_ids: vec![1],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![4, 5],
                parents: vec![1],
                local_ids: vec![1],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![5, 0],
                parents: vec![1],
                local_ids: vec![2],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![4, 0], // diagonal edge for left quadrilateral
                parents: vec![0, 1],
                local_ids: vec![2, 0],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![3, 1], // diagonal edge for right quadrilateral
                parents: vec![2, 3],
                local_ids: vec![2, 0],
                ref_normal: [0.0, 0.0],
            },
            Edge {
                inodes: vec![1, 4], // shared edge between triangles
                parents: vec![0, 3],
                local_ids: vec![1, 2],
                ref_normal: [0.0, 0.0],
            },
        ];
        for edge in edges.iter_mut() {
            edge.compute_ref_tri_normal();
        }
        let flow_in_bnds = vec![
            FlowInBoundary {
                iedges: vec![0, 5],
                value: 2.0,
            },
            FlowInBoundary {
                iedges: vec![1],
                value: 1.0,
            },
        ];
        let flow_out_bnds = vec![FlowOutBoundary {
            iedges: vec![2, 3, 4],
        }];
        let internal_edges = vec![6, 7, 8];
        let boundary_edges = vec![0, 1, 2, 3, 4, 5];
        let elements: Vec<TriangleElement> = vec![
            TriangleElement {
                inodes: [0, 1, 4],
                iedges: [0, 8, 6],
                ineighbors: vec![1, 3],
            },
            TriangleElement {
                inodes: [0, 4, 5],
                iedges: [6, 4, 5],
                ineighbors: vec![0],
            },
            TriangleElement {
                inodes: [1, 2, 3],
                iedges: [1, 2, 7],
                ineighbors: vec![3],
            },
            TriangleElement {
                inodes: [1, 3, 4],
                iedges: [7, 3, 8],
                ineighbors: vec![2, 0],
            },
        ];
        /*
        // print coordinates of nodes in each element
        for element in elements.iter() {
            println!("Element: {:?}", element);
            for inode in element.inodes.iter() {
                println!("Node: {:?}", nodes[*inode]);
            }
        }
        */
        let free_coords = vec![4];
        let interior_node_num = 0;

        let mesh = Mesh2d {
            nodes,
            edges,
            elements,
            flow_in_bnds,
            flow_out_bnds,
            internal_edges,
            boundary_edges,
            free_coords,
            interior_node_num,
            elem_num: 4,
            node_num: 6,
        };
        mesh
    }
}
