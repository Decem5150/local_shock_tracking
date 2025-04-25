// use crate::disc::burgers1d::boundary_condition::{BoundaryQuantity1d, BoundaryType};

use ndarray::{Array, ArrayView1, Ix1};

// pub struct BoundaryPatch1d {
//     pub inode: usize,
//     pub boundary_type: BoundaryType,
//     pub boundary_quantity: Option<BoundaryQuantity1d>,
// }

pub struct Node {
    pub x: f64,
    pub y: f64,
    pub parents: Vec<usize>,
    pub local_ids: Vec<usize>,
}

pub struct Element1d {
    pub inodes: Vec<usize>,
    pub ineighbors: Vec<usize>,
    pub jacob_det: f64,
}
pub struct Mesh1d {
    pub nodes: Vec<Node>,
    pub elements: Vec<Element1d>,
    pub internal_nodes: Vec<usize>,
    pub boundary_nodes: Vec<usize>,
    pub elem_num: usize,
    pub node_num: usize,
}
impl Mesh1d {
    pub fn new(node_num: usize, left_coord: f64, right_coord: f64) -> Self {
        let dx = (right_coord - left_coord) / (node_num - 1) as f64;
        let mut nodes = Vec::new();
        nodes.push(Node {
            x: left_coord,
            y: 0.0,
            parents: vec![0],
            local_ids: vec![0],
        });
        for i in 1..node_num - 1 {
            nodes.push(Node {
                x: left_coord + i as f64 * dx,
                y: 0.0,
                parents: vec![i - 1, i],
                local_ids: vec![1, 0],
            });
        }
        nodes.push(Node {
            x: right_coord,
            y: 0.0,
            parents: vec![node_num - 2],
            local_ids: vec![1],
        });
        let elem_num = node_num - 1;
        let mut elements = Vec::new();
        elements.push(Element1d {
            inodes: vec![0, 1],
            ineighbors: vec![1],
            jacob_det: 0.0,
        });
        for i in 1..elem_num - 1 {
            elements.push(Element1d {
                inodes: vec![i, i + 1],
                ineighbors: vec![i - 1, i + 1],
                jacob_det: 0.0,
            });
        }
        elements.push(Element1d {
            inodes: vec![elem_num - 1, elem_num],
            ineighbors: vec![elem_num - 2],
            jacob_det: 0.0,
        });
        let internal_nodes = (1..node_num - 1).collect();

        let boundary_nodes = vec![0, elem_num];
        let mut mesh1d = Self {
            nodes,
            elements,
            internal_nodes,
            boundary_nodes,
            elem_num,
            node_num,
        };

        mesh1d
    }
    /*
    pub fn compute_jacob_det(&mut self) {
        for elem in self.elements.iter_mut() {
            let inodes: ArrayView1<usize> = elem.inodes.view();
            let x0 = self.nodes[inodes[0]].x;
            let x1 = self.nodes[inodes[1]].x;
            elem.jacob_det = x1 - x0;
        }
    }
    */
    /*
    pub fn compute_dphi(&mut self, basis: &LagrangeBasis1D, cell_ngp: usize) {
        let nbasis = cell_ngp;
        for element in self.elements.iter_mut() {
            let x0 = self.nodes[element.ivertices[0]].x;
            let x1 = self.nodes[element.ivertices[1]].x;
            for igp in 0..cell_ngp {
                for ibasis in 0..nbasis {
                    let dphi_dxi = basis.dphis_cell_gps[(igp, ibasis)].get(&1).unwrap();
                    element.dphis_cell_gps[(igp, ibasis)]
                        .insert((igp, ibasis), *dphi_dxi / element.jacob_det);
                }
            }
        }
    }
    */
}
