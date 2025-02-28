use crate::disc::burgers1d::boundary_condition::{BoundaryQuantity1d, BoundaryType};

use ndarray::{Array, ArrayView1, Ix1};

pub struct BoundaryPatch1d {
    pub inode: usize,
    pub boundary_type: BoundaryType,
    pub boundary_quantity: Option<BoundaryQuantity1d>,
}
pub struct Node {
    pub x: f64,
    pub y: f64,
    pub parent_elements: Array<isize, Ix1>,
    pub local_ids: Array<isize, Ix1>,
}
pub struct Element1d {
    pub inodes: Array<usize, Ix1>,
    pub ineighbors: Array<isize, Ix1>,
    pub jacob_det: f64,
}
pub struct Mesh1d {
    pub nodes: Array<Node, Ix1>,
    pub elements: Array<Element1d, Ix1>,
    pub internal_nodes: Array<usize, Ix1>,
    pub internal_elements: Array<usize, Ix1>, // index of internal elements
    pub boundary_elements: Array<usize, Ix1>, // index of boundary elements
    pub boundary_patches: Array<BoundaryPatch1d, Ix1>,
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
            parent_elements: Array::from_vec(vec![-1, 0]),
            local_ids: Array::from_vec(vec![-1, 0]),
        });
        for i in 1..node_num as isize - 1 {
            nodes.push(Node {
                x: left_coord + i as f64 * dx,
                y: 0.0,
                parent_elements: Array::from_vec(vec![i - 1, i]),
                local_ids: Array::from_vec(vec![1, 0]),
            });
        }
        nodes.push(Node {
            x: right_coord,
            y: 0.0,
            parent_elements: Array::from_vec(vec![node_num as isize - 2, -1]),
            local_ids: Array::from_vec(vec![1, -1]),
        });
        let nodes = Array::from_vec(nodes);
        let elem_num = node_num - 1;
        let mut elements = Vec::new();
        elements.push(Element1d {
            inodes: Array::from_vec(vec![0, 1]),
            ineighbors: Array::from_vec(vec![-1, 1]),
            jacob_det: 0.0,
        });
        for i in 1..elem_num - 1 {
            elements.push(Element1d {
                inodes: Array::from_vec(vec![i, i + 1]),
                ineighbors: Array::from_vec(vec![i as isize - 1, i as isize + 1]),
                jacob_det: 0.0,
            });
        }
        elements.push(Element1d {
            inodes: Array::from_vec(vec![elem_num - 1, elem_num]),
            ineighbors: Array::from_vec(vec![elem_num as isize - 2, -1]),
            jacob_det: 0.0,
        });
        let elements = Array::from_vec(elements);
        let internal_nodes = Array::from_iter(1..node_num - 1);
        let internal_elements = Array::from_iter(0..elem_num - 1);
        let boundary_elements = Array::from_vec(vec![0, elem_num - 1]);
        let mut boundary_patches = vec![
            BoundaryPatch1d {
                inode: 0,
                boundary_type: BoundaryType::Dirichlet,
                boundary_quantity: Some(BoundaryQuantity1d { u: 0.0 }),
            },
            BoundaryPatch1d {
                inode: elem_num,
                boundary_type: BoundaryType::Dirichlet,
                boundary_quantity: Some(BoundaryQuantity1d { u: 1.0 }),
            },
        ];
        boundary_patches.push(BoundaryPatch1d {
            inode: elem_num,
            boundary_type: BoundaryType::Dirichlet,
            boundary_quantity: None,
        });
        let boundary_patches = Array::from_vec(boundary_patches);
        let mut mesh1d = Self {
            nodes,
            elements,
            internal_nodes,
            internal_elements,
            boundary_elements,
            boundary_patches,
            elem_num,
            node_num,
        };
        mesh1d.compute_jacob_det();
        mesh1d
    }
    pub fn compute_jacob_det(&mut self) {
        for elem in self.elements.iter_mut() {
            let inodes: ArrayView1<usize> = elem.inodes.view();
            let x0 = self.nodes[inodes[0]].x;
            let x1 = self.nodes[inodes[1]].x;
            elem.jacob_det = x1 - x0;
        }
    }
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
