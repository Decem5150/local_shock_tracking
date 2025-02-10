use super::{BoundaryType};
use crate::disc::basis::lagrange1d::LagrangeBasis1D;
use ndarray::{Array, ArrayView1, Ix1};

pub struct BoundaryQuantity1d {
    pub rho: f64,
    pub u: f64,
    pub p: f64,
}
pub struct BoundaryPatch1d {
    pub inode: usize,
    pub boundary_type: BoundaryType,
    pub boundary_quantity: Option<BoundaryQuantity1d>,
}
pub struct Node {
    pub x: f64,
    pub y: f64,
    pub parent_elements: Array<usize, Ix1>,
    pub local_ids: Array<usize, Ix1>,
}
pub struct Element1d {
    pub inodes: Array<usize, Ix1>,
    pub ineighbors: Array<isize, Ix1>,
    pub jacob_det: f64,
}
pub struct Mesh1d {
    pub nodes: Array<Node, Ix1>,
    pub elements: Array<Element1d, Ix1>,
    pub internal_node: Array<usize, Ix1>, 
    pub internal_elements: Array<usize, Ix1>, // index of internal elements
    pub boundary_elements: Array<usize, Ix1>, // index of boundary elements
    pub boundary_patches: Array<BoundaryPatch1d, Ix1>,
}
impl Mesh1d {
    pub fn compute_jacob_det(&mut self) {
        for &ielem in self.internal_elements.iter() {
            let inodes: ArrayView1<usize> = self.elements[ielem].inodes.view();
            let x0 = self.nodes[inodes[0]].x;
            let x1 = self.nodes[inodes[1]].x;
            self.elements[ielem].jacob_det = 0.5 * (x1 - x0);
        }
    }
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
}
