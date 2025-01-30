use super::{BoundaryType};
use crate::disc::basis::lagrange1d::LagrangeBasis1D;
use hashbrown::HashMap;
use ndarray::{Array, Ix1, Ix2};

pub struct BoundaryQuantity1d {
    pub rho: f64,
    pub u: f64,
    pub p: f64,
}
pub struct BoundaryPatch1d {
    pub ivertex: usize,
    pub boundary_type: BoundaryType,
    pub boundary_quantity: Option<BoundaryQuantity1d>,
}
pub struct Node1d {
    pub x: f64,
    pub y: f64,
    pub parent_elements: Array<usize, Ix1>,
    pub local_ids: Array<usize, Ix1>,
}
pub struct Element1d {
    pub inodes: Array<usize, Ix1>,
    pub ineighbours: Array<isize, Ix1>,
    pub dphis_cell_gps: Array<HashMap<(usize, usize), f64>, Ix2>,
    pub jacob_det: f64,
}
pub struct Mesh1d {
    pub elements: Array<Element1d, Ix1>,
    pub internal_vertex_indices: Array<usize, Ix1>,
    pub internal_element_indices: Array<usize, Ix1>,
    pub boundary_element_indices: Array<usize, Ix1>,
    pub nodes: Array<Node1d, Ix1>,
    pub boundary_patches: Array<BoundaryPatch1d, Ix1>,
}
impl Mesh1d {
    pub fn compute_jacob_det(&mut self) {
        for ielem in self.internal_element_indices.iter() {
            let ivertices = &self.elements[*ielem].ivertices;
            let x0 = self.nodes[ivertices[0]].x;
            let x1 = self.nodes[ivertices[1]].x;
            self.elements[*ielem].jacob_det = 0.5 * (x1 - x0);
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
