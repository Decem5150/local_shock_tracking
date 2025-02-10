use super::{mesh1d::Node, BoundaryType};
use crate::disc::basis::lagrange1d::LagrangeBasis1D;
use ndarray::{Array, Ix1};

pub struct Edge {
    pub inodes: Array<usize, Ix1>,
    pub parent_elements: Array<usize, Ix1>,
    pub local_ids: Array<usize, Ix1>,
}
pub struct Element2d {
    pub inodes: Array<usize, Ix1>,
    pub edges: Array<Edge, Ix1>,
    pub ineighbors: Array<isize, Ix1>,
    pub jacob_det: f64,
}
pub struct SubMesh2d {
    pub nodes: Array<Node, Ix1>,
    pub edges: Array<Edge, Ix1>,
    pub elements: Array<Element2d, Ix1>,


}
