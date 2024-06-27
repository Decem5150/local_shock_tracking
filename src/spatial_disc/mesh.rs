use ndarray::{Array, Ix1};

pub mod mesh1d;

pub struct Vertex {
    pub x: f64,
    pub y: f64,
    pub iedges: Array<usize, Ix1>,
    pub in_edge_indices: Array<usize, Ix1>,
}
pub enum BoundaryType {
    Wall,
    FarField,
}