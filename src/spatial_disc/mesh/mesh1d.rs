use hashbrown::HashMap;
use ndarray::{Array, Ix1, Ix2, Ix3};
pub struct Vertex {
    pub x: f64,
    pub y: f64,
}
pub enum BoundaryType {
    Wall,
    FarField,
}
pub struct BoundaryQuantity {
    pub rho: f64,
    pub u: f64,
    pub v: f64,
    pub p: f64,
}
pub struct Patch {
    pub iedges: Array<usize, Ix1>,
    pub boundary_type: BoundaryType,
    pub boundary_quantity: Option<BoundaryQuantity>,
}
pub struct Element1d {
    pub ivertices: Array<usize, Ix1>,
    pub ineighbours: Array<isize, Ix1>,
    pub dphis_cell_gps: Array<HashMap<(usize, usize), f64>, Ix2>,
    pub jacob_det: f64,
}
pub struct Mesh1d { 
    pub elements: Array<Element1d, Ix1>,
    pub internal_vertex_indices: Array<usize, Ix1>,
    pub internal_element_indices: Array<usize, Ix1>,
    pub boundary_element_indices: Array<usize, Ix1>,
    pub vertices: Array<Vertex, Ix1>,
    pub patches: Array<Patch, Ix1>,
}
