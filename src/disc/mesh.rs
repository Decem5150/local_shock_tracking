use ndarray::{Array, Ix1};

pub mod mesh1d;
pub mod mesh2d;


pub enum BoundaryType {
    Wall,
    FarField,
}
