use ndarray::{Array, Ix1};

pub mod mesh1d;


pub enum BoundaryType {
    Wall,
    FarField,
}
