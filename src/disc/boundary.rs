pub mod scalar1d;

#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryPosition {
    Lower,
    Right,
    Upper,
    Left,
}
