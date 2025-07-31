pub mod scalar1d;
pub mod system1d;
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryPosition {
    Lower,
    Right,
    Upper,
    Left,
}
