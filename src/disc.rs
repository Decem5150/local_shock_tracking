pub mod ader;
pub mod boundary;
pub mod cg_basis;
pub mod dg_basis;
pub mod hoist;
// pub mod flux;
pub mod finite_difference;
pub mod gauss_points;
pub mod geometric;
pub mod mesh;
// pub mod riemann_solver;
// pub mod advection1d_space_time_quad;
// pub mod advection1d_space_time_tri;
// pub mod burgers1d;
pub mod burgers1d_space_time;
pub mod linear_elliptic;
pub mod space_time_1d_scalar;
// pub mod space_time_1d_system;
// pub mod euler1d;
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};

use crate::disc::{
    geometric::Geometric2D,
    mesh::mesh2d::{Mesh2d, Status, TriangleElement},
    space_time_1d_scalar::SpaceTime1DScalar,
};
