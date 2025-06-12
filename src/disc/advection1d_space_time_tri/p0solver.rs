use ndarray::{Array2, ArrayView2, ArrayViewMut2};

use crate::disc::mesh::mesh2d::{Mesh2d, TriangleElement};

pub struct P0Solver<'a> {
    mesh: &'a Mesh2d<TriangleElement>,
    advection_speed: f64,
}

impl<'a> P0Solver<'a> {
    pub fn new(mesh: &'a Mesh2d<TriangleElement>, advection_speed: f64) -> Self {
        Self {
            mesh,
            advection_speed,
        }
    }
    pub fn solve(&self, solutions: &mut Array2<f64>) {}
    fn compute_residuals(&self, solutions: ArrayView2<f64>, residuals: ArrayViewMut2<f64>) {
        for &iedge in self.mesh.internal_edges.iter() {}
        for &iedge in self.mesh.boundary_edges.iter() {}
    }
}
