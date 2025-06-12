mod flux;

use faer::{Col, Mat, linalg::solvers::DenseSolveCore, mat, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use flux::burgers1d_space_time_flux;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, array, s};
use ndarray_stats::QuantileExt;
use std::autodiff::autodiff_reverse;

use super::{
    basis::triangle::TriangleBasis,
    mesh::mesh2d::{Mesh2d, TriangleElement},
};
use crate::solver::SolverParameters;

fn compute_normal(x0: f64, y0: f64, x1: f64, y1: f64) -> [f64; 2] {
    // normalized normal vector
    let normal = [y1 - y0, x0 - x1];
    let normal_magnitude = (normal[0].powi(2) + normal[1].powi(2)).sqrt();
    [normal[0] / normal_magnitude, normal[1] / normal_magnitude]
}
fn compute_ref_normal(local_id: usize) -> [f64; 2] {
    match local_id {
        0 => {
            // Bottom edge: from (0,0) to (1,0)
            // Outward normal points downward
            [0.0, -1.0]
        }
        1 => {
            // Hypotenuse edge: from (1,0) to (0,1)
            // Edge vector: (-1, 1), normal: (1, 1) normalized
            let sqrt2_inv = 1.0 / (2.0_f64.sqrt());
            [sqrt2_inv, sqrt2_inv]
        }
        2 => {
            // Left edge: from (0,1) to (0,0)
            // Outward normal points leftward
            [-1.0, 0.0]
        }
        _ => {
            panic!("Invalid edge ID");
        }
    }
}
fn compute_ref_edge_length(local_id: usize) -> f64 {
    match local_id {
        0 => 2.0,
        1 => 2.0 * 2.0_f64.sqrt(),
        2 => 2.0,
        _ => panic!("Invalid edge ID"),
    }
}
pub struct Disc1dBurgers1dSpaceTime<'a> {
    basis: TriangleBasis,
    enriched_basis: TriangleBasis,
    interp_node_to_cubature: Array2<f64>,
    interp_node_to_enriched_cubature: Array2<f64>,
    interp_node_to_enriched_quadrature: Array2<f64>,
    pub mesh: &'a mut Mesh2d<TriangleElement>,
    solver_param: &'a SolverParameters,
}
impl<'a> Disc1dBurgers1dSpaceTime<'a> {}
