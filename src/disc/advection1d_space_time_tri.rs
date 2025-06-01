mod flux;

use faer::{Col, Mat, linalg::solvers::DenseSolveCore, mat, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use flux::space_time_flux1d;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, array, s};
use ndarray_stats::QuantileExt;
use std::autodiff::autodiff_reverse;

use super::{
    basis::triangle::TriangleBasis,
    mesh::mesh2d::{Mesh2d, TriangleElement},
};
use crate::solver::SolverParameters;
fn compute_normal(x0: f64, y0: f64, x1: f64, y1: f64) -> [f64; 2] {
    [y1 - y0, x0 - x1]
}

pub struct Disc1dAdvectionSpaceTimeTri<'a> {
    basis: TriangleBasis,
    enriched_basis: TriangleBasis,
    pub interp_to_cubature: Array2<f64>,
    //pub interp_to_quadrature: Array2<f64>,
    //pub interp_to_enriched: Array2<f64>,
    pub mesh: &'a mut Mesh2d<TriangleElement>,
    solver_param: &'a SolverParameters,
    advection_speed: f64,
}
fn evaluate_jacob(_xi: f64, _eta: f64, x: &[f64], y: &[f64]) -> (f64, [f64; 4]) {
    // For triangular elements with area coordinates (xi, eta)
    // The third coordinate is zeta = 1 - xi - eta
    // Shape functions for linear triangle:
    // N1 = 1 - xi - eta (node 0)
    // N2 = xi            (node 1)
    // N3 = eta           (node 2)

    let dn_dxi = [
        -1.0, // dN1/dξ
        1.0,  // dN2/dξ
        0.0,  // dN3/dξ
    ];
    let dn_deta = [
        -1.0, // dN1/dη
        0.0,  // dN2/dη
        1.0,  // dN3/dη
    ];

    let mut dx_dxi = 0.0;
    let mut dx_deta = 0.0;
    let mut dy_dxi = 0.0;
    let mut dy_deta = 0.0;

    for k in 0..3 {
        dx_dxi += dn_dxi[k] * x[k];
        dx_deta += dn_deta[k] * x[k];
        dy_dxi += dn_dxi[k] * y[k];
        dy_deta += dn_deta[k] * y[k];
    }

    let jacob_det = dx_dxi * dy_deta - dx_deta * dy_dxi;
    let jacob_inv_t = [
        dy_deta / jacob_det,
        -dy_dxi / jacob_det,
        -dx_deta / jacob_det,
        dx_dxi / jacob_det,
    ];

    (jacob_det, jacob_inv_t)
}
impl<'a> Disc1dAdvectionSpaceTimeTri<'a> {
    pub fn new(
        basis: TriangleBasis,
        enriched_basis: TriangleBasis,
        mesh: &'a mut Mesh2d<TriangleElement>,
        solver_param: &'a SolverParameters,
    ) -> Disc1dAdvectionSpaceTimeTri<'a> {
        let interp_to_cubature = Self::compute_interp_matrix(
            solver_param.polynomial_order,
            basis.inv_vandermonde.view(),
            basis.cub_r.view(),
            basis.cub_s.view(),
        );
        Disc1dAdvectionSpaceTimeTri {
            basis,
            enriched_basis,
            interp_to_cubature,
            mesh,
            solver_param,
            advection_speed: 0.1,
        }
    }
    fn compute_interp_matrix(
        n: usize,
        inv_vandermonde: ArrayView2<f64>,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
    ) -> Array2<f64> {
        let v = TriangleBasis::vandermonde2d(n, r, s);
        println!("v: {:?}", v);
        println!("inv_vandermonde: {:?}", inv_vandermonde);
        let interp_matrix = v.dot(&inv_vandermonde);
        println!("interp_matrix: {:?}", interp_matrix);
        interp_matrix
    }
}
