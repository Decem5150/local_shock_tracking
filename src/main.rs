#![feature(autodiff)]
mod disc;
mod initialization;
mod io;
mod solver;
use disc::basis::triangle::TriangleBasis;
use disc::mesh::mesh2d::Mesh2d;
use initialization::initialize_enriched_basis;
use initialization::initialize_two_element_mesh2d;

use crate::initialization::initialize_basis;
use crate::initialization::initialize_mesh1d;
fn main() {
    let solver_params = initialization::initialize_params_advection();
    let basis = TriangleBasis::new(solver_params.polynomial_order);
    let enriched_basis = TriangleBasis::new(solver_params.polynomial_order + 1);
    let mut mesh = Mesh2d::create_eight_tri_mesh();
    let mut solver =
        initialization::initialize_tri_solver(&mut mesh, basis, enriched_basis, &solver_params);
    solver.solve();
}
