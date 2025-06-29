#![feature(autodiff)]
mod disc;
mod initialization;
mod io;
mod solver;
use disc::basis::triangle::TriangleBasis;
use disc::mesh::mesh2d::Mesh2d;

use initialization::initialize_two_element_mesh2d;

use crate::disc::basis::lagrange1d::LobattoBasis;
use crate::disc::basis::quadrilateral::QuadrilateralBasis;
use crate::initialization::initialize_mesh1d;
use crate::solver::Solver;
fn main() {
    let solver_params = initialization::initialize_params_by_file("inputs/solverparam.json");
    // let basis = TriangleBasis::new(solver_params.polynomial_order);
    // let enriched_basis = TriangleBasis::new(solver_params.polynomial_order + 1);
    let space_basis = LobattoBasis::new(solver_params.polynomial_order);
    let time_basis = LobattoBasis::new(solver_params.polynomial_order);
    let space_time_basis = QuadrilateralBasis::new(solver_params.polynomial_order);
    // let mut mesh = Mesh2d::create_eight_tri_mesh();
    let mesh = initialize_mesh1d(40, -1.0, 1.0);
    // let mut solver =
    //    initialization::initialize_tri_solver(&mut mesh, basis, enriched_basis, &solver_params);
    let mut solver = Solver::new(
        space_basis,
        time_basis,
        space_time_basis,
        &mesh,
        &solver_params,
    );
    solver.solve();
}
