#![feature(autodiff)]
mod disc;
mod initialization;
mod io;
mod solver;
use initialization::initialize_enriched_basis;
use initialization::initialize_two_element_mesh2d;

use crate::initialization::initialize_basis;
use crate::initialization::initialize_mesh1d;
fn main() {
    let solver_params = initialization::initialize_params_advection();
    let basis = initialize_basis(solver_params.cell_gp_num);
    let enriched_basis = initialize_enriched_basis(solver_params.cell_gp_num + 1);
    let mesh = initialize_two_element_mesh2d(&basis, &enriched_basis);
    let mut solver =
        initialization::initialize_solver(&mesh, basis, enriched_basis, &solver_params);
    solver.solve();
}
