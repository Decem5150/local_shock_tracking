mod disc;
mod initialization;
mod io;
mod solver;
use crate::initialization::initialize_basis;
use crate::initialization::initialize_mesh1d;
fn main() {
    let (flow_params, solver_params) = initialization::initialize_params();
    let basis = initialize_basis(solver_params.cell_gp_num);
    let node_num = 81;
    let left_coord = -1.0;
    let right_coord = 1.0;
    let mesh = initialize_mesh1d(node_num, left_coord, right_coord);
    let mut solver = initialization::initialize_solver(&mesh, basis, &flow_params, &solver_params);
    solver.solve();
}
