mod solver;
mod disc;
mod temporal_disc;
mod initialization;
mod io;
fn main() {
    let (flow_params, solver_params) = initialization::initialize_params();
    let mut solver = initialization::initialize_solver(&mesh, &basis_function, &gauss_points, &flow_param, &mesh_param, &solver_param);
}
