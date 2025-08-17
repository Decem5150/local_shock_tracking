use std::f64::consts::PI;

use local_shock_tracking::disc::hoist::HOIST;
use ndarray::Array2;

use local_shock_tracking::disc::boundary::scalar1d::burgers_bnd_condition_2;
use local_shock_tracking::disc::dg_basis::lagrange1d::LobattoBasis;
use local_shock_tracking::disc::dg_basis::quadrilateral::QuadrilateralBasis;
use local_shock_tracking::disc::dg_basis::triangle::TriangleBasis;
// use local_shock_tracking::disc::burgers1d::Disc1dBurgers;
use local_shock_tracking::disc::burgers1d_space_time::Disc1dBurgers1dSpaceTime;
use local_shock_tracking::disc::geometric::Geometric2D;
use local_shock_tracking::disc::mesh::mesh2d::Mesh2d;
use local_shock_tracking::disc::space_time_1d_scalar::SpaceTime1DScalar;
use local_shock_tracking::initialization::{initialize_mesh1d, initialize_params};
use local_shock_tracking::io::write_to_vtu::write_nodal_solutions;
fn main() {
    /*
    let solver_params = initialization::initialize_params_by_file("inputs/solverparam.json");
    let space_basis = LobattoBasis::new(solver_params.polynomial_order);
    let time_basis = LobattoBasis::new(solver_params.polynomial_order);
    let space_time_basis = QuadrilateralBasis::new(solver_params.polynomial_order);

    println!("Enter the number of nodes: ");
    let mut node_num_str = String::new();
    std::io::stdin()
        .read_line(&mut node_num_str)
        .expect("Failed to read line");

    let node_num: usize = node_num_str.trim().parse().expect("Please type a number!");

    let mesh = initialize_mesh1d(node_num, -1.0, 1.0);

    let mut disc = Disc1dBurgers::new(
        space_basis,
        time_basis,
        space_time_basis,
        &mesh,
        &solver_params,
    );
    let mut solutions = Array2::<f64>::zeros((mesh.elem_num, disc.space_basis.xi.len()));
    disc.initialize_solution(solutions.view_mut(), &|x| -(PI * x).sin());
    disc.solve(solutions.view_mut());
    */
    let solver_params = initialize_params();
    let basis = TriangleBasis::new(solver_params.polynomial_order);
    let enriched_basis = TriangleBasis::new(solver_params.polynomial_order + 1);
    let mut mesh =
        Mesh2d::create_tri_mesh(5, 5, 0.0, 1.0, 0.0, 0.5, solver_params.polynomial_order);
    // dbg!(&mesh.boundaries.function);
    // dbg!(&mesh.boundaries.constant);
    // dbg!(&burgers_bnd_condition_2(-0.2, 0.0));
    // dbg!(&burgers_bnd_condition_2(-0.5, 0.0));
    let disc = Disc1dBurgers1dSpaceTime::new(basis, enriched_basis, &solver_params);
    let mut solutions = Array2::<f64>::zeros((mesh.elements.len(), disc.basis.xi.len()));

    disc.initialize_solution(solutions.view_mut());
    disc.solve(&mut mesh, &mut solutions);
}
