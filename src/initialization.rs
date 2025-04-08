use ndarray::Array3;

use crate::{
    disc::{
        basis::lagrange1d::{LagrangeBasis1D, LagrangeBasis1DLobatto},
        mesh::{mesh1d::Mesh1d, mesh2d::Mesh2d},
    },
    io::params_parser::SolverParamParser,
    solver::{FlowParameters, ShockTrackingSolver, Solver, SolverParameters},
};

pub fn initialize_params() -> (FlowParameters, SolverParameters) {
    let solver_param_parser = SolverParamParser::parse("inputs/solverparam.json");
    let polynomial_order = solver_param_parser.polynomial_order;
    let cell_gp_num = polynomial_order + 1;
    let solver_params = SolverParameters {
        cfl: solver_param_parser.cfl,
        final_time: solver_param_parser.final_time,
        final_step: solver_param_parser.final_step,
        polynomial_order,
        cell_gp_num,
        equation_num: 1,
    };
    let flow_params = FlowParameters { hcr: 1.4 };
    (flow_params, solver_params)
}
pub fn initialize_params_advection() -> SolverParameters {
    let polynomial_order = 2;
    let cell_gp_num = polynomial_order + 1;
    let solver_params = SolverParameters {
        cfl: 1.0,
        final_time: 0.0,
        final_step: 10,
        polynomial_order,
        cell_gp_num,
        equation_num: 1,
    };
    solver_params
}
pub fn initialize_basis(cell_gp_num: usize) -> LagrangeBasis1DLobatto {
    LagrangeBasis1DLobatto::new(cell_gp_num)
}
pub fn initialize_enriched_basis(enriched_cell_gp_num: usize) -> LagrangeBasis1DLobatto {
    LagrangeBasis1DLobatto::new(enriched_cell_gp_num)
}
pub fn initialize_mesh1d(node_num: usize, left_coord: f64, right_coord: f64) -> Mesh1d {
    Mesh1d::new(node_num, left_coord, right_coord)
}
pub fn initialize_two_element_mesh2d(
    basis: &LagrangeBasis1DLobatto,
    enriched_basis: &LagrangeBasis1DLobatto,
) -> Mesh2d {
    Mesh2d::create_two_element_mesh(basis, enriched_basis)
}
pub fn initialize_solver<'a>(
    mesh: &'a Mesh2d,
    basis: LagrangeBasis1DLobatto,
    enriched_basis: LagrangeBasis1DLobatto,
    solver_param: &'a SolverParameters,
) -> ShockTrackingSolver<'a> {
    let solver = ShockTrackingSolver::new(basis, enriched_basis, mesh, solver_param);
    solver
}
