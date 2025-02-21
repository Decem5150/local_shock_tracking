use crate::{
    disc::{basis::lagrange1d::LagrangeBasis1D, burgers1d::Disc1dBurgers, mesh::mesh1d::Mesh1d},
    io::params_parser::SolverParamParser,
    solver::{FlowParameters, Solver, SolverParameters},
};

pub fn initialize_params() -> (FlowParameters, SolverParameters) {
    let solver_param_parser = SolverParamParser::parse("input/solverparam.json");
    let polynomial_order = solver_param_parser.polynomial_order;
    let cell_gp_num = polynomial_order + 1;
    let mut solver_params = SolverParameters {
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
pub fn initialize_basis(cell_gp_num: usize) -> LagrangeBasis1D {
    let basis = LagrangeBasis1D::new(cell_gp_num);
    basis
}
pub fn initialize_mesh1d(node_num: usize, left_coord: f64, right_coord: f64) -> Mesh1d {
    let mut mesh = Mesh1d::new(node_num, left_coord, right_coord);
    mesh
}
pub fn initialize_solver<'a>(
    mesh: &'a Mesh1d,
    basis: LagrangeBasis1D,
    flow_param: FlowParameters,
    solver_param: SolverParameters,
) -> Solver<'a> {
    let disc = Disc1dBurgers::new(basis, mesh, &flow_param, &solver_param);
    let solver = Solver::new(disc, mesh, flow_param, solver_param);
    solver
}
