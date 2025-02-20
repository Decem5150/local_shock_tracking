use crate::{disc::{basis::lagrange1d::LagrangeBasis1D, mesh::mesh1d::Mesh1d}, solver::{FlowParameters, MeshParameters, SolverParameters}, };
use crate::io::param_parser::SolverParamParser;

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
    let flow_params = FlowParameters {
        hcr: 1.4,
    };
    (flow_params, solver_params)
}
pub fn initialize_basis(cell_gp_num: usize) -> LagrangeBasis1D {
    let basis = LagrangeBasis1D::new(cell_gp_num);
    basis
}
