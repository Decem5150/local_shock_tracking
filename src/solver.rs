use crate::disc::{basis::lagrange1d::LagrangeBasis1DLobatto, burgers1d::Disc1dBurgers, mesh::mesh1d::Mesh1d};
use ndarray::{Array, Array3, Ix3};

pub struct SolverParameters {
    pub cfl: f64,
    pub final_time: f64,
    pub final_step: usize,
    pub polynomial_order: usize,
    pub cell_gp_num: usize,
    pub equation_num: usize,
}
pub struct ShockTrackingParameters {
    pub cell_gp_num: usize,
    pub edge_gp_num: usize,
    pub basis_num: usize,
}
pub struct FlowParameters {
    pub hcr: f64,
}
pub struct Solver<'a> {
    pub solutions: Array<f64, Ix3>,
    pub disc: Disc1dBurgers<'a>,
    pub mesh: &'a Mesh1d,
    pub flow_params: &'a FlowParameters,
    pub solver_params: &'a SolverParameters,
    // pub shock_tracking_param: ShockTrackingParameters,
}
impl<'a> Solver<'a> {
    pub fn new(basis: LagrangeBasis1DLobatto, mesh: &'a Mesh1d, flow_params: &'a FlowParameters, solver_params: &'a SolverParameters) -> Self {
        let solutions = Array::zeros((mesh.elem_num, solver_params.cell_gp_num, solver_params.equation_num));
        let disc = Disc1dBurgers::new(basis, mesh, flow_params, solver_params);
        Self { disc, mesh, flow_params, solver_params, solutions }
    }
    pub fn solve(&mut self) {
        self.disc.solve(self.solutions.view_mut());
    }
}
