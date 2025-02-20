use crate::disc::burgers1d::Disc1dBurgers;
use crate::{disc::Disc, disc::mesh::mesh1d::Mesh1d};
use crate::disc::SpatialDisc;
use crate::temporal_disc::TemporalDisc;
use ndarray::{Array, Ix3};

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
pub struct MeshParameters {
    pub elem_num: usize,
    pub node_num: usize,
    pub patch_num: usize,
}
pub struct FlowParameters {
    pub hcr: f64,
}
pub struct Solver<'a> {
    pub solutions: Array<f64, Ix3>,
    pub disc: Disc1dBurgers<'a>,
    pub mesh: Mesh1d,
    pub flow_param: FlowParameters,
    pub mesh_param: MeshParameters,
    pub solver_param: SolverParameters,
    // pub shock_tracking_param: ShockTrackingParameters,
}
