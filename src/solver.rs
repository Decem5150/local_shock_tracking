use crate::{disc::Disc, disc::mesh::mesh1d::Mesh1d};
use crate::disc::SpatialDisc;
use crate::temporal_disc::TemporalDisc;
use ndarray::{Array, Ix3};

pub struct SolverParameters {
    pub cfl: f64,
    pub final_time: f64,
    pub final_step: usize,
    pub cell_gp_num: usize,
    pub edge_gp_num: usize,
    pub equation_num: usize,
    pub basis_num: usize,
    pub polynomial_order: usize,
    pub spatial_order: usize,
    pub temporal_order: usize,
}
pub struct MeshParameters {
    pub elem_num: usize,
    pub node_num: usize,
    pub patch_num: usize,
}
pub struct FlowParameters {
    pub hcr: f64,
    pub gas_constant: f64,
    pub prandtl_number: f64,
}
pub struct Solver<'a, T>
where
    T: Disc,
{
    pub residuals: Array<f64, Ix3>,
    pub solutions: Array<f64, Ix3>,
    pub disc: T,
    pub flow_param: &'a FlowParameters,
    pub mesh_param: &'a MeshParameters,
    pub solver_param: &'a SolverParameters,
}
