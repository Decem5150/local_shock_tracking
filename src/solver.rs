pub struct SolverParameters {
    pub cfl: f64,
    pub final_time: f64,
    pub final_step: usize,
    pub cell_gp_num: usize,
    pub edge_gp_num: usize,
    pub equation_num: usize,
    pub basis_num: usize,
    pub polynomial_order: usize,
}
pub struct MeshParameters {
    pub elem_num: usize,
    pub edge_num: usize,
    pub node_num: usize,
    pub patch_num: usize,
}
pub struct FlowParameters {
    pub hcr: f64,
    pub gas_constant: f64,
    pub prandtl_number: f64,
}
pub struct Solver<T: SpatialDisc, 'a> {
    pub residuals: Array<f64, Ix3>,
    pub solutions: Array<f64, Ix3>,
    pub spatial_disc: T,
    pub temporal_disc: TemperalDisc<'a>,
    pub mesh: &'a Mesh,
    pub basis: &'a DubinerBasis,
    pub gauss_points: &'a GaussPoints,
    pub flow_param: &'a FlowParameters,
    pub mesh_param: &'a MeshParameters,
    pub solver_param: &'a SolverParameters,
}