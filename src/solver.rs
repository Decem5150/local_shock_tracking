pub struct SolverParameters {
    pub cfl: f64,
    pub final_time: f64,
    pub final_step: usize,
    pub number_of_cell_gp: usize,
    pub number_of_edge_gp: usize,
    pub number_of_equations: usize,
    pub number_of_basis_functions: usize,
    pub order_of_polynomials: usize,
}
pub struct MeshParameters {
    pub number_of_elements: usize,
    pub number_of_edges: usize,
    pub number_of_vertices: usize,
    pub number_of_patches: usize,
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