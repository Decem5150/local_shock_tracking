use ndarray::{Array, Ix3};

pub struct TemperalDisc<'a> {
    pub temporary_solutions: Vec<Array<f64, Ix3>>,
    pub current_time: f64,
    pub current_step: usize,
    pub time_scheme: TimeScheme,
    pub mesh: &'a ,
    pub basis: &'a Lagra
    pub gauss_points: &'a GaussPoints,
    pub flow_param: &'a FlowParameters,
    pub mesh_param: &'a MeshParameters,
    pub solver_param: &'a SolverParameters,
}