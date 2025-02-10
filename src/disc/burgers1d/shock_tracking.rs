mod flux;
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3};

use crate::{disc::mesh::mesh2d::SubMesh2d, solver::{FlowParameters, MeshParameters, SolverParameters}};

pub struct ShockTracking<'a> {
    residuals: Array3<f64>,
    submesh: SubMesh2d,
    flow_param: &'a FlowParameters,
    mesh_param: &'a MeshParameters,
    solver_param: &'a SolverParameters,
}
impl<'a> ShockTracking<'a> {
    pub fn new(flow_param: &'a FlowParameters, mesh_param: &'a MeshParameters, solver_param: &'a SolverParameters) -> Self {
        let residuals = Array3::zeros((mesh_param.n_elements, mesh_param.n_nodes, 2));
        let submesh = SubMesh2d::new(mesh_param);
        Self { residuals, submesh, flow_param, mesh_param, solver_param }
    }
    pub fn compute_residuals(&mut self) {
    }
    pub fn volume_integral(&mut self) {
    }
    pub fn edge_integral(&mut self) {
    }
}

