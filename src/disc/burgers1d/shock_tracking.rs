mod flux;
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayViewMut2};

use crate::{disc::mesh::mesh2d::SubMesh2d, solver::{FlowParameters, MeshParameters, ShockTrackingParameters, SolverParameters}};

pub struct ShockTracking<'a> {
    residuals: Array3<f64>,
    submesh: SubMesh2d,
    flow_param: &'a FlowParameters,
    mesh_param: &'a MeshParameters,
    shock_tracking_param: &'a ShockTrackingParameters,
    solver_param: &'a SolverParameters,
}
impl<'a> ShockTracking<'a> {
    pub fn new(flow_param: &'a FlowParameters, mesh_param: &'a MeshParameters, shock_tracking_param: &'a ShockTrackingParameters, solver_param: &'a SolverParameters) -> Self {
        let residuals: Array3<f64> = Array3::zeros((mesh_param.n_elements, mesh_param.n_nodes, 2));
        let submesh = SubMesh2d::new(mesh_param);
        Self { residuals, submesh, flow_param, mesh_param, shock_tracking_param, solver_param }
    }
    pub fn solve(&mut self) {
    }
    pub fn volume_integral(
        &self, 
        sol: ArrayView2<f64>, // (ndof, neq)
        res: ArrayViewMut2<f64> // (ndof, neq)
    ) {
        let cell_ngp = self.shock_tracking_param.cell_gp_num;
        let nbasis = cell_ngp;
    }
    pub fn edge_integral(
        &self,
        left_bnd_sol: ArrayView2<f64>, // (nedgegp, neq)
        right_bnd_sol: ArrayView2<f64>, // (nedgegp, neq)
        left_res: ArrayViewMut2<f64>, // (ndof, neq)
        right_res: ArrayViewMut2<f64>, // (ndof, neq)
    ) {
    }
}

