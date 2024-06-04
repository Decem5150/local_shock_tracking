use basis::lagrange::LagrangeBasis1DLobatto;
use flux::flux1d;
use mesh::mesh1d::Mesh1d;
use ndarray::{s, Array, Ix2};
use crate::solver::{FlowParameters, MeshParameters, SolverParameters};

mod gauss_points;
mod basis;
mod mesh;
mod flux;

pub trait SpatialDisc<T, D>
where
    T: ndarray::Data<Elem = f64>,
    D: ndarray::Dimension,
{
    fn compute_residuals(&self, solutions: &Array<T, D>, residuals: &mut Array<T, D>);
}
pub struct SpatialDisc1DEuler<'a> {
    pub basis: LagrangeBasis1DLobatto,
    pub mesh: Mesh1d,
    pub flow_param: &'a FlowParameters,
    pub mesh_param: &'a MeshParameters,
    pub solver_param: &'a SolverParameters,
}
impl<'a> SpatialDisc<f64, Ix2> for SpatialDisc1DEuler<'a> {
    fn compute_residuals(&self, solutions: &Array<f64, Ix2>, residuals: &mut Array<f64, Ix2>) {
        self.integrate_over_cell_euler(residuals, solutions);
        self.integrate_over_edges_euler(residuals, solutions);
        self.apply_bc_euler(residuals, solutions);
    }
}
impl<'a>  SpatialDisc1DEuler<'a> {
    fn integrate_over_cell_euler(&self, residuals: &mut Array<f64, Ix2>, solutions: &Array<f64, Ix2>) {
        let nelem = self.mesh_param.number_of_elements;
        let cell_ngp = self.solver_param.number_of_cell_gp;
        let neq = self.solver_param.number_of_equations;
        let weights = &self.basis.cell_gauss_weights;
        for ielem in self.mesh.internal_element_indices.iter() {
            for igp in 0..cell_ngp {
                let f = flux1d(solutions.slice(s![ielem, .., igp]), self.flow_param.hcr);
                for ivar in 0..solutions.shape()[1]
            }
        }
    }
}
