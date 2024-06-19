use basis::lagrange1d::LagrangeBasis1DLobatto;
use flux::flux1d;
use mesh::mesh1d::Mesh1d;
use ndarray::{s, Array, Ix1, Ix2, Ix3};
use riemann_solver::hllc1d;
use crate::solver::{FlowParameters, MeshParameters, SolverParameters};
mod riemann_solver;
mod boundary_conditions;
mod gauss_points;
mod basis;
mod mesh;
mod flux;

pub trait SpatialDisc<D>
where
    D: ndarray::Dimension,
{
    fn compute_residuals(&self, solutions: &Array<f64, D>, residuals: &mut Array<f64, D>);
}
pub struct SpatialDisc1dEuler<'a> {
    pub basis: LagrangeBasis1DLobatto,
    pub mesh: Mesh1d,
    pub flow_param: &'a FlowParameters,
    pub mesh_param: &'a MeshParameters,
    pub solver_param: &'a SolverParameters,
}
impl<'a> SpatialDisc<Ix2> for SpatialDisc1dEuler<'a> {
    fn compute_residuals(&self, solutions: &Array<f64, Ix2>, residuals: &mut Array<f64, Ix2>) {
        self.integrate_over_cells(residuals, solutions);
        self.integrate_over_edges(residuals, solutions);
        self.apply_bc(residuals, solutions);
    }
}
impl<'a> SpatialDisc1dEuler<'a> {
    fn integrate_over_cells(&self, residuals: &mut Array<f64, Ix2>, solutions: &Array<f64, Ix2>) {
        let nelem = self.mesh_param.number_of_elements;
        let cell_ngp = self.solver_param.number_of_cell_gp;
        let nbasis = cell_ngp;
        let neq = self.solver_param.number_of_equations;
        let weights = &self.basis.cell_gauss_weights;
        for ielem in self.mesh.internal_element_indices.iter() {
            let element = &self.mesh.elements[*ielem];
            for igp in 0..cell_ngp {
                let f = flux1d(solutions.slice(s![ielem, .., igp]), self.flow_param.hcr);
                for ivar in 0..neq {
                    for ibasis in 0..nbasis {
                        let dphi_dx = element.dphis_cell_gps[(igp, ibasis)].get(&(igp, 1)).unwrap();
                        residuals += weights[igp] * f[ivar] * dphi_dx * element.jacob_det;
                    }
                }
            }
        }
    }
    fn integrate_over_edges(&self, residuals: &mut Array<f64, Ix2>, solutions: &Array<f64, Ix2>) {
        let nelem = self.mesh_param.number_of_elements;
        let cell_ngp = self.solver_param.number_of_cell_gp;
        let nbasis = cell_ngp;
        let neq = self.solver_param.number_of_equations;
        for ivertex in self.mesh.internal_vertex_indices.iter() {
            let vertex = &self.mesh.vertices[*ivertex];
            let ilelem = vertex.iedges[0] as usize;
            let irelem = vertex.iedges[1] as usize;
            let left_dofs = solutions.slice(s![ilelem, .., ..]);
            let right_dofs = solutions.slice(s![irelem, .., ..]);
            let mut left_values: Array<f64, Ix1> = Array::zeros(neq);
            let mut right_values: Array<f64, Ix1> = Array::zeros(neq);
            for ivar in 0..neq {
                for ibasis in 0..nbasis {
                    left_values[ivar] += left_dofs[ivar] * self.basis.phis_cell_gps[[cell_ngp, ibasis]];
                    right_values[ivar] += right_dofs[ivar] * self.basis.phis_cell_gps[[0, ibasis]];
                }
            }
            let num_flux = match hllc1d(&left_values, &right_values,&self.flow_param.hcr) {
                    Ok(flux) => flux,
                    Err(e) => {
                        println!("{}", e);
                        println!("ivertex: {:?}", ivertex);
                        panic!("Error in HLLC flux computation!");
                }
            };
            for ivar in 0..residuals.shape()[1] {
                for ibasis in 0..residuals.shape()[2] {
                    residuals[[ilelem, ivar, ibasis]] -= num_flux[ivar] * self.basis.phis_cell_gps[[cell_ngp, ibasis]];
                    residuals[[irelem, ivar, ibasis]] += num_flux[ivar] * self.basis.phis_cell_gps[[0, ibasis]];
                }
            }
        }
    }
    fn apply_bc(&self, residuals: &mut Array<f64, Ix2>, solutions: &Array<f64, Ix2>) {
        let cell_ngp = self.solver_param.number_of_cell_gp;
        let nelem = self.mesh.elements.len();
        let neq = self.solver_param.number_of_equations;
        let nbasis = cell_ngp;
        let cell_weights = &self.basis.cell_gauss_weights;
        for ielem in self.mesh.in
    }
}
