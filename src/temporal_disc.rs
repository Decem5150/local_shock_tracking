use ndarray::{Array, Dimension, Ix3};
use crate::spatial_disc::SpatialDisc;

pub struct TemperalDisc<'a> {
    pub temp_sols: Vec<Array<f64, Ix3>>,
    pub curr_time: f64,
    pub curr_step: usize,
    pub time_scheme: TimeScheme,
    pub basis: &'a LagrangeBasis1D,
    pub gauss_points: &'a GaussPoints,
    pub flow_param: &'a FlowParameters,
    pub mesh_param: &'a MeshParameters,
    pub solver_param: &'a SolverParameters,
}
impl<'a> TemperalDisc<'a> {
    pub fn time_march<T: SpatialDisc<D>, D: Dimension>(&mut self, spatial_disc: T, residuals: &mut Array<f64, Ix3>, solutions: &mut Array<f64, Ix3>) {
        while self.curr_step < self.solver_param.final_step && self.curr_time < self.solver_param.final_time {
            let mut time_step = self.compute_time_step(solutions);
            if self.curr_time + time_step > self.solver_param.final_time {
                time_step = self.solver_param.final_time - self.curr_time;
            }
        }
    }
}