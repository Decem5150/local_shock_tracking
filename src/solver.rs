use crate::disc::{
    SQP, SpaceTimeSolver1DScalar,
    advection1d_space_time_tri::Disc1dAdvectionSpaceTimeTri,
    basis::{lagrange1d::LobattoBasis, quadrilateral::QuadrilateralBasis, triangle::TriangleBasis},
    burgers1d::Disc1dBurgers,
    burgers1d_space_time::Disc1dBurgers1dSpaceTime,
    mesh::{
        mesh1d::Mesh1d,
        mesh2d::{Mesh2d, QuadrilateralElement, TriangleElement},
    },
};
use ndarray::{Array, Array3, Ix2};

pub struct SolverParameters {
    pub cfl: f64,
    pub final_time: f64,
    pub final_step: usize,
    pub polynomial_order: usize,
    pub equation_num: usize,
}
pub struct ShockTrackingParameters {
    pub cell_gp_num: usize,
    pub edge_gp_num: usize,
    pub basis_num: usize,
}
/*
pub struct ShockTrackingSolverQuad<'a> {
    pub solutions: Array<f64, Ix2>,
    pub disc: Disc1dAdvectionSpaceTimeQuad<'a>,
    // pub mesh: &'a Mesh2d,
    pub solver_params: &'a SolverParameters,
}
impl<'a> ShockTrackingSolverQuad<'a> {
    pub fn new(
        basis: LagrangeBasis1DLobatto,
        enriched_basis: LagrangeBasis1DLobatto,
        mesh: &'a mut Mesh2d<QuadrilateralElement>,
        solver_params: &'a SolverParameters,
    ) -> Self {
        let ngp = (solver_params.polynomial_order + 1) * (solver_params.polynomial_order + 1);
        let solutions = Array::zeros((mesh.elem_num, ngp));
        let disc = Disc1dAdvectionSpaceTimeQuad::new(basis, enriched_basis, mesh, solver_params);
        Self {
            solutions,
            disc,
            solver_params,
        }
    }
    pub fn solve(&mut self) {
        self.disc.initialize_solution(self.solutions.view_mut());
        self.disc.solve(self.solutions.view_mut());
    }
}
*/
pub struct ShockTrackingSolverTri<'a, T: SpaceTimeSolver1DScalar + SQP> {
    pub solutions: Array<f64, Ix2>,
    pub disc: T,
    pub solver_params: &'a SolverParameters,
}
impl<'a> ShockTrackingSolverTri<'a, Disc1dAdvectionSpaceTimeTri<'a>> {
    pub fn new(
        basis: TriangleBasis,
        enriched_basis: TriangleBasis,
        mesh: &'a mut Mesh2d<TriangleElement>,
        solver_params: &'a SolverParameters,
    ) -> Self {
        let n = solver_params.polynomial_order;
        let solutions = Array::zeros((mesh.elem_num, (n + 1) * (n + 2) / 2));
        let disc = Disc1dAdvectionSpaceTimeTri::new(basis, enriched_basis, mesh, solver_params);
        Self {
            solutions,
            disc,
            solver_params,
        }
    }
}
impl<'a> ShockTrackingSolverTri<'a, Disc1dBurgers1dSpaceTime<'a>> {
    pub fn new(
        basis: TriangleBasis,
        enriched_basis: TriangleBasis,
        mesh: &'a mut Mesh2d<TriangleElement>,
        solver_params: &'a SolverParameters,
    ) -> Self {
        let n = solver_params.polynomial_order;
        let solutions = Array::zeros((mesh.elem_num, (n + 1) * (n + 2) / 2));
        let disc = Disc1dBurgers1dSpaceTime::new(basis, enriched_basis, mesh, solver_params);
        Self {
            solutions,
            disc,
            solver_params,
        }
    }
}
impl<'a, T: SpaceTimeSolver1DScalar + SQP> ShockTrackingSolverTri<'a, T> {
    pub fn solve(&mut self) {
        self.disc.initialize_solution(self.solutions.view_mut());
        self.disc.solve(self.solutions.view_mut());
    }
}

pub struct Solver<'a> {
    pub solutions: Array3<f64>,
    pub disc: Disc1dBurgers<'a>,
    pub solver_params: &'a SolverParameters,
    // pub shock_tracking_param: ShockTrackingParameters,
}
impl<'a> Solver<'a> {
    pub fn new(
        space_basis: LobattoBasis,
        time_basis: LobattoBasis,
        space_time_basis: QuadrilateralBasis,
        mesh: &'a Mesh1d,
        solver_params: &'a SolverParameters,
    ) -> Self {
        let n = solver_params.polynomial_order;
        let ngp = n + 1;
        let solutions = Array3::zeros((mesh.elem_num, ngp, solver_params.equation_num));
        let disc = Disc1dBurgers::new(
            space_basis,
            time_basis,
            space_time_basis,
            mesh,
            solver_params,
        );
        Self {
            disc,
            solver_params,
            solutions,
        }
    }
    pub fn solve(&mut self) {
        self.disc
            .initialize_solution(self.solutions.view_mut(), &|x| {
                -(x * std::f64::consts::PI).sin()
            });
        self.disc.solve(self.solutions.view_mut());
    }
}
