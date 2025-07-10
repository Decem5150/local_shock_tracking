use ndarray::Array3;

use crate::{
    disc::{
        advection1d_space_time_tri::Disc1dAdvectionSpaceTimeTri,
        basis::triangle::TriangleBasis,
        burgers1d_space_time::Disc1dBurgers1dSpaceTime,
        mesh::{
            mesh1d::Mesh1d,
            mesh2d::{Mesh2d, QuadrilateralElement, TriangleElement},
        },
    },
    io::params_parser::SolverParamParser,
    solver::SolverParameters,
};

pub fn initialize_params_by_file(file_path: &str) -> SolverParameters {
    let solver_param_parser = SolverParamParser::parse(file_path);
    let polynomial_order = solver_param_parser.polynomial_order;
    let solver_params = SolverParameters {
        cfl: solver_param_parser.cfl,
        final_time: solver_param_parser.final_time,
        final_step: solver_param_parser.final_step,
        polynomial_order,
        equation_num: 1,
    };

    solver_params
}
pub fn initialize_params() -> SolverParameters {
    let polynomial_order = 2;
    let solver_params = SolverParameters {
        cfl: 1.0,
        final_time: 0.0,
        final_step: 10,
        polynomial_order,
        equation_num: 1,
    };
    solver_params
}

pub fn initialize_mesh1d(node_num: usize, left_coord: f64, right_coord: f64) -> Mesh1d {
    Mesh1d::new(node_num, left_coord, right_coord)
}
/*
pub fn initialize_two_element_mesh2d() -> Mesh2d<QuadrilateralElement> {
    Mesh2d::create_two_quad_mesh()
}
*/
/*
pub fn initialize_quad_solver<'a>(
    mesh: &'a mut Mesh2d<QuadrilateralElement>,
    basis: LagrangeBasis1DLobatto,
    enriched_basis: LagrangeBasis1DLobatto,
    solver_param: &'a SolverParameters,
) -> ShockTrackingSolverQuad<'a> {
    let solver = ShockTrackingSolverQuad::new(basis, enriched_basis, mesh, solver_param);
    solver
}
*/

/*
pub fn initialize_tri_solver<'a>(
    mesh: &'a mut Mesh2d<TriangleElement>,
    basis: TriangleBasis,
    enriched_basis: TriangleBasis,
    solver_param: &'a SolverParameters,
) -> ShockTrackingSolverTri<'a, Disc1dAdvectionSpaceTimeTri<'a>> {
    let solver = ShockTrackingSolverTri::<Disc1dAdvectionSpaceTimeTri>::new(
        basis,
        enriched_basis,
        mesh,
        solver_param,
    );
    solver
}
*/
