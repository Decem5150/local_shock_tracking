#![feature(autodiff)]
mod disc;
mod initialization;
mod io;
mod solver;
use disc::basis::triangle::TriangleBasis;
use disc::mesh::mesh2d::Mesh2d;

use initialization::initialize_two_element_mesh2d;
use nshare::IntoNalgebra;

use crate::disc::basis::lagrange1d::LobattoBasis;
use crate::disc::basis::quadrilateral::QuadrilateralBasis;
use crate::disc::burgers1d_space_time::Disc1dBurgers1dSpaceTime;
use crate::initialization::initialize_mesh1d;
use crate::solver::{ShockTrackingSolverTri, Solver};
use disc::SpaceTimeSolver1DScalar;
fn main() {
    let solver_params = initialization::initialize_params_by_file("inputs/solverparam.json");
    // let basis = TriangleBasis::new(solver_params.polynomial_order);
    // let enriched_basis = TriangleBasis::new(solver_params.polynomial_order + 1);
    let space_basis = LobattoBasis::new(solver_params.polynomial_order);
    let time_basis = LobattoBasis::new(solver_params.polynomial_order);
    let space_time_basis = QuadrilateralBasis::new(solver_params.polynomial_order);
    // let mut mesh = Mesh2d::create_eight_tri_mesh();
    let mesh = initialize_mesh1d(40, -1.0, 1.0);
    // let mut solver =
    //    initialization::initialize_tri_solver(&mut mesh, basis, enriched_basis, &solver_params);
    let mut solver = Solver::new(
        space_basis,
        time_basis,
        space_time_basis,
        &mesh,
        &solver_params,
    );
    /*
    // test compute initial condition value
    {
        let solver_params = initialization::initialize_params();
        let mut mesh = Mesh2d::create_eight_tri_mesh();
        let basis = TriangleBasis::new(solver_params.polynomial_order);
        let enriched_basis = TriangleBasis::new(solver_params.polynomial_order + 1);
        let mut shock_tracking_solver = ShockTrackingSolverTri::<Disc1dBurgers1dSpaceTime>::new(
            basis,
            enriched_basis,
            &mut mesh,
            &solver_params,
        );
        let basis = &shock_tracking_solver.disc.basis();
        let parent_element_sol_points = &basis.basis1d.xi;
        println!("parent_element_sol_points: {:?}", parent_element_sol_points);
        let initial_condition_coarse = parent_element_sol_points.mapv(|x| x * x);
        println!("initial_condition_coarse: {:?}", initial_condition_coarse);
        let mut initial_condition_fine = vec![0.0; basis.basis1d.xi.len()];
        /*
        compute_initial_condition_value(
            initial_condition_coarse.view(),
            basis.basis1d.n,
            basis.basis1d.xi.view(),
            basis.basis1d.inv_vandermonde.view(),
            0.0,
            0.5,
            &mut initial_condition_fine,
        );
        */
        // let initial_condition_coarse_nalgebra = initial_condition_coarse.into_nalgebra();
        let initial_condition_coarse_nalgebra = initial_condition_coarse.into_nalgebra();
        let inv_vandermonde_nalgebra = basis.basis1d.inv_vandermonde.to_owned().into_nalgebra();
        let xi = basis.basis1d.xi[0];
        let (init, dinit_dx0, dinit_dx1) = dinit_dx_nalgebra(
            &initial_condition_coarse_nalgebra,
            basis.basis1d.n,
            xi,
            &inv_vandermonde_nalgebra,
            0.0,
            0.5,
            1.0,
        );
        println!("init: {:?}", init);
        println!("dinit_dx0: {:?}", dinit_dx0);
        println!("dinit_dx1: {:?}", dinit_dx1);
    }
    */

    // solver.solve();
}
