pub mod boundary_condition;
// mod cell_shock_detector;
mod flux;
mod precompute_matrix;
mod riemann_solver;
// mod shock_tracking;
use super::mesh::{
    mesh1d::Mesh1d,
    mesh2d::{Mesh2d, TriangleElement},
};
use crate::disc::ader::ADER1DShockTracking;
use crate::disc::{SQP, SpaceTimeSolver1DScalar, geometric::Geometric1D};
use crate::disc::{
    ader::{ADER1DMatrices, ADER1DScalar},
    basis::lagrange1d::LobattoBasis,
};
use crate::disc::{
    basis::{quadrilateral::QuadrilateralBasis, triangle::TriangleBasis},
    burgers1d_space_time::Disc1dBurgers1dSpaceTime,
};
use crate::solver::SolverParameters;
use hashbrown::HashMap;
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Axis,
    s,
};
use ndarray_linalg::Inverse;
use riemann_solver::rusanov::rusanov;

pub struct Disc1dBurgers<'a> {
    pub current_time: f64,
    pub current_step: usize,
    pub space_basis: LobattoBasis,
    pub time_basis: LobattoBasis,
    pub space_time_basis: QuadrilateralBasis,
    mesh: &'a Mesh1d,
    solver_param: &'a SolverParameters,
    space_im_mat: Array2<f64>,
    space_time_im_mat: Array2<f64>,
    kxi_mat: Array2<f64>,
    ik1_mat: Array2<f64>,
    f0_mat: Array2<f64>,
    shock_tracker: Disc1dBurgers1dSpaceTime<'a>,
}
impl<'a> Disc1dBurgers<'a> {
    pub fn new(
        space_basis: LobattoBasis,
        time_basis: LobattoBasis,
        space_time_basis: QuadrilateralBasis,
        mesh: &'a Mesh1d,
        solver_param: &'a SolverParameters,
    ) -> Disc1dBurgers<'a> {
        let space_m_mat = Self::compute_space_mass_mat(&space_basis);
        let space_im_mat = space_m_mat.inv().unwrap();
        let space_time_m_mat = Self::compute_space_time_mass_mat(&space_basis, &time_basis);
        let space_time_im_mat = space_time_m_mat.inv().unwrap();
        let kxi_mat = Self::compute_kxi_mat(&space_basis, &time_basis);
        let ik1_mat = Self::compute_ik1_mat(&space_basis, &time_basis);
        let f0_mat = Self::compute_f0_mat(&space_basis);

        // println!("space_m_mat: {:?}", space_m_mat);
        // println!("space_im_mat: {:?}", space_im_mat);
        // println!("space_time_m_mat: {:?}", space_time_m_mat);
        // println!("space_time_im_mat: {:?}", space_time_im_mat);
        // println!("kxi_mat: {:?}", kxi_mat);
        // println!("ik1_mat: {:?}", ik1_mat);
        // println!("f0_mat: {:?}", f0_mat);

        let shock_tracking_basis = TriangleBasis::new(solver_param.polynomial_order);
        let shock_tracking_enriched_basis = TriangleBasis::new(solver_param.polynomial_order + 1);
        let shock_tracker = Disc1dBurgers1dSpaceTime::new(
            shock_tracking_basis,
            shock_tracking_enriched_basis,
            solver_param,
        );
        let disc = Disc1dBurgers {
            current_time: 0.0,
            current_step: 0,
            space_basis,
            time_basis,
            space_time_basis,
            mesh,
            solver_param,
            space_im_mat,
            space_time_im_mat,
            kxi_mat,
            ik1_mat,
            f0_mat,
            shock_tracker,
        };
        /*
        let (space_m_mat_ref, space_time_m_mat_ref) = disc.compute_m_mat_ref();
        let kxi_mat_ref = disc.compute_kxi_mat_ref();
        let ik1_mat_ref = disc.compute_ik1_mat_ref();
        let f0_mat_ref = disc.compute_f0_mat_ref();
        println!("space_m_mat_ref: {:?}", space_m_mat_ref);
        println!("space_time_m_mat_ref: {:?}", space_time_m_mat_ref);
        println!("kxi_mat_ref: {:?}", kxi_mat_ref);
        println!("ik1_mat_ref: {:?}", ik1_mat_ref);
        println!("f0_mat_ref: {:?}", f0_mat_ref);
        */
        /*
        dbg!(&disc.ss_m_mat);
        dbg!(&disc.ss_im_mat);
        dbg!(&disc.sst_kxi_mat);
        dbg!(&disc.stst_m_mat);
        dbg!(&disc.stst_im_mat);
        dbg!(&disc.stst_kxi_mat);
        dbg!(&disc.stst_ik1_mat);
        dbg!(&disc.sts_f0_mat);
        */
        disc
    }
    pub fn solve(&mut self, mut solutions: ArrayViewMut2<f64>) {
        let nelem = self.mesh.elem_num;
        let space_time_ndof = self.space_time_basis.xi.len();
        let space_ndof = self.space_basis.xi.len();
        let nedge_basis = self.space_time_basis.quad_p.len();
        let edge_weights = &self.space_time_basis.quad_w;
        let mut residuals = Array2::zeros((nelem, space_ndof));
        let mut lqh = Array2::zeros((nelem, space_time_ndof));

        while self.current_step < self.solver_param.final_step
            && self.current_time < self.solver_param.final_time
        {
            println!("current_step: {}", self.current_step);
            println!("current_time: {}", self.current_time);
            println!("solutions: {:?}", solutions);
            let mut sub_solutions = Vec::<Array2<f64>>::new();
            let mut sub_meshes = Vec::<Mesh2d<TriangleElement>>::new();
            let mut is_troubled = Array1::from_elem(nelem, false);
            is_troubled[8] = true;
            let troubled_elems = vec![8];
            residuals.fill(0.0);
            let mut dt = self.compute_time_step(solutions.view());
            // let mut dt = 0.002;
            if self.current_time + dt > self.solver_param.final_time {
                dt = self.solver_param.final_time - self.current_time;
            }

            let mut map_elem_to_troubled_index = HashMap::<usize, usize>::new();
            let mut troubled_count = 0;

            for (ielem, elem) in self.mesh.elements.iter().enumerate() {
                match is_troubled[ielem] {
                    false => {
                        let inodes = &elem.inodes;
                        let x_slice: [f64; 2] =
                            std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
                        let sol_slice = solutions.slice(s![ielem, ..]);
                        lqh.slice_mut(s![ielem, ..])
                            .assign(&self.local_space_time_predictor(sol_slice, x_slice, dt));
                    }
                    true => {
                        let mut submesh = Mesh2d::create_tri_mesh(9);
                        let mut local_solutions = Array2::<f64>::zeros((
                            submesh.elem_num,
                            self.shock_tracker.basis.r.len(),
                        ));
                        self.shock_tracker
                            .initialize_solution(local_solutions.view_mut());
                        self.shock_tracker
                            .solve(&mut submesh, local_solutions.view_mut());
                        sub_solutions.push(local_solutions);
                        sub_meshes.push(submesh);

                        map_elem_to_troubled_index.insert(ielem, troubled_count);
                        troubled_count += 1;
                    }
                }
            }
            for (ielem, elem) in self.mesh.elements.iter().enumerate() {
                match is_troubled[ielem] {
                    false => {
                        let inodes = &elem.inodes;
                        let x_slice: [f64; 2] =
                            std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
                        let lqh_slice = lqh.slice(s![ielem, ..]);
                        for itest_func in 0..space_ndof {
                            let res = self.volume_integral(itest_func, lqh_slice, x_slice, dt);
                            residuals[[ielem, itest_func]] -= res;
                        }
                    }
                    true => {}
                }
            }
            for &inode in self.mesh.internal_nodes.iter() {
                let node = &self.mesh.nodes[inode];
                let ilelem = node.parents[0];
                let irelem = node.parents[1];
                match (is_troubled[ilelem], is_troubled[irelem]) {
                    (false, false) => {
                        let left_inodes = &self.mesh.elements[ilelem].inodes;
                        let right_inodes = &self.mesh.elements[irelem].inodes;
                        let left_x_slice: [f64; 2] =
                            std::array::from_fn(|i| self.mesh.nodes[left_inodes[i]].x);
                        let right_x_slice: [f64; 2] =
                            std::array::from_fn(|i| self.mesh.nodes[right_inodes[i]].x);
                        let left_jacob_det = Self::compute_interval_length(&left_x_slice);
                        let right_jacob_det = Self::compute_interval_length(&right_x_slice);
                        let nodes_along_edges = &self.space_time_basis.nodes_along_edges;
                        let local_ids: [usize; 2] = [1, 3];
                        let left_lqh_slice = lqh.slice(s![ilelem, ..]).select(
                            Axis(0),
                            nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        let right_lqh_slice = lqh.slice(s![irelem, ..]).select(
                            Axis(0),
                            nodes_along_edges
                                .slice(s![local_ids[1], ..])
                                .as_slice()
                                .unwrap(),
                        );
                        for i in 0..nedge_basis {
                            let left_value = left_lqh_slice[i];
                            let right_value = right_lqh_slice[nedge_basis - 1 - i];
                            let num_flux = rusanov(left_value, right_value);
                            let left_scaling = dt / left_jacob_det;
                            let right_scaling = dt / right_jacob_det;
                            let left_transformed_flux = num_flux * left_scaling;
                            let right_transformed_flux = -num_flux * right_scaling;
                            let left_itest_func = self.space_basis.xi.len() - 1;
                            let right_itest_func = 0;
                            residuals[(ilelem, left_itest_func)] -=
                                edge_weights[i] * left_transformed_flux;
                            residuals[(irelem, right_itest_func)] -=
                                edge_weights[i] * right_transformed_flux;
                        }
                    }
                    (false, true) => {
                        let troubled_index = map_elem_to_troubled_index[&irelem];
                        let submesh = &mut sub_meshes[troubled_index];
                        let local_sols = &sub_solutions[troubled_index];

                        let left_inodes = &self.mesh.elements[ilelem].inodes;
                        let left_x_slice: [f64; 2] =
                            std::array::from_fn(|i| self.mesh.nodes[left_inodes[i]].x);
                        let left_jacob_det = Self::compute_interval_length(&left_x_slice);
                        let nodes_along_edges = &self.space_time_basis.nodes_along_edges;
                        let local_ids: [usize; 2] = [1, 3];
                        let left_lqh_slice = lqh.slice(s![ilelem, ..]).select(
                            Axis(0),
                            nodes_along_edges
                                .slice(s![local_ids[0], ..])
                                .as_slice()
                                .unwrap(),
                        );

                        let right_sols_along_boundary =
                            self.shock_tracker.get_solutions_along_boundary(
                                submesh,
                                &submesh.left_bnd.iedges,
                                local_sols,
                            );
                        let etas = &self.time_basis.xi; // quadrature points in time direction
                        let elem_node_coords = Array1::from_iter(
                            submesh
                                .left_bnd
                                .inodes
                                .iter()
                                .map(|&node| submesh.nodes[node].y),
                        );
                        let elements_in_which_nodes_lie =
                            Self::find_elements_in_which_nodes_lie(&elem_node_coords, &etas);
                        let local_coords = Self::compute_local_coords_in_subelements(
                            &elements_in_which_nodes_lie,
                            &elem_node_coords,
                            &etas,
                        );

                        let mut right_sols = Array1::<f64>::zeros(local_coords.len());
                        for (i, &local_coord) in local_coords.indexed_iter() {
                            let local_coord_array = Array1::from_elem(1, local_coord);
                            let interp_matrix = Self::compute_interp_matrix(
                                self.solver_param.polynomial_order,
                                &self.shock_tracker.basis.inv_vandermonde,
                                &local_coord_array,
                            );
                            right_sols[i] =
                                interp_matrix.dot(&right_sols_along_boundary.slice(s![i, ..]))[0];
                        }

                        for i in 0..nedge_basis {
                            let left_value = left_lqh_slice[i];
                            let right_value = right_sols[nedge_basis - 1 - i];
                            let num_flux = rusanov(left_value, right_value);
                            let left_scaling = dt / left_jacob_det;
                            let left_transformed_flux = num_flux * left_scaling;
                            let left_itest_func = self.space_basis.xi.len() - 1;
                            residuals[(ilelem, left_itest_func)] -=
                                edge_weights[i] * left_transformed_flux;
                        }

                        submesh
                            .left_bnd
                            .nodal_coeffs
                            .assign(&left_lqh_slice.view().slice(s![..;-1]));
                    }
                    (true, false) => {
                        let troubled_index = map_elem_to_troubled_index[&ilelem];
                        let submesh = &mut sub_meshes[troubled_index];
                        let local_sols = &sub_solutions[troubled_index];

                        let right_inodes = &self.mesh.elements[irelem].inodes;
                        let right_x_slice: [f64; 2] =
                            std::array::from_fn(|i| self.mesh.nodes[right_inodes[i]].x);
                        let right_jacob_det = Self::compute_interval_length(&right_x_slice);
                        let nodes_along_edges = &self.space_time_basis.nodes_along_edges;
                        let local_ids: [usize; 2] = [1, 3];
                        let right_lqh_slice = lqh.slice(s![irelem, ..]).select(
                            Axis(0),
                            nodes_along_edges
                                .slice(s![local_ids[1], ..])
                                .as_slice()
                                .unwrap(),
                        );

                        let left_sols_along_boundary =
                            self.shock_tracker.get_solutions_along_boundary(
                                submesh,
                                &submesh.right_bnd.iedges,
                                local_sols,
                            );
                        let etas = &self.time_basis.xi; // quadrature points in time direction
                        let elem_node_coords = Array1::from_iter(
                            submesh
                                .right_bnd
                                .inodes
                                .iter()
                                .map(|&node| submesh.nodes[node].y),
                        );
                        let elements_in_which_nodes_lie =
                            Self::find_elements_in_which_nodes_lie(&elem_node_coords, &etas);
                        let local_coords = Self::compute_local_coords_in_subelements(
                            &elements_in_which_nodes_lie,
                            &elem_node_coords,
                            &etas,
                        );

                        let mut left_sols = Array1::<f64>::zeros(local_coords.len());
                        for (i, &local_coord) in local_coords.indexed_iter() {
                            let local_coord_array = Array1::from_elem(1, local_coord);
                            let interp_matrix = Self::compute_interp_matrix(
                                self.solver_param.polynomial_order,
                                &self.shock_tracker.basis.inv_vandermonde,
                                &local_coord_array,
                            );
                            left_sols[i] =
                                interp_matrix.dot(&left_sols_along_boundary.slice(s![i, ..]))[0];
                        }

                        for i in 0..nedge_basis {
                            let left_value = left_sols[i];
                            let right_value = right_lqh_slice[nedge_basis - 1 - i];
                            let num_flux = rusanov(left_value, right_value);
                            let right_scaling = dt / right_jacob_det;
                            let right_transformed_flux = -num_flux * right_scaling;
                            let right_itest_func = 0;
                            residuals[(irelem, right_itest_func)] -=
                                edge_weights[i] * right_transformed_flux;
                        }

                        submesh
                            .right_bnd
                            .nodal_coeffs
                            .assign(&right_lqh_slice.view().slice(s![..;-1]));
                    }
                    (true, true) => {
                        todo!()
                    }
                }
            }
            // apply bc
            // left boundary
            let inode: usize = 0;
            let node = &self.mesh.nodes[inode];
            let ielem = node.parents[0];
            let inodes = &self.mesh.elements[ielem].inodes;
            let x_slice: [f64; 2] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
            let jacob_det = Self::compute_interval_length(&x_slice);
            let nodes_along_edges = &self.space_time_basis.nodes_along_edges;
            let local_id: usize = 3;
            let lqh_slice = lqh.slice(s![ielem, ..]).select(
                Axis(0),
                nodes_along_edges
                    .slice(s![local_id, ..])
                    .as_slice()
                    .unwrap(),
            );
            let bnd_value = 0.0;
            for i in 0..nedge_basis {
                let left_value = bnd_value;
                let right_value = lqh_slice[i];
                let bnd_flux = rusanov(left_value, right_value);
                let scaling = dt / jacob_det;
                let transformed_flux = -bnd_flux * scaling;
                let itest_func = 0;
                residuals[(ielem, itest_func)] -= edge_weights[i] * transformed_flux;
            }
            // right boundary
            let inode: usize = self.mesh.node_num - 1;
            let node = &self.mesh.nodes[inode];
            let ielem = node.parents[0];
            let inodes = &self.mesh.elements[ielem].inodes;
            let x_slice: [f64; 2] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
            let jacob_det = Self::compute_interval_length(&x_slice);
            let nodes_along_edges = &self.space_time_basis.nodes_along_edges;
            let local_id: usize = 1;
            let lqh_slice = lqh.slice(s![ielem, ..]).select(
                Axis(0),
                nodes_along_edges
                    .slice(s![local_id, ..])
                    .as_slice()
                    .unwrap(),
            );
            let bnd_value = 0.0;
            for i in 0..nedge_basis {
                let left_value = lqh_slice[i];
                let right_value = bnd_value;
                let bnd_flux = rusanov(left_value, right_value);
                let scaling = dt / jacob_det;
                let transformed_flux = bnd_flux * scaling;
                let itest_func = self.space_basis.xi.len() - 1;
                residuals[(ielem, itest_func)] -= edge_weights[i] * transformed_flux;
            }
            // multiply inverse mass matrix, accounting for physical domain scaling
            for (ielem, _elem) in self.mesh.elements.iter().enumerate() {
                let computed = self.space_im_mat.dot(&residuals.slice(s![ielem, ..]));
                residuals.slice_mut(s![ielem, ..]).assign(&computed);
            }
            // update solution
            solutions.scaled_add(1.0, &residuals.view());
            /*
            // detect shock

            for &ielem in self.mesh.internal_nodes.iter() {
                let old_sol: Array3<f64> = {
                    let neighbor_num = self.mesh.elements[ielem].ineighbors.len();
                    let mut old_sol: Array3<f64> = Array3::zeros((neighbor_num + 1, cell_ngp, 1));
                    old_sol
                        .slice_mut(s![0, .., ..])
                        .assign(&old_solutions.slice(s![ielem, .., ..]));
                    for (i, &ineigh) in self.mesh.elements[ielem].ineighbors.indexed_iter() {
                        let ineigh = ineigh as usize;
                        old_sol
                            .slice_mut(s![i + 1, .., ..])
                            .assign(&old_solutions.slice(s![ineigh, .., ..]));
                    }
                    old_sol
                };
                /*
                write_to_csv(
                    solutions.view(),
                    self.mesh,
                    &self.basis,
                    &format!("outputs/solutions_{}.csv", self.current_step + 1),
                )
                .unwrap();
                */
                let candidate_sol: ArrayView2<f64> = solutions.slice(s![ielem, .., ..]);
                if self.detect_shock(old_sol.view(), candidate_sol) {
                    println!("Index of shock: {}", ielem);
                    let final_error = write_to_csv(
                        solutions.view(),
                        self.mesh,
                        &self.basis,
                        self.current_time,
                        &format!("outputs/solutions_final.csv"),
                    )
                    .unwrap();
                    println!("Final L² error: {:.4e}", final_error);
                    println!("Final step: {}", self.current_step);
                    println!("Final time: {}", self.current_time);
                    panic!("Shock detected!");
                    // left for shock tracking
                }
            }
            */
            /*
            {
                let ielem = 0;
                let solution_slice = solutions.slice(s![ielem, .., 0]);
                println!("solution_slice: {:?}", solution_slice);
                let vertex_coords = Array1::linspace(0.0, 1.0, 4);
                let subpoint_coords =
                    Self::compute_subpoint_coords(&self.space_basis, vertex_coords.view());
                println!("subpoint_coords: {:?}", subpoint_coords);
                let interp_matrix = Self::compute_interp_matrix(
                    self.solver_param.polynomial_order,
                    self.space_basis.inv_vandermonde.view(),
                    subpoint_coords.view(),
                );
                println!("interp_matrix: {:?}", interp_matrix);
                let interp_solution = interp_matrix.dot(&solution_slice);
                println!("interp_solution: {:?}", interp_solution);
            }
            */

            self.current_time += dt;
            self.current_step += 1;
            println!("step: {}, time: {}", self.current_step, self.current_time);

            if self.current_step % 10 == 0 {
                /*
                    let error = write_to_csv(
                        solutions.view(),
                        self.mesh,
                        &self.basis,
                        self.current_time,
                        &format!("outputs/solutions_{}.csv", self.current_step),
                    )
                    .unwrap();
                */
                // println!("Step {} L² error: {:.4e}", self.current_step, error);
            }
        }

        // Final error calculation
        /*
        let final_error = write_to_csv(
            solutions.view(),
            self.mesh,
            &self.basis,
            self.current_time,
            &format!("outputs/solutions_final.csv"),
        )
        .unwrap();
        println!("Final L² error: {:.4e}", final_error);
        */
        println!("Final step: {}", self.current_step);
        println!("Final time: {}", self.current_time);
        println!("solutions: {:?}", solutions);
    }
    pub fn initialize_solution(
        &self,
        mut solutions: ArrayViewMut3<f64>,
        init_func: &dyn Fn(f64) -> f64,
    ) {
        let ncell_gp = self.space_basis.xi.len();
        for (ielem, elem) in self.mesh.elements.iter().enumerate() {
            let inodes = &elem.inodes;
            let x_slice: [f64; 2] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
            let x_left = x_slice[0];
            let jacob_det = Self::compute_interval_length(&x_slice);
            for igp in 0..ncell_gp {
                let xi = self.space_basis.xi[igp];
                let x = x_left + xi * jacob_det;
                solutions[(ielem, igp, 0)] = init_func(x);
            }
        }
    }
    fn compute_time_step(&self, solutions: ArrayView2<f64>) -> f64 {
        let ndof = self.space_basis.xi.len();
        let mut time_step: f64 = 1.0e10;
        for (ielem, elem) in self.mesh.elements.iter().enumerate() {
            // Compute average velocity in element
            let mut u = 0.0;
            for idof in 0..ndof {
                u += solutions[(ielem, idof)];
            }
            u /= ndof as f64;

            // Wave speed for Burgers equation is |u|
            let speed = u.abs();
            let inodes = &elem.inodes;
            let x_slice: [f64; 2] = std::array::from_fn(|i| self.mesh.nodes[inodes[i]].x);
            let dx = Self::compute_interval_length(&x_slice);
            let dt = self.solver_param.cfl * dx
                / ((self.solver_param.polynomial_order as f64 * 2.0 + 1.0) * speed);

            time_step = time_step.min(dt);
        }

        time_step
    }
    fn local_space_time_predictor(
        &self,
        sol: ArrayView1<f64>, // (ndof)
        x_slice: [f64; 2],
        dt: f64,
    ) -> Array1<f64> {
        let ndof = self.space_time_basis.xi.len();
        let nspace_basis = self.space_basis.xi.len();
        let mut lqh = Array1::zeros(ndof);
        let jacob_det = Self::compute_interval_length(&x_slice);
        // Dimensions: (dof, var) for better memory access in Rust
        let mut lfh = Array1::zeros(ndof); // flux tensor

        // Initial guess for current element
        for idof in 0..ndof {
            let ix = idof % nspace_basis;
            lqh[idof] = sol[ix];
        }

        // Picard iterations for current element
        for _iter in 0..self.solver_param.polynomial_order + 3 {
            // Compute fluxes
            for idof in 0..ndof {
                let f = self.physical_flux(lqh[idof]);
                let transformed_f = dt / jacob_det * f;
                lfh[idof] = transformed_f;
            }
            // update lqh
            // Perform matrix multiplication and store result back in lqh
            let result: Array1<f64> = self
                .ik1_mat
                .dot(&(self.f0_mat.dot(&sol) - self.kxi_mat.dot(&lfh)));
            lqh.assign(&result);
        }
        lqh
    }
    fn volume_integral(
        &self,
        itest_func: usize,
        lqh: ArrayView1<f64>, // (ndof, neq)
        x_slice: [f64; 2],
        dt: f64,
    ) -> f64 {
        let ngp = self.space_time_basis.xi.len();
        let nspace_basis = self.space_basis.xi.len();
        let weights = &self.space_time_basis.cub_w;
        let jacob_det = Self::compute_interval_length(&x_slice);
        let mut res: f64 = 0.0;
        for igp in 0..ngp {
            let f = self.physical_flux(lqh[igp]);
            let transformed_f = dt / jacob_det * f;
            let ix = igp % nspace_basis;
            let dtest_func_dxi = self.space_basis.dxi[(ix, itest_func)];
            res -= weights[igp] * transformed_f * dtest_func_dxi;
        }
        res
    }
}
impl ADER1DScalar for Disc1dBurgers<'_> {
    fn physical_flux(&self, u: f64) -> f64 {
        0.5 * u * u
    }
}
impl ADER1DMatrices for Disc1dBurgers<'_> {}
impl Geometric1D for Disc1dBurgers<'_> {}
impl ADER1DShockTracking for Disc1dBurgers<'_> {}
