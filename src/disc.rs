pub mod ader;
pub mod basis;
pub mod boundary;
// pub mod flux;
pub mod finite_difference;
pub mod gauss_points;
pub mod geometric;
pub mod mesh;
// pub mod riemann_solver;
// pub mod advection1d_space_time_quad;
// pub mod advection1d_space_time_tri;
// pub mod burgers1d;
pub mod burgers1d_space_time;
pub mod space_time_1d_scalar;
pub mod space_time_1d_system;
// pub mod euler1d;
use faer::{Col, linalg::solvers::DenseSolveCore, prelude::Solve};
use faer_ext::{IntoFaer, IntoNdarray};
use nalgebra::{DMatrix, DVector, coordinates::X, dvector};
use ndarray::{
    Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3,
    ArrayViewMut4, Axis, Zip, array, concatenate, s,
};
use ndarray_linalg::Inverse;
use ndarray_stats::QuantileExt;

use crate::{
    disc::{
        basis::{Basis1D, lagrange1d::LobattoBasis, triangle::TriangleBasis},
        finite_difference::{
            FiniteDifference, compute_flux_scaling_derivatives, compute_numerical_flux_derivatives,
            compute_volume_derivatives,
        },
        geometric::Geometric2D,
        mesh::mesh2d::{Mesh2d, Status, TriangleElement},
        space_time_1d_scalar::SpaceTime1DScalar,
    },
    io::write_to_vtu::{write_average, write_nodal_solutions},
};
pub trait P0Solver: Geometric2D + SpaceTime1DScalar {
    fn compute_initial_guess(&self, mesh: &Mesh2d<TriangleElement>) -> Array2<f64> {
        let mut solutions = Array2::zeros((mesh.elem_num, 1));
        self.initialize_solution(solutions.view_mut());
        let nelem = mesh.elem_num;
        let mut residuals = Array2::<f64>::zeros((nelem, 1));
        let max_iter = 5000;
        let tol = 1e-10;

        for i in 0..max_iter {
            self.compute_p0_residuals(mesh, solutions.view(), residuals.view_mut());

            let res_norm = residuals.iter().map(|x| x.powi(2)).sum::<f64>().sqrt() / (nelem as f64);

            if i % 100 == 0 {
                dbg!(&residuals);
                println!("solutions: {:?}", solutions);
                println!("PTC Iter: {}, Res norm: {}", i, res_norm);
            }

            if res_norm < tol {
                println!("PTC converged after {} iterations.", i);
                println!("solutions: {:?}", solutions);
                return solutions;
            }

            let dts = self.compute_time_steps(mesh, solutions.view());
            for ielem in 0..nelem {
                if let Status::Active(elem) = &mesh.elements[ielem] {
                    let phys_x: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[elem.inodes[i]].as_ref().x);
                    let phys_y: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[elem.inodes[i]].as_ref().y);
                    let area = Self::compute_element_area(&phys_x, &phys_y);
                    solutions[[ielem, 0]] -= dts[ielem] / area * residuals[[ielem, 0]];
                }
            }
            println!("iter: {}", i);
            // println!("p0_solutions: {:?}", solutions);
        }
        println!("PTC did not converge within {} iterations", max_iter);
        solutions
    }

    fn compute_p0_residuals(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        solutions: ArrayView2<f64>,
        mut residuals: ArrayViewMut2<f64>,
    ) {
        residuals.fill(0.0);

        // Internal edges
        for &iedge in &mesh.interior_edges {
            if let Status::Active(edge) = &mesh.edges[iedge] {
                let ileft = edge.parents[0];
                let iright = edge.parents[1];

                let lelem = mesh.elements[ileft].as_ref();

                let u_left = solutions[(ileft, 0)];
                let u_right = solutions[(iright, 0)];

                let local_ids = &edge.local_ids;
                let n0 = mesh.phys_nodes[lelem.inodes[local_ids[0]]].as_ref();
                let n1 = mesh.phys_nodes[lelem.inodes[(local_ids[0] + 1) % 3]].as_ref();

                let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();

                let flux =
                    self.compute_smoothed_numerical_flux(u_left, u_right, n0.x, n1.x, n0.y, n1.y);

                residuals[(ileft, 0)] += flux * edge_length;
                residuals[(iright, 0)] -= flux * edge_length;
            }
        }

        // Constant boundaries
        for bnd in &mesh.boundaries.constant {
            let ub = bnd.value;
            for &iedge in &bnd.iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let u = solutions[(ielem, 0)];
                    let local_id = edge.local_ids[0];
                    let n0 = mesh.phys_nodes[elem.inodes[local_id]].as_ref();
                    let n1 = mesh.phys_nodes[elem.inodes[(local_id + 1) % 3]].as_ref();
                    let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();

                    // For boundary edges, the node ordering is assumed to be counter-clockwise
                    // for the parent element, so the normal computed from (n0, n1) is outward-pointing.
                    let flux = self.compute_smoothed_boundary_flux(u, ub, n0.x, n1.x, n0.y, n1.y);

                    residuals[(ielem, 0)] += flux * edge_length;
                }
            }
        }
        // Function boundaries
        for bnd in &mesh.boundaries.function {
            let iedges = &bnd.iedges;
            let func = bnd.func;

            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let elem = mesh.elements[ielem].as_ref();
                    let u = solutions[(ielem, 0)];
                    let local_id = edge.local_ids[0];
                    let n0 = mesh.phys_nodes[elem.inodes[local_id]].as_ref();
                    let n1 = mesh.phys_nodes[elem.inodes[(local_id + 1) % 3]].as_ref();
                    let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();
                    let (mid_x, mid_y) = (0.5 * (n0.x + n1.x), 0.5 * (n0.y + n1.y));
                    let bnd_value = func(mid_x, mid_y);
                    let flux =
                        self.compute_smoothed_boundary_flux(u, bnd_value, n0.x, n1.x, n0.y, n1.y);

                    residuals[(ielem, 0)] += flux * edge_length;
                }
            }
        }
        // Polynomial boundaries
        /*
        for bnd in &mesh.boundaries.polynomial {
            let iedges = &bnd.iedges;
            let nodal_coeffs = &bnd.nodal_coeffs;
            let n0_bnd = mesh.nodes[bnd.inodes[0]].as_ref();
            let n1_bnd = mesh.nodes[bnd.inodes[1]].as_ref();
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let lelem = mesh.elements[ielem].as_ref();
                    let u = solutions[(ielem, 0)];
                    let local_id = edge.local_ids[0];
                    let n0 = mesh.nodes[lelem.inodes[local_id]].as_ref();
                    let n1 = mesh.nodes[lelem.inodes[(local_id + 1) % 3]].as_ref();
                    let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();
                    let xi = array![0.5];
                    let bnd_value = Self::compute_boundary_value_by_interpolation(
                        &nodal_coeffs,
                        self.basis().basis1d.n,
                        &xi,
                        &self.basis().basis1d.inv_vandermonde,
                        n0.x,
                        n1.x,
                        n0.y,
                        n1.y,
                        n0_bnd.x,
                        n1_bnd.x,
                        n0_bnd.y,
                        n1_bnd.y,
                    );

                    let flux = self.compute_boundary_flux(u, bnd_value[0], n0.x, n1.x, n0.y, n1.y);
                    residuals[(ielem, 0)] += flux * edge_length;
                }
            }
        }
        */
        // Open boundaries
        for bnd in &mesh.boundaries.open {
            let iedges = &bnd.iedges;
            for &iedge in iedges {
                if let Status::Active(edge) = &mesh.edges[iedge] {
                    let ielem = edge.parents[0];
                    let lelem = mesh.elements[ielem].as_ref();
                    let u = solutions[(ielem, 0)];
                    let local_id = edge.local_ids[0];
                    let n0 = mesh.phys_nodes[lelem.inodes[local_id]].as_ref();
                    let n1 = mesh.phys_nodes[lelem.inodes[(local_id + 1) % 3]].as_ref();
                    let edge_length = ((n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2)).sqrt();
                    let flux = self.compute_open_boundary_flux(u, n0.x, n1.x, n0.y, n1.y);
                    residuals[(ielem, 0)] += flux * edge_length;
                }
            }
        }
    }
    fn compute_time_steps(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        solutions: ArrayView2<f64>,
    ) -> Array1<f64>;
}

pub trait SQP: P0Solver + SpaceTime1DScalar {
    fn compute_node_constraints(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        new_to_old: &Vec<usize>,
    ) -> Array2<f64> {
        let n_nodes = mesh.node_num;
        let total_dofs = 2 * n_nodes;

        let n_free_dofs = new_to_old.len();

        let mut constraint_matrix = Array2::zeros((total_dofs, n_free_dofs));

        for (new_idx, &old_idx) in new_to_old.iter().enumerate() {
            constraint_matrix[[old_idx, new_idx]] = 1.0;
        }

        constraint_matrix
    }
    #[allow(non_snake_case)]
    fn solve_linear_subproblem(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        node_constraints: ArrayView2<f64>,
        res: ArrayView2<f64>,
        hessian_uu: ArrayView2<f64>,
        hessian_ux: ArrayView2<f64>,
        hessian_xx: ArrayView2<f64>,
        dsol: ArrayView2<f64>,
        dcoord: ArrayView2<f64>,
        dobj_dsol: ArrayView1<f64>,
        dobj_dcoord: ArrayView1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let free_coords = &mesh.free_bnd_x.len() + &mesh.free_bnd_y.len();
        let interior_nnodes = mesh.interior_nodes.len();
        let num_u = dsol.shape()[1];
        let num_x: usize = free_coords + 2 * interior_nnodes;
        let num_lambda = num_u;
        let n_total = num_u + num_x + num_lambda;

        let mut A_ndarray = Array2::<f64>::zeros((n_total, n_total));
        let mut b_ndarray = Array1::<f64>::zeros(n_total);

        A_ndarray
            .slice_mut(s![0..num_u, 0..num_u])
            .assign(&hessian_uu);
        A_ndarray
            .slice_mut(s![0..num_u, num_u..num_u + num_x])
            .assign(&hessian_ux);
        A_ndarray
            .slice_mut(s![0..num_u, num_u + num_x..n_total])
            .assign(&dsol.t());

        A_ndarray
            .slice_mut(s![num_u..num_u + num_x, 0..num_u])
            .assign(&hessian_ux.t());
        A_ndarray
            .slice_mut(s![num_u..num_u + num_x, num_u..num_u + num_x])
            .assign(&hessian_xx);
        A_ndarray
            .slice_mut(s![num_u..num_u + num_x, num_u + num_x..n_total])
            .assign(&dcoord.dot(&node_constraints).t());

        A_ndarray
            .slice_mut(s![num_u + num_x..n_total, 0..num_u])
            .assign(&dsol);
        A_ndarray
            .slice_mut(s![num_u + num_x..n_total, num_u..num_u + num_x])
            .assign(&dcoord.dot(&node_constraints));

        b_ndarray
            .slice_mut(s![0..num_u])
            .assign(&(&dobj_dsol * -1.0));
        b_ndarray
            .slice_mut(s![num_u..num_u + num_x])
            .assign(&(&dobj_dcoord * -1.0));
        b_ndarray
            .slice_mut(s![num_u + num_x..n_total])
            .assign(&(res.flatten() * -1.0));

        let A = A_ndarray.view().into_faer();
        let b = Col::<f64>::from_iter(b_ndarray.view().iter().copied());
        let flu = A.partial_piv_lu();
        let u_x_lambda = flu.solve(&b);

        let delta_u = u_x_lambda.subrows(0, num_u);
        let delta_x = u_x_lambda.subrows(num_u, num_x);

        let delta_u_ndarray = Array1::from_iter(delta_u.iter().copied());
        let delta_x_ndarray = Array1::from_iter(delta_x.iter().copied());
        (delta_u_ndarray, delta_x_ndarray)
    }
    fn update_solutions(
        &self,
        solutions: &mut Array2<f64>,
        new_to_old: &Vec<usize>,
        alpha: f64,
        delta_u: &Array1<f64>,
    ) {
        let delta_u_reshaped = delta_u
            .to_shape((new_to_old.len(), self.basis().xi.len()))
            .unwrap();
        for (i, &new_idx) in new_to_old.iter().enumerate() {
            let delta_u_slice = delta_u_reshaped.row(i);
            solutions.row_mut(new_idx).scaled_add(alpha, &delta_u_slice);
        }
    }
    fn solve(&self, mesh: &mut Mesh2d<TriangleElement>, solutions: &mut Array2<f64>) {
        let initial_guess = self.compute_initial_guess(mesh);
        let initial_solutions = self.set_initial_solution(mesh, initial_guess.view());
        solutions.assign(&initial_solutions);
        let nelem = mesh.elem_num;
        let nnode = mesh.node_num;
        let mut new_to_old_elem: Vec<usize> = (0..nelem).collect();
        let ncell_basis = self.basis().xi.len();
        let enriched_ncell_basis = self.enriched_basis().xi.len();
        let epsilon1 = 1e-5;
        let epsilon2 = 1e-10;
        let mut gamma_k = 1e-2; // regularization parameter for hessian_xx
        let gamma_min = 1e-8;
        let k1 = 1e-2;
        let k2 = 1e-1;
        let sigma: f64 = 0.5;
        let max_line_search_iter = 20;
        let max_sqp_iter = 40;
        // let free_coords = &self.mesh.free_coords;
        // println!("free_coords: {:?}", free_coords);

        let mut residuals: Array2<f64> = Array2::zeros((nelem, ncell_basis));
        let mut dsol: Array4<f64> = Array4::zeros((nelem, ncell_basis, nelem, ncell_basis));
        let mut dx: Array3<f64> = Array3::zeros((nelem, ncell_basis, nnode));
        let mut dy: Array3<f64> = Array3::zeros((nelem, ncell_basis, nnode));
        let mut enriched_residuals: Array2<f64> = Array2::zeros((nelem, enriched_ncell_basis));
        let mut enriched_dsol: Array4<f64> =
            Array4::zeros((nelem, enriched_ncell_basis, nelem, ncell_basis));
        let mut enriched_dx: Array3<f64> = Array3::zeros((nelem, enriched_ncell_basis, nnode));
        let mut enriched_dy: Array3<f64> = Array3::zeros((nelem, enriched_ncell_basis, nnode));

        let mut iter: usize = 0;
        while iter < max_sqp_iter {
            println!("iter: {:?}", iter);
            // reset residuals, dsol, dx, enriched_residuals, enriched_dsol, enriched_dx
            residuals.fill(0.0);
            dsol.fill(0.0);
            dx.fill(0.0);
            dy.fill(0.0);
            enriched_residuals.fill(0.0);
            enriched_dsol.fill(0.0);
            enriched_dx.fill(0.0);
            enriched_dy.fill(0.0);

            // mesh.print_free_node_coords();
            if iter % 5 == 0 {
                write_average("solution", solutions, mesh, &self.basis(), iter);
                write_nodal_solutions("solution", solutions, mesh, &self.basis(), iter);
            }

            mesh.collapse_small_elements(0.2, &mut new_to_old_elem);

            let new_to_old_node = mesh.rearrange_node_dofs();
            let node_constraints = self.compute_node_constraints(mesh, &new_to_old_node);

            self.compute_residuals_and_derivatives(
                mesh,
                solutions.view(),
                residuals.view_mut(),
                dsol.view_mut(),
                dx.view_mut(),
                dy.view_mut(),
                false,
                iter,
            );
            println!("residuals computed");
            let res_active = residuals.select(Axis(0), &new_to_old_elem);
            if iter == 24 {
                dbg!(&residuals.slice(s![4, ..]));
                dbg!(&residuals.slice(s![6, ..]));
            }
            let dsol_active = dsol
                .select(Axis(0), &new_to_old_elem)
                .select(Axis(2), &new_to_old_elem)
                .to_shape((
                    new_to_old_elem.len() * ncell_basis,
                    new_to_old_elem.len() * ncell_basis,
                ))
                .unwrap()
                .to_owned();
            let dx_active = dx
                .select(Axis(0), &new_to_old_elem)
                .to_shape((new_to_old_elem.len() * ncell_basis, nnode))
                .unwrap()
                .to_owned();
            let dy_active = dy
                .select(Axis(0), &new_to_old_elem)
                .to_shape((new_to_old_elem.len() * ncell_basis, nnode))
                .unwrap()
                .to_owned();

            self.compute_residuals_and_derivatives(
                mesh,
                solutions.view(),
                enriched_residuals.view_mut(),
                enriched_dsol.view_mut(),
                enriched_dx.view_mut(),
                enriched_dy.view_mut(),
                true,
                iter,
            );
            println!("enriched residuals computed");
            let enr_res_active = enriched_residuals.select(Axis(0), &new_to_old_elem);
            if iter == 24 {
                dbg!(&enriched_residuals.slice(s![4, ..]));
                dbg!(&enriched_residuals.slice(s![6, ..]));
            }
            let enr_dsol_active = enriched_dsol
                .select(Axis(0), &new_to_old_elem)
                .select(Axis(2), &new_to_old_elem)
                .to_shape((
                    new_to_old_elem.len() * enriched_ncell_basis,
                    new_to_old_elem.len() * ncell_basis,
                ))
                .unwrap()
                .to_owned();
            let enr_dx_active = enriched_dx
                .select(Axis(0), &new_to_old_elem)
                .to_shape((new_to_old_elem.len() * enriched_ncell_basis, nnode))
                .unwrap()
                .to_owned();
            let enr_dy_active = enriched_dy
                .select(Axis(0), &new_to_old_elem)
                .to_shape((new_to_old_elem.len() * enriched_ncell_basis, nnode))
                .unwrap()
                .to_owned();

            /*
            {
                let mut perturbed_solutions = solutions.to_owned();
                perturbed_solutions[(0, 1)] += 1e-6;
                let mut perturbed_residuals = Array2::<f64>::zeros((nelem, enriched_ncell_basis));
                self.compute_residuals(
                    mesh,
                    perturbed_solutions.view(),
                    perturbed_residuals.view_mut(),
                    true,
                );
                let dres_dsol = (&perturbed_residuals - &enriched_residuals) / 1e-6;
                println!("dres_dsol by FD: {:?}", dres_dsol);
                println!("dsol: {:?}", enriched_dsol.slice(s![.., 1]));
            }
            {
                let mut perturbed_mesh = mesh.clone();
                perturbed_mesh.nodes[1].x += 1e-6;
                let mut perturbed_residuals = Array2::<f64>::zeros((nelem, enriched_ncell_basis));
                self.compute_residuals(
                    &perturbed_mesh,
                    solutions.view(),
                    perturbed_residuals.view_mut(),
                    true,
                );
                let dres_dx = (&perturbed_residuals - &enriched_residuals) / 1e-6;
                println!("dres_dx by FD: {:?}", dres_dx);
                println!("enriched_dx: {:?}", enriched_dx.slice(s![.., 1]));
            }
            */
            let dcoord = concatenate(Axis(1), &[dx_active.view(), dy_active.view()]).unwrap();
            let dobj_dsol = enr_dsol_active.t().dot(&enr_res_active.flatten());
            let enriched_dcoord =
                concatenate(Axis(1), &[enr_dx_active.view(), enr_dy_active.view()]).unwrap();
            let dobj_dcoord = enriched_dcoord
                .t()
                .dot(&enr_res_active.flatten())
                .dot(&node_constraints);
            let dsol_inv = dsol_active.inv().unwrap();
            let dsol_inv_t = dsol_inv.t();
            let dobj_dsol_t = dobj_dsol.t();
            let lambda_hat = dsol_inv_t.dot(&dobj_dsol_t);
            let mu = lambda_hat.mapv(f64::abs).max().copied().unwrap() * 2.0;
            // termination criteria
            let optimality = &dobj_dcoord
                - &dcoord
                    .dot(&node_constraints)
                    .t()
                    .dot(&dsol_inv_t)
                    .dot(&dobj_dsol.t());
            let optimality_norm = optimality.mapv(|x| x.powi(2)).sum().sqrt();
            let feasibility_norm = res_active.mapv(|x| x.powi(2)).sum().sqrt();
            println!("optimality: {:?}", optimality_norm);
            println!("feasibility: {:?}", feasibility_norm);
            if optimality_norm < epsilon1 && feasibility_norm < epsilon2 {
                println!("Terminating SQP at iter: {:?}", iter);
                break;
            }
            let hessian_uu = enr_dsol_active.t().dot(&enr_dsol_active);
            let hessian_ux = enr_dsol_active
                .t()
                .dot(&enriched_dcoord.dot(&node_constraints));
            let mut hessian_xx = enriched_dcoord
                .dot(&node_constraints)
                .t()
                .dot(&enriched_dcoord.dot(&node_constraints));
            println!("gamma_k: {:?}", gamma_k);
            hessian_xx += &(gamma_k * &Array2::eye(hessian_xx.shape()[0]));

            let (delta_u, delta_x) = self.solve_linear_subproblem(
                mesh,
                node_constraints.view(),
                res_active.view(),
                hessian_uu.view(),
                hessian_ux.view(),
                hessian_xx.view(),
                dsol_active.view(),
                dcoord.view(),
                dobj_dsol.view(),
                dobj_dcoord.view(),
            );
            println!("linear subproblem solved");
            // backtracking line search
            let merit_func = |alpha: f64| -> f64 {
                let mut tmp_mesh = mesh.clone();
                let mut u = solutions.clone();
                self.update_solutions(&mut u, &new_to_old_elem, alpha, &delta_u);
                /*
                let delta_u_ndarray = Array::from_iter(delta_u.iter().copied());
                let u_flat = &solutions.flatten() + alpha * &delta_u_ndarray;
                let u = u_flat.to_shape((nelem, ncell_basis)).unwrap();
                */
                tmp_mesh.update_node_coords(&new_to_old_node, alpha, delta_x.view());
                let mut tmp_res = Array2::zeros((nelem, ncell_basis));
                self.compute_residuals(&tmp_mesh, u.view(), tmp_res.view_mut(), false);
                let mut tmp_enr_res = Array2::zeros((nelem, enriched_ncell_basis));
                self.compute_residuals(&tmp_mesh, u.view(), tmp_enr_res.view_mut(), true);
                let f = 0.5 * &tmp_enr_res.flatten().dot(&tmp_enr_res.flatten());
                let l1_norm = tmp_res.mapv(f64::abs).sum();

                f + mu * l1_norm
            };
            let merit_func_0 = merit_func(0.0);
            println!("merit_func_0: {:?}", merit_func_0);

            let dir_deriv = dobj_dsol.dot(&delta_u) + dobj_dcoord.dot(&delta_x)
                - mu * residuals.mapv(f64::abs).sum();
            let c: f64 = 1e-4;
            let tau: f64 = 0.5;
            let mut n: i32 = 1;
            let mut alpha: f64 = tau.powi(n - 1);
            let mut line_search_iter: usize = 0;
            while line_search_iter < max_line_search_iter {
                let merit_func_alpha = merit_func(alpha);
                if merit_func_alpha <= merit_func_0 + c * alpha * dir_deriv {
                    println!("merit_func_alpha: {:?}", merit_func_alpha);
                    break;
                }
                alpha *= tau;
                n += 1;
                line_search_iter += 1;
            }
            if line_search_iter == max_line_search_iter {
                panic!(
                    "Warning: Line search did not converge within {} iterations.",
                    max_line_search_iter
                );
            }
            println!("line search done");

            // update gamma_k
            let delta_x_norm = delta_x.mapv(|x| x.powi(2)).sum().sqrt();
            let gamma_k_bar = {
                if delta_x_norm < k1 {
                    sigma * gamma_k
                } else if delta_x_norm > k2 {
                    1.0 / sigma * gamma_k
                } else {
                    gamma_k
                }
            };
            gamma_k = gamma_k_bar.max(gamma_min);

            //solutions.scaled_add(alpha, &delta_u.to_shape(solutions.shape()).unwrap());
            self.update_solutions(solutions, &new_to_old_elem, alpha, &delta_u);
            mesh.update_node_coords(&new_to_old_node, alpha, delta_x.view());

            iter += 1;
        }
        write_average("solution", solutions, mesh, &self.basis(), iter);
        write_nodal_solutions("solution", solutions, mesh, &self.basis(), iter);
        write_nodal_solutions("residual", &residuals, mesh, &self.basis(), iter);
        write_nodal_solutions(
            "enriched_residual",
            &enriched_residuals,
            mesh,
            &self.enriched_basis(),
            iter,
        );
        /*
        println!("enriched_dsol[.., 6]: {:?}", enriched_dsol.slice(s![.., 6]));
        {
            println!("=== Computing finite difference for enriched_dsol[.., 6] ===");
            let epsilon = 1e-7;
            let isol_dof_to_check = 6;
            let ncell_basis = self.basis.r.len();
            let ielem_to_check = isol_dof_to_check / ncell_basis;
            let idof_to_check = isol_dof_to_check % ncell_basis;

            let base_residuals = enriched_residuals.clone();

            let mut perturbed_solutions = solutions.to_owned();
            perturbed_solutions[(ielem_to_check, idof_to_check)] += epsilon;

            let mut perturbed_residuals: Array2<f64> = Array2::zeros((nelem, enriched_ncell_basis));
            self.compute_residuals(
                &self.mesh,
                perturbed_solutions.view(),
                perturbed_residuals.view_mut(),
                true,
            );

            let fd_dsol = (perturbed_residuals
                .into_shape(nelem * enriched_ncell_basis)
                .unwrap()
                - base_residuals
                    .into_shape(nelem * enriched_ncell_basis)
                    .unwrap())
                / epsilon;
            println!("FD enriched_dsol[.., 6]: {:?}", fd_dsol);
        }
        println!("enriched_dx[.., 4]: {:?}", enriched_dx.slice(s![.., 4]));
        {
            println!("=== Computing finite difference for enriched_dx[.., 4] ===");
            let epsilon = 1e-7;
            let inode_to_check = 4;
            let base_residuals = enriched_residuals.clone();
            let original_x = self.mesh.nodes[inode_to_check].x;
            self.mesh.nodes[inode_to_check].x += epsilon;
            let mut perturbed_residuals: Array2<f64> = Array2::zeros((nelem, enriched_ncell_basis));
            self.compute_residuals(
                &self.mesh,
                solutions.view(),
                perturbed_residuals.view_mut(),
                true,
            );
            self.mesh.nodes[inode_to_check].x = original_x;
            let fd_dx = (perturbed_residuals
                .into_shape(nelem * enriched_ncell_basis)
                .unwrap()
                - base_residuals
                    .into_shape(nelem * enriched_ncell_basis)
                    .unwrap())
                / epsilon;
            println!("FD enriched_dx[.., 4]: {:?}", fd_dx);
        }
        */
    }
}

pub fn compute_source_value_nalgebra(
    source_solution: &DVector<f64>,
    n: usize,
    xi: f64, // must be mapped to [0, 1] before passing in
    inv_vandermonde: &DMatrix<f64>,
    x0: f64,
    x1: f64,
) -> f64 {
    let coord = xi * (x1 - x0) + x0;
    let coord_mapped = dvector![coord * 2.0 - 1.0];
    let v = LobattoBasis::vandermonde1d_nalgebra(n, &coord_mapped);
    let interp_matrix = v * inv_vandermonde;
    let values = interp_matrix * source_solution;
    values[0]
}
pub fn print_solutions(solutions: &Array2<f64>, new_to_old_elem: &Array1<usize>) {
    println!("Solutions:");
    for &i in new_to_old_elem {
        let row_str: Vec<String> = solutions
            .row(i)
            .iter()
            .map(|&val| format!("{:.4}", val))
            .collect();
        println!("[{}]", row_str.join(", "));
    }
}
