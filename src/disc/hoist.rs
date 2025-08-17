use faer::{Col, prelude::Solve};
use faer_ext::IntoFaer;
use ndarray::{
    Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3, Axis, concatenate, s,
};
use ndarray_linalg::Inverse;
use ndarray_stats::QuantileExt;

use crate::{
    disc::{
        Mesh2d, SpaceTime1DScalar,
        cg_basis::triangle::TriangleCGBasis,
        dg_basis::triangle::TriangleBasis,
        finite_difference::{FiniteDifference, compute_distortion_derivatives},
        linear_elliptic::LinearElliptic,
        mesh::mesh2d::{Status, TriangleElement},
        space_time_1d_scalar::P0Solver,
    },
    io::write_to_vtu::{write_average, write_nodal_solutions},
};

pub trait HOIST: P0Solver + SpaceTime1DScalar {
    fn compute_mesh_residuals(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        mut residuals: ArrayViewMut1<f64>,
    ) {
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let inodes = &elem.inodes;

                let ref_x: [f64; 3] = std::array::from_fn(|i| mesh.ref_nodes[inodes[i]].as_ref().x);
                let ref_y: [f64; 3] = std::array::from_fn(|i| mesh.ref_nodes[inodes[i]].as_ref().y);
                let ref_distortion =
                    Self::compute_elementwise_distortion(&ref_x, &ref_y, self.basis());

                let phys_x: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                let phys_y: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);
                let phys_distortion =
                    Self::compute_elementwise_distortion(&phys_x, &phys_y, self.basis());

                residuals[ielem] = phys_distortion - ref_distortion;
            }
        }
    }
    fn compute_mesh_residuals_and_derivatives(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        mut residuals: ArrayViewMut1<f64>,
        mut dx: ArrayViewMut2<f64>,
        mut dy: ArrayViewMut2<f64>,
    ) {
        let fd = FiniteDifference::new();

        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let inodes = &elem.inodes;

                let ref_x: [f64; 3] = std::array::from_fn(|i| mesh.ref_nodes[inodes[i]].as_ref().x);
                let ref_y: [f64; 3] = std::array::from_fn(|i| mesh.ref_nodes[inodes[i]].as_ref().y);
                let ref_distortion =
                    Self::compute_elementwise_distortion(&ref_x, &ref_y, self.basis());

                let phys_x: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                let phys_y: [f64; 3] =
                    std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);

                // Compute distortion and its derivatives
                let mut d_phys_x = [0.0; 3];
                let mut d_phys_y = [0.0; 3];

                let phys_distortion = compute_distortion_derivatives(
                    &fd,
                    |x, y, basis| Self::compute_elementwise_distortion(x, y, basis),
                    &phys_x,
                    &phys_y,
                    &mut d_phys_x,
                    &mut d_phys_y,
                    self.basis(),
                );

                residuals[ielem] = phys_distortion - ref_distortion;

                // Add the distortion derivatives to the dx and dy arrays
                // Note: we're adding contributions to the global node indices
                for (local_idx, &global_idx) in inodes.iter().enumerate() {
                    dx[(ielem, global_idx)] = d_phys_x[local_idx];
                    dy[(ielem, global_idx)] = d_phys_y[local_idx];
                }
            }
        }
    }
    fn compute_elementwise_distortion(x: &[f64], y: &[f64], basis: &TriangleBasis) -> f64 {
        let dimention = 2;
        let dn_dxi = [
            -1.0, // dN0/dξ
            1.0,  // dN1/dξ
            0.0,  // dN2/dξ
        ];
        let dn_deta = [
            -1.0, // dN0/dη
            0.0,  // dN1/dη
            1.0,  // dN2/dη
        ];
        let mut distortion = 0.0;
        for i in 0..basis.cub_xi.len() {
            let _xi = basis.cub_xi[i];
            let _eta = basis.cub_eta[i];

            let mut dx_dxi = 0.0;
            let mut dx_deta = 0.0;
            let mut dy_dxi = 0.0;
            let mut dy_deta = 0.0;

            for k in 0..3 {
                dx_dxi += dn_dxi[k] * x[k];
                dx_deta += dn_deta[k] * x[k];
                dy_dxi += dn_dxi[k] * y[k];
                dy_deta += dn_deta[k] * y[k];
            }

            let jacob_det = dx_dxi * dy_deta - dx_deta * dy_dxi;
            let jacobian = [dx_dxi, dx_deta, dy_dxi, dy_deta];

            let jacobian_frobenius_norm = jacobian.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            distortion += basis.cub_w[i]
                * (jacobian_frobenius_norm.powi(2) / jacob_det.powi(dimention / 2)).powi(2);
        }
        distortion
    }
    fn compute_node_constraints(
        &self,
        mesh: &Mesh2d<TriangleElement>,
        new_to_old: &[usize],
    ) -> Array2<f64> {
        let n_nodes = mesh.phys_nodes.len();
        let total_dofs = 2 * n_nodes;

        let n_free_dofs = new_to_old.len();

        let mut constraint_matrix = Array2::zeros((total_dofs, n_free_dofs));

        for (new_idx, &old_idx) in new_to_old.iter().enumerate() {
            constraint_matrix[[old_idx, new_idx]] = 1.0;
        }

        constraint_matrix
    }
    #[allow(non_snake_case, clippy::too_many_arguments)]
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
        let free_coords = mesh.free_bnd_x.len() + mesh.free_bnd_y.len();
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
        new_to_old: &[usize],
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
        let nelem = mesh.elements.len();
        let nnode = mesh.phys_nodes.len();
        let mut new_to_old_elem: Vec<usize> = (0..nelem).collect();
        let ncell_basis = self.basis().xi.len();
        let enriched_ncell_basis = self.enr_basis().xi.len();
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

        let mut new_to_old_node = mesh.rearrange_node_dofs();
        let mut node_constraints = self.compute_node_constraints(mesh, &new_to_old_node);

        let cg_basis = TriangleCGBasis::new(1);
        let linear_elliptic = LinearElliptic::new(cg_basis);
        let mut regularization_matrix = linear_elliptic.compute_stiffness(mesh);

        let mut iter: usize = 0;
        while iter < max_sqp_iter {
            println!("iter: {iter}");
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
            if iter.is_multiple_of(5) {
                write_average("solution", solutions, mesh, self.basis(), iter);
                write_nodal_solutions("solution", solutions, mesh, self.basis(), iter);
            }

            let performed_collapse = mesh.collapse_small_elements(0.2, &mut new_to_old_elem);

            if performed_collapse {
                // Update node constraints and regularization matrix if edge collapse is performed.
                new_to_old_node = mesh.rearrange_node_dofs();
                node_constraints = self.compute_node_constraints(mesh, &new_to_old_node);
                regularization_matrix = linear_elliptic.compute_stiffness(mesh);
            }

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
            println!("optimality: {optimality_norm}");
            println!("feasibility: {feasibility_norm}");
            if optimality_norm < epsilon1 && feasibility_norm < epsilon2 {
                println!("Terminating SQP at iter: {iter}");
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
            println!("gamma_k: {gamma_k}");
            hessian_xx += &(gamma_k * &regularization_matrix);

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
            println!("merit_func_0: {merit_func_0}");

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
                    println!("merit_func_alpha: {merit_func_alpha}");
                    break;
                }
                alpha *= tau;
                n += 1;
                line_search_iter += 1;
            }
            if line_search_iter == max_line_search_iter {
                panic!(
                    "Warning: Line search did not converge within {max_line_search_iter} iterations.",
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
        write_average("solution", solutions, mesh, self.basis(), iter);
        write_nodal_solutions("solution", solutions, mesh, self.basis(), iter);
        write_nodal_solutions("residual", &residuals, mesh, self.basis(), iter);
        write_nodal_solutions(
            "enriched_residual",
            &enriched_residuals,
            mesh,
            self.enr_basis(),
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

/*
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
*/
pub fn print_solutions(solutions: &Array2<f64>, new_to_old_elem: &Array1<usize>) {
    println!("Solutions:");
    for &i in new_to_old_elem {
        let row_str: Vec<String> = solutions
            .row(i)
            .iter()
            .map(|&val| format!("{val:.4}"))
            .collect();
        println!("[{}]", row_str.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disc::mesh::mesh2d::Status;
    use ndarray::{Array1, Array2};

    // Minimal struct that implements only what's needed for HOIST trait
    struct MockHOISTSolver {
        basis: TriangleBasis,
    }

    impl MockHOISTSolver {
        fn new() -> Self {
            Self {
                basis: TriangleBasis::new(2),
            }
        }
        
        fn basis(&self) -> &TriangleBasis {
            &self.basis
        }
    }

    // Simple wrapper that allows us to use HOIST trait methods
    impl MockHOISTSolver {
        fn compute_mesh_residuals_and_derivatives_wrapper(
            &self,
            mesh: &Mesh2d<TriangleElement>,
            mut residuals: ArrayViewMut1<f64>,
            mut dx: ArrayViewMut2<f64>,
            mut dy: ArrayViewMut2<f64>,
        ) {
            // Directly implement the method logic from HOIST trait
            let fd = FiniteDifference::new();

            for (ielem, elem) in mesh.elements.iter().enumerate() {
                if let Status::Active(elem) = elem {
                    let inodes = &elem.inodes;

                    let ref_x: [f64; 3] = std::array::from_fn(|i| mesh.ref_nodes[inodes[i]].as_ref().x);
                    let ref_y: [f64; 3] = std::array::from_fn(|i| mesh.ref_nodes[inodes[i]].as_ref().y);
                    let ref_distortion =
                        Self::compute_elementwise_distortion_static(&ref_x, &ref_y, self.basis());

                    let phys_x: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().x);
                    let phys_y: [f64; 3] =
                        std::array::from_fn(|i| mesh.phys_nodes[inodes[i]].as_ref().y);

                    // Compute distortion and its derivatives
                    let mut d_phys_x = [0.0; 3];
                    let mut d_phys_y = [0.0; 3];

                    let phys_distortion = compute_distortion_derivatives(
                        &fd,
                        |x, y, basis| Self::compute_elementwise_distortion_static(x, y, basis),
                        &phys_x,
                        &phys_y,
                        &mut d_phys_x,
                        &mut d_phys_y,
                        self.basis(),
                    );

                    residuals[ielem] = phys_distortion - ref_distortion;

                    // Add the distortion derivatives to the dx and dy arrays
                    // Note: we're adding contributions to the global node indices
                    for (local_idx, &global_idx) in inodes.iter().enumerate() {
                        dx[(ielem, global_idx)] = d_phys_x[local_idx];
                        dy[(ielem, global_idx)] = d_phys_y[local_idx];
                    }
                }
            }
        }
        
        fn compute_elementwise_distortion_static(x: &[f64], y: &[f64], basis: &TriangleBasis) -> f64 {
            let dimention = 2;
            let dn_dxi = [
                -1.0, // dN0/dξ
                1.0,  // dN1/dξ
                0.0,  // dN2/dξ
            ];
            let dn_deta = [
                -1.0, // dN0/dη
                0.0,  // dN1/dη
                1.0,  // dN2/dη
            ];
            let mut distortion = 0.0;
            for i in 0..basis.cub_xi.len() {
                let _xi = basis.cub_xi[i];
                let _eta = basis.cub_eta[i];

                let mut dx_dxi = 0.0;
                let mut dx_deta = 0.0;
                let mut dy_dxi = 0.0;
                let mut dy_deta = 0.0;

                for k in 0..3 {
                    dx_dxi += dn_dxi[k] * x[k];
                    dx_deta += dn_deta[k] * x[k];
                    dy_dxi += dn_dxi[k] * y[k];
                    dy_deta += dn_deta[k] * y[k];
                }

                let jacob_det = dx_dxi * dy_deta - dx_deta * dy_dxi;
                let jacobian = [dx_dxi, dx_deta, dy_dxi, dy_deta];

                let jacobian_frobenius_norm = jacobian.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                distortion += basis.cub_w[i]
                    * (jacobian_frobenius_norm.powi(2) / jacob_det.powi(dimention / 2)).powi(2);
            }
            distortion
        }
    }

    fn create_simple_triangle_mesh() -> Mesh2d<TriangleElement> {
        // Create a mesh with a single triangle
        let x_num = 2;
        let y_num = 2;
        let x0 = 0.0;
        let x1 = 1.0;
        let y0 = 0.0;
        let y1 = 1.0;
        let mut mesh = Mesh2d::<TriangleElement>::create_tri_mesh(x_num, y_num, x0, x1, y0, y1, 1);
        
        // Keep only the first triangle for simplicity
        mesh.elements.truncate(1);
        
        // The mesh already has nodes created, ensure they're identical
        // The physical nodes should already be identical to reference nodes from creation
        
        mesh
    }

    #[test]
    fn test_zero_residual_when_meshes_identical() {
        let solver = MockHOISTSolver::new();
        let mesh = create_simple_triangle_mesh();
        
        let nelem = mesh.elements.len();
        let nnode = mesh.phys_nodes.len();
        
        let mut residuals = Array1::<f64>::zeros(nelem);
        let mut dx = Array2::<f64>::zeros((nelem, nnode));
        let mut dy = Array2::<f64>::zeros((nelem, nnode));
        
        solver.compute_mesh_residuals_and_derivatives_wrapper(
            &mesh,
            residuals.view_mut(),
            dx.view_mut(),
            dy.view_mut(),
        );
        
        // When reference and physical meshes are identical, residuals should be zero
        assert!(residuals.iter().all(|&r| r.abs() < 1e-10), 
                "Residuals should be zero when meshes are identical, but got: {:?}", residuals);
    }

    #[test]
    fn test_nonzero_residual_when_meshes_different() {
        let solver = MockHOISTSolver::new();
        let mut mesh = create_simple_triangle_mesh();
        
        // Perturb the physical mesh
        if let Status::Active(ref mut node) = mesh.phys_nodes[1] {
            node.x += 0.1;
        }
        
        let nelem = mesh.elements.len();
        let nnode = mesh.phys_nodes.len();
        
        let mut residuals = Array1::<f64>::zeros(nelem);
        let mut dx = Array2::<f64>::zeros((nelem, nnode));
        let mut dy = Array2::<f64>::zeros((nelem, nnode));
        
        solver.compute_mesh_residuals_and_derivatives_wrapper(
            &mesh,
            residuals.view_mut(),
            dx.view_mut(),
            dy.view_mut(),
        );
        
        // When meshes are different, residuals should be non-zero
        assert!(residuals.iter().any(|&r| r.abs() > 1e-10), 
                "Residuals should be non-zero when meshes are different");
    }

    #[test]
    fn test_derivatives_finite_difference_consistency() {
        let solver = MockHOISTSolver::new();
        let mesh = create_simple_triangle_mesh();
        
        let nelem = mesh.elements.len();
        let nnode = mesh.phys_nodes.len();
        
        let mut residuals = Array1::<f64>::zeros(nelem);
        let mut dx = Array2::<f64>::zeros((nelem, nnode));
        let mut dy = Array2::<f64>::zeros((nelem, nnode));
        
        solver.compute_mesh_residuals_and_derivatives_wrapper(
            &mesh,
            residuals.view_mut(),
            dx.view_mut(),
            dy.view_mut(),
        );
        
        let base_residual = residuals[0];
        
        // Test x-derivative for node 0
        let epsilon = 1e-7;
        let mut perturbed_mesh = mesh.clone();
        if let Status::Active(ref mut node) = perturbed_mesh.phys_nodes[0] {
            node.x += epsilon;
        }
        
        let mut perturbed_residuals = Array1::<f64>::zeros(nelem);
        let mut dummy_dx = Array2::<f64>::zeros((nelem, nnode));
        let mut dummy_dy = Array2::<f64>::zeros((nelem, nnode));
        
        solver.compute_mesh_residuals_and_derivatives_wrapper(
            &perturbed_mesh,
            perturbed_residuals.view_mut(),
            dummy_dx.view_mut(),
            dummy_dy.view_mut(),
        );
        
        let fd_derivative = (perturbed_residuals[0] - base_residual) / epsilon;
        let analytical_derivative = dx[(0, 0)];
        
        assert!((fd_derivative - analytical_derivative).abs() < 1e-5,
                "Finite difference derivative ({}) should match analytical derivative ({})",
                fd_derivative, analytical_derivative);
    }
}
