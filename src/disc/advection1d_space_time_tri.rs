use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, s};
use ndarray_linalg::Inverse;
use std::autodiff::autodiff_reverse;

use super::Geometric;
use super::{
    basis::{Basis, triangle::TriangleBasis},
    mesh::mesh2d::{Mesh2d, TriangleElement},
};
use crate::disc::{P0Solver, SQP, SpaceTimeSolver1DScalar};
use crate::solver::SolverParameters;

pub struct Disc1dAdvectionSpaceTimeTri<'a> {
    basis: TriangleBasis,
    enriched_basis: TriangleBasis,
    interp_node_to_cubature: Array2<f64>,
    interp_node_to_enriched_cubature: Array2<f64>,
    interp_node_to_enriched_quadrature: Array2<f64>,
    pub mesh: &'a mut Mesh2d<TriangleElement>,
    solver_param: &'a SolverParameters,
    advection_speed: f64,
}
impl<'a> Disc1dAdvectionSpaceTimeTri<'a> {
    pub fn new(
        basis: TriangleBasis,
        enriched_basis: TriangleBasis,
        mesh: &'a mut Mesh2d<TriangleElement>,
        solver_param: &'a SolverParameters,
    ) -> Disc1dAdvectionSpaceTimeTri<'a> {
        basis.validate_modal_derivatives(solver_param.polynomial_order, 1e-6);
        let interp_node_to_cubature = Self::compute_interp_matrix_2d(
            solver_param.polynomial_order,
            basis.inv_vandermonde.view(),
            basis.cub_r.view(),
            basis.cub_s.view(),
        );
        let interp_node_to_enriched_cubature = Self::compute_interp_matrix_2d(
            solver_param.polynomial_order,
            basis.inv_vandermonde.view(),
            enriched_basis.cub_r.view(),
            enriched_basis.cub_s.view(),
        );
        let gauss_lobatto_points = &basis.quad_p;
        let enriched_gauss_lobatto_points = &enriched_basis.quad_p;
        let inv_vandermonde_1d = TriangleBasis::vandermonde1d(
            solver_param.polynomial_order,
            gauss_lobatto_points.view(),
        )
        .inv()
        .unwrap();
        let interp_node_to_enriched_quadrature = Self::compute_interp_matrix_1d(
            solver_param.polynomial_order,
            inv_vandermonde_1d.view(),
            enriched_gauss_lobatto_points.view(),
        );
        println!("=== Testing derivative sum property ===");
        let mut total_sum_dxi = 0.0;
        let mut total_sum_deta = 0.0;

        let ncell_basis = basis.r.len();
        let weights = &basis.cub_w;
        let ngp = basis.cub_r.len();
        for itest_func in 0..ncell_basis {
            let mut total_dxi = 0.0;
            let mut total_deta = 0.0;
            for igp in 0..ngp {
                let dtest_func_dxi = basis.dr_cub[(igp, itest_func)];
                let dtest_func_deta = basis.ds_cub[(igp, itest_func)];
                total_dxi += weights[igp] * dtest_func_dxi;
                total_deta += weights[igp] * dtest_func_deta;
            }
            total_sum_dxi += total_dxi;
            total_sum_deta += total_deta;
        }
        println!(
            "Sum over all basis functions: [{:.6e}, {:.6e}]",
            total_sum_dxi, total_sum_deta
        );

        println!("=== Checking cubature weight sum ===");
        let total_weight: f64 = basis.cub_w.iter().sum();
        println!("Sum of cubature weights: {}", total_weight);
        println!("Expected (triangle area): 2.0");
        println!("Error: {:.2e}", (total_weight - 2.0).abs());

        println!("=== Checking derivative matrix property ===");
        let ngp = basis.cub_r.len();
        let ncell_basis = basis.r.len();
        let weights = &basis.cub_w;

        for itest_func in 0..ncell_basis {
            let mut total_dxi = 0.0;
            let mut total_deta = 0.0;
            for igp in 0..ngp {
                let dtest_func_dxi = basis.dr_cub[(igp, itest_func)];
                let dtest_func_deta = basis.ds_cub[(igp, itest_func)];
                total_dxi += weights[igp] * dtest_func_dxi;
                total_deta += weights[igp] * dtest_func_deta;
            }
            println!(
                "Basis {}: ∫∇v dΩ = [{:.6e}, {:.6e}]",
                itest_func, total_dxi, total_deta
            );
        }
        /*
        println!("=== Testing modal basis orthogonality (Mass Matrix) ===");
        // The modal basis functions are rows of V.
        // We want to check ∫ φ_i φ_j dΩ.
        // φ_i at cubature points are given by interp_node_to_cubature.row(i)
        // Mass matrix M_ij = Σ_k w_k * φ_i(r_k, s_k) * φ_j(r_k, s_k)
        // M = interp_node_to_cubature^T * W * interp_node_to_cubature
        // where W is a diagonal matrix with cubature weights.
        // For an orthogonal basis, M should be diagonal.

        let modal_basis_at_cubature = TriangleBasis::vandermonde2d(
            solver_param.polynomial_order,
            basis.cub_r.view(),
            basis.cub_s.view(),
        );

        let mut modal_mass_matrix = Array2::<f64>::zeros((ncell_basis, ncell_basis));
        for i in 0..ncell_basis {
            for j in 0..ncell_basis {
                let mut integral = 0.0;
                for k in 0..ngp {
                    integral += weights[k]
                        * modal_basis_at_cubature[(k, i)]
                        * modal_basis_at_cubature[(k, j)];
                }
                modal_mass_matrix[(i, j)] = integral;
            }
        }

        println!("Mass Matrix M:\n{:?}", modal_mass_matrix);
        */
        Disc1dAdvectionSpaceTimeTri {
            basis,
            enriched_basis,
            interp_node_to_cubature,
            interp_node_to_enriched_cubature,
            interp_node_to_enriched_quadrature,
            mesh,
            solver_param,
            advection_speed: 0.6,
        }
    }
}
impl<'a> SpaceTimeSolver1DScalar for Disc1dAdvectionSpaceTimeTri<'a> {
    fn basis(&self) -> &TriangleBasis {
        &self.basis
    }
    fn enriched_basis(&self) -> &TriangleBasis {
        &self.enriched_basis
    }
    fn interp_node_to_cubature(&self) -> &Array2<f64> {
        &self.interp_node_to_cubature
    }
    fn interp_node_to_enriched_cubature(&self) -> &Array2<f64> {
        &self.interp_node_to_enriched_cubature
    }
    fn interp_node_to_enriched_quadrature(&self) -> &Array2<f64> {
        &self.interp_node_to_enriched_quadrature
    }
    fn mesh(&self) -> &Mesh2d<TriangleElement> {
        self.mesh
    }
    fn mesh_mut(&mut self) -> &mut Mesh2d<TriangleElement> {
        self.mesh
    }
    #[autodiff_reverse(
        dvolume, Const, Const, Const, Duplicated, Duplicated, Duplicated, Active
    )]
    fn volume_integral(
        &self,
        basis: &TriangleBasis,
        itest_func: usize,
        sol: &[f64],
        x: &[f64],
        y: &[f64],
    ) -> f64 {
        let ngp = basis.cub_r.len();
        let weights = &basis.cub_w;
        let mut res = 0.0;
        for igp in 0..ngp {
            let f = self.physical_flux(sol[igp]);
            let xi = basis.cub_r[igp];
            let eta = basis.cub_s[igp];
            let (jacob_det, jacob_inv_t) = Self::evaluate_jacob(xi, eta, x, y);
            let transformed_f = {
                [
                    jacob_det * (f[0] * jacob_inv_t[0] + f[1] * jacob_inv_t[2]),
                    jacob_det * (f[0] * jacob_inv_t[1] + f[1] * jacob_inv_t[3]),
                ]
            };
            let dtest_func_dxi = basis.dr_cub[(igp, itest_func)];
            let dtest_func_deta = basis.ds_cub[(igp, itest_func)];
            /*
            println!("itest_func: {}, sol values: {:?}", itest_func, sol);
            println!("flux values: {:?}", f);
            println!("transformed_f: {:?}", transformed_f);
            println!("derivatives: dξ={}, dη={}", dtest_func_dxi, dtest_func_deta);
            */
            res -= weights[igp] * transformed_f[0] * dtest_func_dxi
                + weights[igp] * transformed_f[1] * dtest_func_deta;
        }
        res
    }
    #[autodiff_reverse(dbnd_flux, Const, Active, Active, Active, Active, Active, Active)]
    fn compute_boundary_flux(&self, u: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
        let advection_speed = self.advection_speed;
        let normal = Self::compute_normal(x0, y0, x1, y1);
        let beta = [advection_speed, 1.0];
        let beta_dot_normal = beta[0] * normal[0] + beta[1] * normal[1];
        let result = beta_dot_normal * u;
        result
    }
    #[autodiff_reverse(
        dnum_flux, Const, Active, Active, Active, Active, Active, Active, Active
    )]
    fn compute_numerical_flux(&self, ul: f64, ur: f64, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
        let advection_speed = self.advection_speed;
        let normal = Self::compute_normal(x0, y0, x1, y1);
        let beta = [advection_speed, 1.0];
        let beta_dot_normal = beta[0] * normal[0] + beta[1] * normal[1];
        let result = 0.5
            * (beta_dot_normal * (ul + ur)
                + (beta_dot_normal * (100.0 * beta_dot_normal).tanh()) * (ul - ur));
        result
    }
    #[autodiff_reverse(dscaling, Const, Const, Const, Const, Duplicated, Duplicated, Active)]
    fn compute_flux_scaling(
        &self,
        xi: f64,
        eta: f64,
        ref_normal: [f64; 2],
        x: &[f64],
        y: &[f64],
    ) -> f64 {
        let (jacob_det, jacob_inv_t) = Self::evaluate_jacob(xi, eta, x, y);
        let transformed_normal = {
            [
                jacob_inv_t[0] * ref_normal[0] + jacob_inv_t[1] * ref_normal[1],
                jacob_inv_t[2] * ref_normal[0] + jacob_inv_t[3] * ref_normal[1],
            ]
        };
        let normal_magnitude =
            (transformed_normal[0].powi(2) + transformed_normal[1].powi(2)).sqrt();
        jacob_det * normal_magnitude
    }
    fn physical_flux(&self, u: f64) -> [f64; 2] {
        let advection_speed = self.advection_speed;
        [advection_speed * u, u]
    }
    fn initialize_solution(&self, mut solutions: ArrayViewMut2<f64>) {
        /*
        solutions.slice_mut(s![0, ..]).fill(4.0);
        solutions.slice_mut(s![1, ..]).fill(4.0);
        solutions.slice_mut(s![2, ..]).fill(0.0);
        solutions.slice_mut(s![3, ..]).fill(0.0);
        */
        solutions.slice_mut(s![0, ..]).fill(3.0);
        solutions.slice_mut(s![1, ..]).fill(3.0);
        solutions.slice_mut(s![2, ..]).fill(1.5);
        solutions.slice_mut(s![3, ..]).fill(1.5);

        solutions.slice_mut(s![4, ..]).fill(3.0);
        solutions.slice_mut(s![5, ..]).fill(3.0);
        solutions.slice_mut(s![6, ..]).fill(1.5);
        solutions.slice_mut(s![7, ..]).fill(1.5);
    }
}
impl P0Solver for Disc1dAdvectionSpaceTimeTri<'_> {
    fn compute_time_steps(&self, _solutions: ArrayView2<f64>) -> Array1<f64> {
        let nelem = self.mesh.elem_num;
        let mut dts = Array1::zeros(nelem);
        let cfl = 0.5;
        let beta_mag = (self.advection_speed.powi(2) + 1.0).sqrt();
        for (ielem, elem) in self.mesh.elements.iter().enumerate() {
            let mut max_len_sq = std::f64::MAX;
            for &iedge in &elem.iedges {
                let edge = &self.mesh().edges[iedge];
                let n0 = &self.mesh().nodes[edge.inodes[0]];
                let n1 = &self.mesh().nodes[edge.inodes[1]];
                let len_sq = (n1.x - n0.x).powi(2) + (n1.y - n0.y).powi(2);
                if len_sq < max_len_sq {
                    max_len_sq = len_sq;
                }
            }
            let max_len = max_len_sq.sqrt();

            let x: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[elem.inodes[i]].x);
            let y: [f64; 3] = std::array::from_fn(|i| self.mesh.nodes[elem.inodes[i]].y);
            let area = Self::compute_element_area(&x, &y);
            let char_len = 2.0 * area / max_len;

            dts[ielem] = cfl * char_len / beta_mag;
        }
        dts
    }
}
impl SQP for Disc1dAdvectionSpaceTimeTri<'_> {}
impl Geometric for Disc1dAdvectionSpaceTimeTri<'_> {}
