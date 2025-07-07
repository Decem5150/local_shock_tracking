use ndarray::{Array1, Array2, ArrayView1, ArrayView2, linalg::kron};
use ndarray_linalg::Inverse;

use crate::disc::basis::{
    Basis, lagrange1d::LobattoBasis, quadrilateral::QuadrilateralBasis, triangle::TriangleBasis,
};

pub trait ADER1DScalar {
    fn physical_flux(&self, u: f64) -> f64;
}
pub trait ADER1DMatrices {
    fn compute_space_mass_mat(basis: &LobattoBasis) -> Array2<f64> {
        /*
        let n_pts_2d = basis.xi.len();
        let n_pts_1d = (n_pts_2d as f64).sqrt() as usize;
        let mut m_1d = Array2::zeros((n_pts_1d, n_pts_1d));
        for i in 0..n_pts_1d {
            m_1d[(i, i)] = basis.quad_w[i];
        }
        m_1d
        */
        0.5 * basis.inv_vandermonde.t().dot(&basis.inv_vandermonde)
    }
    fn compute_space_time_mass_mat(
        space_basis: &LobattoBasis,
        time_basis: &LobattoBasis,
    ) -> Array2<f64> {
        let space_mass_mat = Self::compute_space_mass_mat(space_basis);
        let time_mass_mat = Self::compute_space_mass_mat(time_basis);
        kron(&time_mass_mat, &space_mass_mat)
    }
    fn compute_kxi_mat(space_basis: &LobattoBasis, time_basis: &LobattoBasis) -> Array2<f64> {
        let space_mass_mat = Self::compute_space_mass_mat(space_basis);
        let time_mass_mat = Self::compute_space_mass_mat(time_basis);
        let space_kxi_mat = space_mass_mat.dot(&space_basis.dxi);
        println!("space_kxi_mat: {:?}", space_kxi_mat);
        kron(&time_mass_mat, &space_kxi_mat)
    }
    fn compute_ik1_mat(space_basis: &LobattoBasis, time_basis: &LobattoBasis) -> Array2<f64> {
        let n_pts_1d = space_basis.xi.len();

        let space_mass_mat = Self::compute_space_mass_mat(space_basis);
        let time_basis_product_at_one = {
            let mut tmp = Array2::zeros((n_pts_1d, n_pts_1d));
            tmp[(n_pts_1d - 1, n_pts_1d - 1)] = 1.0;
            tmp
        };
        let f1_mat = kron(&time_basis_product_at_one, &space_mass_mat);
        println!("f1_mat: {:?}", f1_mat);

        let time_mass_mat = Self::compute_space_mass_mat(time_basis);
        let time_stiffness_mat = time_mass_mat.dot(&time_basis.dxi);
        let t_mat = kron(&time_stiffness_mat.t(), &space_mass_mat);
        println!("t_mat: {:?}", t_mat);
        let k1_mat = &f1_mat - &t_mat;
        println!("k1_mat: {:?}", k1_mat);
        k1_mat.inv().unwrap()
    }
    fn compute_f0_mat(space_basis: &LobattoBasis) -> Array2<f64> {
        let n_pts_1d = space_basis.xi.len();

        let space_mass_mat = Self::compute_space_mass_mat(space_basis);
        let time_basis_at_zero = {
            let mut tmp = Array2::zeros((n_pts_1d, 1));
            tmp[(0, 0)] = 1.0;
            tmp
        };

        kron(&time_basis_at_zero, &space_mass_mat)
    }
}
pub trait ADER1DShockTracking {
    fn compute_subpoint_coords(
        target_basis: &LobattoBasis,
        vertex_coords: ArrayView1<f64>, // local coordinates inside the element
    ) -> Array1<f64> {
        let n_subpoints = target_basis.xi.len() * (vertex_coords.len() - 1);
        let mut coords = Array1::zeros(n_subpoints);
        for ivertex in 0..vertex_coords.len() - 1 {
            for inode in 0..target_basis.xi.len() {
                coords[ivertex * target_basis.xi.len() + inode] = vertex_coords[ivertex]
                    + target_basis.xi[inode]
                        * (vertex_coords[ivertex + 1] - vertex_coords[ivertex]);
            }
        }
        coords
    }
    fn compute_interp_matrix(
        n: usize,
        inv_vandermonde: ArrayView2<f64>,
        r: ArrayView1<f64>,
    ) -> Array2<f64> {
        let r_map = r.mapv(|v| 2.0 * v - 1.0);
        let v = LobattoBasis::vandermonde1d(n, r_map.view());
        let interp_matrix = v.dot(&inv_vandermonde);
        interp_matrix
    }
    fn find_elements_in_which_nodes_lie(
        elem_node_coords: &Array1<f64>,
        sol_node_coords: &Array1<f64>,
    ) -> Array1<f64> {
        let mut element_indices = Array1::zeros(sol_node_coords.len());
        let n_elements = elem_node_coords.len() - 1; // n nodes define n-1 elements

        for (i, &sol_coord) in sol_node_coords.indexed_iter() {
            // Find which element this sol_node belongs to
            let mut found_element = 0.0;

            // Linear search through elements to find the containing interval
            for elem_idx in 0..n_elements {
                let left_bound = elem_node_coords[elem_idx];
                let right_bound = elem_node_coords[elem_idx + 1];

                // Check if sol_coord is in this element's interval
                if elem_idx == n_elements - 1 {
                    // Last element: include right boundary [left, right]
                    if sol_coord >= left_bound && sol_coord <= right_bound {
                        found_element = elem_idx as f64;
                        break;
                    }
                } else {
                    // Other elements: exclude right boundary [left, right)
                    if sol_coord >= left_bound && sol_coord < right_bound {
                        found_element = elem_idx as f64;
                        break;
                    }
                }
            }
            element_indices[i] = found_element;
        }

        element_indices
    }
}
