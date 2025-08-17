use ndarray::Array2;

use crate::disc::{
    cg_basis::triangle::TriangleCGBasis,
    mesh::mesh2d::{Mesh2d, Status, TriangleElement},
};

pub struct LinearElliptic {
    basis: TriangleCGBasis,
}
impl LinearElliptic {
    pub fn new(basis: TriangleCGBasis) -> Self {
        Self { basis }
    }
    pub fn compute_stiffness(&self, mesh: &Mesh2d<TriangleElement>) -> Array2<f64> {
        let mut stiffness = Array2::<f64>::zeros((mesh.ref_nodes.len(), mesh.ref_nodes.len()));
        let (_min_id, min_area) = Self::find_smallest_element(mesh);
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let x: [f64; 3] =
                    std::array::from_fn(|i| mesh.ref_nodes[elem.inodes[i]].as_ref().x);
                let y: [f64; 3] =
                    std::array::from_fn(|i| mesh.ref_nodes[elem.inodes[i]].as_ref().y);
                let area = Self::compute_element_area(&x, &y);
                let kappa = min_area / area;

                // k for test functions, l for trial functions
                for k in 0..self.basis.dxi_cub.shape()[1] {
                    for l in 0..self.basis.dxi_cub.shape()[1] {
                        let global_dof_ids = Self::get_global_dof_ids(mesh, ielem, 1);
                        let global_k = global_dof_ids[k];
                        let global_l = global_dof_ids[l];
                        // Integrate over quadrature points
                        for igp in 0..self.basis.cub_l1.len() {
                            let l1 = self.basis.cub_l1[igp];
                            let l2 = self.basis.cub_l2[igp];
                            let l3 = 1.0 - l1 - l2;
                            let weight = self.basis.cub_w[igp];

                            // Get Jacobian and its inverse transpose
                            let (jacob_det, jacob_inv_t) =
                                Self::evaluate_jacob(1, l1, l2, l3, &x, &y);

                            // Transform gradients from reference to physical coordinates
                            // grad_phi_physical = J^{-T} * grad_phi_reference
                            let dphi_k_dxi = self.basis.dxi_cub[[igp, k]];
                            let dphi_k_deta = self.basis.deta_cub[[igp, k]];
                            let dphi_l_dxi = self.basis.dxi_cub[[igp, l]];
                            let dphi_l_deta = self.basis.deta_cub[[igp, l]];

                            // Compute physical gradients for test function k
                            let dphi_k_dx =
                                jacob_inv_t[0] * dphi_k_dxi + jacob_inv_t[1] * dphi_k_deta;
                            let dphi_k_dy =
                                jacob_inv_t[2] * dphi_k_dxi + jacob_inv_t[3] * dphi_k_deta;

                            // Compute physical gradients for trial function l
                            let dphi_l_dx =
                                jacob_inv_t[0] * dphi_l_dxi + jacob_inv_t[1] * dphi_l_deta;
                            let dphi_l_dy =
                                jacob_inv_t[2] * dphi_l_dxi + jacob_inv_t[3] * dphi_l_deta;

                            // Compute integrand: kappa * grad(phi_k) . grad(phi_l) * |J| * w
                            stiffness[(global_k, global_l)] += kappa
                                * (dphi_k_dx * dphi_l_dx + dphi_k_dy * dphi_l_dy)
                                * jacob_det
                                * weight;
                        }
                    }
                }
            }
        }
        stiffness
    }
    fn find_smallest_element(mesh: &Mesh2d<TriangleElement>) -> (usize, f64) {
        let mut min_area = f64::INFINITY;
        let mut min_element_id = 0;
        for (ielem, elem) in mesh.elements.iter().enumerate() {
            if let Status::Active(elem) = elem {
                let x: [f64; 3] =
                    std::array::from_fn(|i| mesh.ref_nodes[elem.inodes[i]].as_ref().x);
                let y: [f64; 3] =
                    std::array::from_fn(|i| mesh.ref_nodes[elem.inodes[i]].as_ref().y);
                let area = Self::compute_element_area(&x, &y);
                if area < min_area {
                    min_area = area;
                    min_element_id = ielem;
                }
            }
        }
        (min_element_id, min_area)
    }
    fn compute_element_area(x: &[f64], y: &[f64]) -> f64 {
        // For triangular elements, assumes x and y are slices of length 3
        0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])).abs()
    }
    fn evaluate_jacob(
        order: usize,
        l1: f64,
        l2: f64,
        l3: f64,
        x: &[f64],
        y: &[f64],
    ) -> (f64, [f64; 4]) {
        // Note: l1, l2, l3 are barycentric coordinates
        // Conversion from (xi, eta) to (l1, l2, l3):
        // l1 = 1 - xi - eta
        // l2 = xi
        // l3 = eta
        // Therefore: dL1/dxi = -1, dL1/deta = -1
        //           dL2/dxi = 1,  dL2/deta = 0
        //           dL3/dxi = 0,  dL3/deta = 1

        match order {
            1 => {
                // Linear element (P1) with 3 nodes
                // Shape functions: N0 = L1, N1 = L2, N2 = L3
                // Derivatives w.r.t. reference coordinates (xi, eta):
                let dn_dxi = [
                    -1.0, // dN0/dξ = dL1/dξ
                    1.0,  // dN1/dξ = dL2/dξ
                    0.0,  // dN2/dξ = dL3/dξ
                ];
                let dn_deta = [
                    -1.0, // dN0/dη = dL1/dη
                    0.0,  // dN1/dη = dL2/dη
                    1.0,  // dN2/dη = dL3/dη
                ];

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
                let jacob_inv_t = [
                    dy_deta / jacob_det,
                    -dy_dxi / jacob_det,
                    -dx_deta / jacob_det,
                    dx_dxi / jacob_det,
                ];

                (jacob_det, jacob_inv_t)
            }
            2 => {
                // Quadratic element (P2) with 6 nodes
                // Node ordering for P2 triangle:
                // Vertices: 0, 1, 2
                // Edge midpoints: 3 (between 0-1), 4 (between 1-2), 5 (between 2-0)

                // Shape function derivatives for P2 elements
                // Vertex nodes use: Ni = Li * (2*Li - 1)
                // Edge nodes use: Nij = 4 * Li * Lj
                // Derivatives w.r.t. xi and eta using chain rule:
                let dn_dxi = [
                    -(4.0 * l1 - 1.0), // dN0/dxi (vertex 0)
                    4.0 * l2 - 1.0,    // dN1/dxi (vertex 1)
                    0.0,               // dN2/dxi (vertex 2)
                    4.0 * (l1 - l2),   // dN3/dxi (edge 0-1)
                    4.0 * l3,          // dN4/dxi (edge 1-2)
                    -4.0 * l3,         // dN5/dxi (edge 2-0)
                ];

                let dn_deta = [
                    -(4.0 * l1 - 1.0), // dN0/deta (vertex 0)
                    0.0,               // dN1/deta (vertex 1)
                    4.0 * l3 - 1.0,    // dN2/deta (vertex 2)
                    -4.0 * l2,         // dN3/deta (edge 0-1)
                    4.0 * l2,          // dN4/deta (edge 1-2)
                    4.0 * (l1 - l3),   // dN5/deta (edge 2-0)
                ];

                let mut dx_dxi = 0.0;
                let mut dx_deta = 0.0;
                let mut dy_dxi = 0.0;
                let mut dy_deta = 0.0;

                for i in 0..6 {
                    dx_dxi += dn_dxi[i] * x[i];
                    dx_deta += dn_deta[i] * x[i];
                    dy_dxi += dn_dxi[i] * y[i];
                    dy_deta += dn_deta[i] * y[i];
                }

                let jacob_det = dx_dxi * dy_deta - dx_deta * dy_dxi;
                let jacob_inv_t = [
                    dy_deta / jacob_det,
                    -dy_dxi / jacob_det,
                    -dx_deta / jacob_det,
                    dx_dxi / jacob_det,
                ];
                (jacob_det, jacob_inv_t)
            }
            _ => {
                panic!(
                    "evaluate_jacob: Only linear (order=1) and quadratic (order=2) elements are supported"
                );
            }
        }
    }
    fn get_global_dof_ids(
        mesh: &Mesh2d<TriangleElement>,
        ielem: usize,
        order: usize,
    ) -> Vec<usize> {
        if let Status::Active(elem) = &mesh.elements[ielem] {
            match order {
                1 => elem.inodes.to_vec(),
                2 => {
                    let mut global_dofs = elem.inodes.to_vec();
                    let nnodes = mesh.ref_nodes.len();
                    for &iedge in &elem.iedges {
                        global_dofs.push(nnodes + iedge);
                    }
                    global_dofs
                }
                _ => {
                    panic!(
                        "get_global_dof_id: Only linear (order=1) and quadratic (order=2) elements are supported"
                    );
                }
            }
        } else {
            panic!("get_global_dof_id: Element is not active");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disc::mesh::mesh2d::Node;

    fn create_simple_triangle_mesh() -> Mesh2d<TriangleElement> {
        use crate::disc::mesh::mesh2d::{Boundaries, Edge};
        
        let ref_nodes = vec![
            Status::Active(Node { x: 0.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 1.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 0.0, y: 1.0, parents: vec![] }),
        ];
        
        let phys_nodes = ref_nodes.clone();
        
        let edges = vec![
            Status::Active(Edge { inodes: vec![0, 1], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![1, 2], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![2, 0], parents: vec![], local_ids: vec![] }),
        ];
        
        let elements = vec![
            Status::Active(TriangleElement {
                inodes: [0, 1, 2],
                iedges: [0, 1, 2],
                ineighbors: vec![],
            }),
        ];
        
        Mesh2d {
            ref_nodes,
            phys_nodes,
            edges,
            elements,
            boundaries: Boundaries { constant: vec![], function: vec![], open: vec![], polynomial: vec![] },
            interior_edges: vec![],
            free_bnd_x: vec![],
            free_bnd_y: vec![],
            interior_nodes: vec![],
        }
    }

    fn create_two_triangle_mesh() -> Mesh2d<TriangleElement> {
        use crate::disc::mesh::mesh2d::{Boundaries, Edge};
        
        let ref_nodes = vec![
            Status::Active(Node { x: 0.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 1.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 1.0, y: 1.0, parents: vec![] }),
            Status::Active(Node { x: 0.0, y: 1.0, parents: vec![] }),
        ];
        
        let phys_nodes = ref_nodes.clone();
        
        let edges = vec![
            Status::Active(Edge { inodes: vec![0, 1], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![1, 2], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![2, 0], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![2, 3], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![3, 0], parents: vec![], local_ids: vec![] }),
        ];
        
        let elements = vec![
            Status::Active(TriangleElement {
                inodes: [0, 1, 2],
                iedges: [0, 1, 2],
                ineighbors: vec![1],
            }),
            Status::Active(TriangleElement {
                inodes: [0, 2, 3],
                iedges: [2, 3, 4],
                ineighbors: vec![0],
            }),
        ];
        
        Mesh2d {
            ref_nodes,
            phys_nodes,
            edges,
            elements,
            boundaries: Boundaries { constant: vec![], function: vec![], open: vec![], polynomial: vec![] },
            interior_edges: vec![2],
            free_bnd_x: vec![],
            free_bnd_y: vec![],
            interior_nodes: vec![],
        }
    }

    #[test]
    fn test_compute_stiffness_single_triangle() {
        let basis = TriangleCGBasis::new(1);
        let elliptic = LinearElliptic::new(basis);
        let mesh = create_simple_triangle_mesh();
        
        let stiffness = elliptic.compute_stiffness(&mesh);
        
        assert_eq!(stiffness.shape(), &[3, 3]);
        
        for i in 0..3 {
            for j in 0..3 {
                assert!(stiffness[(i, j)].is_finite());
            }
        }
        
        for i in 0..3 {
            for j in i+1..3 {
                assert!((stiffness[(i, j)] - stiffness[(j, i)]).abs() < 1e-10,
                        "Stiffness matrix should be symmetric");
            }
        }
        
        let row_sum: f64 = (0..3).map(|j| stiffness[(0, j)]).sum();
        assert!(row_sum.abs() < 1e-10, 
                "Row sum should be approximately zero for Neumann BC");
    }

    #[test]
    fn test_compute_stiffness_two_triangles() {
        let basis = TriangleCGBasis::new(1);
        let elliptic = LinearElliptic::new(basis);
        let mesh = create_two_triangle_mesh();
        
        let stiffness = elliptic.compute_stiffness(&mesh);
        
        assert_eq!(stiffness.shape(), &[4, 4]);
        
        for i in 0..4 {
            for j in 0..4 {
                assert!(stiffness[(i, j)].is_finite());
            }
        }
        
        for i in 0..4 {
            for j in i+1..4 {
                assert!((stiffness[(i, j)] - stiffness[(j, i)]).abs() < 1e-10,
                        "Stiffness matrix should be symmetric");
            }
        }
        
        // The stiffness matrix for this specific mesh configuration results in:
        // Nodes 0 and 3 are connected (both belong to the second triangle)
        // Nodes 1 and 3 are not connected (no shared elements)
        // Note: The specific values depend on the element areas and kappa scaling
        assert!(stiffness[(0, 3)].abs() > 1e-10, "Nodes 0 and 3 are connected through element 2");
        assert!(stiffness[(1, 3)].abs() < 1e-10, "Nodes 1 and 3 are not connected");
    }

    #[test]
    fn test_compute_stiffness_equilateral_triangle() {
        use crate::disc::mesh::mesh2d::{Boundaries, Edge};
        
        let basis = TriangleCGBasis::new(1);
        let elliptic = LinearElliptic::new(basis);
        
        let sqrt3_2 = (3.0_f64).sqrt() / 2.0;
        let ref_nodes = vec![
            Status::Active(Node { x: 0.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 1.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 0.5, y: sqrt3_2, parents: vec![] }),
        ];
        
        let phys_nodes = ref_nodes.clone();
        
        let edges = vec![
            Status::Active(Edge { inodes: vec![0, 1], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![1, 2], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![2, 0], parents: vec![], local_ids: vec![] }),
        ];
        
        let elements = vec![
            Status::Active(TriangleElement {
                inodes: [0, 1, 2],
                iedges: [0, 1, 2],
                ineighbors: vec![],
            }),
        ];
        
        let mesh = Mesh2d {
            ref_nodes,
            phys_nodes,
            edges,
            elements,
            boundaries: Boundaries { constant: vec![], function: vec![], open: vec![], polynomial: vec![] },
            interior_edges: vec![],
            free_bnd_x: vec![],
            free_bnd_y: vec![],
            interior_nodes: vec![],
        };
        
        let stiffness = elliptic.compute_stiffness(&mesh);
        
        assert!((stiffness[(0, 0)] - stiffness[(1, 1)]).abs() < 1e-10,
                "Diagonal entries should be equal for equilateral triangle");
        assert!((stiffness[(0, 0)] - stiffness[(2, 2)]).abs() < 1e-10,
                "Diagonal entries should be equal for equilateral triangle");
        
        assert!((stiffness[(0, 1)] - stiffness[(0, 2)]).abs() < 1e-10,
                "Off-diagonal entries should be equal for equilateral triangle");
        assert!((stiffness[(0, 1)] - stiffness[(1, 2)]).abs() < 1e-10,
                "Off-diagonal entries should be equal for equilateral triangle");
    }

    #[test]
    fn test_compute_stiffness_positive_definite() {
        let basis = TriangleCGBasis::new(1);
        let elliptic = LinearElliptic::new(basis);
        let mesh = create_simple_triangle_mesh();
        
        let stiffness = elliptic.compute_stiffness(&mesh);
        
        let diag_dominant = (0..3).all(|i| {
            let diag = stiffness[(i, i)].abs();
            let off_diag_sum: f64 = (0..3)
                .filter(|&j| j != i)
                .map(|j| stiffness[(i, j)].abs())
                .sum();
            diag >= off_diag_sum
        });
        
        assert!(diag_dominant, "Matrix should be diagonally dominant (weak test for positive semi-definiteness)");
    }

    #[test]
    fn test_compute_stiffness_inactive_elements() {
        let basis = TriangleCGBasis::new(1);
        let elliptic = LinearElliptic::new(basis);
        let mut mesh = create_two_triangle_mesh();
        
        mesh.elements[1] = Status::Removed;
        
        let stiffness = elliptic.compute_stiffness(&mesh);
        
        assert_eq!(stiffness.shape(), &[4, 4]);
        
        assert!(stiffness[(3, 3)].abs() < 1e-10, 
                "Node 3 contributions should be zero since its element is inactive");
        assert!(stiffness[(0, 3)].abs() < 1e-10 && stiffness[(3, 0)].abs() < 1e-10,
                "Connections to node 3 should be zero");
    }

    #[test]
    fn test_compute_stiffness_different_element_sizes() {
        use crate::disc::mesh::mesh2d::{Boundaries, Edge};
        
        let basis = TriangleCGBasis::new(1);
        let elliptic = LinearElliptic::new(basis);
        
        let ref_nodes = vec![
            Status::Active(Node { x: 0.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 2.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 0.0, y: 1.0, parents: vec![] }),
            Status::Active(Node { x: 1.0, y: 0.0, parents: vec![] }),
            Status::Active(Node { x: 0.0, y: 0.5, parents: vec![] }),
        ];
        
        let phys_nodes = ref_nodes.clone();
        
        let edges = vec![
            Status::Active(Edge { inodes: vec![0, 1], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![1, 2], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![2, 0], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![0, 3], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![3, 4], parents: vec![], local_ids: vec![] }),
            Status::Active(Edge { inodes: vec![4, 0], parents: vec![], local_ids: vec![] }),
        ];
        
        let elements = vec![
            Status::Active(TriangleElement {
                inodes: [0, 1, 2],
                iedges: [0, 1, 2],
                ineighbors: vec![],
            }),
            Status::Active(TriangleElement {
                inodes: [0, 3, 4],
                iedges: [3, 4, 5],
                ineighbors: vec![],
            }),
        ];
        
        let mesh = Mesh2d {
            ref_nodes,
            phys_nodes,
            edges,
            elements,
            boundaries: Boundaries { constant: vec![], function: vec![], open: vec![], polynomial: vec![] },
            interior_edges: vec![],
            free_bnd_x: vec![],
            free_bnd_y: vec![],
            interior_nodes: vec![],
        };
        
        let stiffness = elliptic.compute_stiffness(&mesh);
        
        assert_eq!(stiffness.shape(), &[5, 5]);
        
        for i in 0..5 {
            for j in 0..5 {
                assert!(stiffness[(i, j)].is_finite(), 
                        "All stiffness matrix entries should be finite");
            }
        }
        
        for i in 0..5 {
            for j in i+1..5 {
                assert!((stiffness[(i, j)] - stiffness[(j, i)]).abs() < 1e-10,
                        "Stiffness matrix should be symmetric");
            }
        }
    }
}
