use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array, s};
use ndarray_linalg::{Inverse, Solve};

use crate::disc::{
    basis::{Basis, lagrange1d::LobattoBasis},
    gauss_points::lobatto_points::get_lobatto_points_interval,
};
pub struct TriangleBasis {
    pub n: usize,
    pub r: Array1<f64>,
    pub s: Array1<f64>,
    pub vandermonde: Array2<f64>,
    pub inv_vandermonde: Array2<f64>,
    pub dr: Array2<f64>,
    pub ds: Array2<f64>,
    pub nodes_along_edges: Array2<usize>,
    pub quad_p: Array1<f64>,
    pub quad_w: Array1<f64>,
    pub cub_r: Array1<f64>,
    pub cub_s: Array1<f64>,
    pub cub_w: Array1<f64>,
    pub dr_cub: Array2<f64>,
    pub ds_cub: Array2<f64>,
    pub basis1d: LobattoBasis,
}
impl TriangleBasis {
    pub fn new(n: usize) -> Self {
        let (x, y) = Self::nodes2d(n);
        let (r, s) = Self::xy_to_rs(x.view(), y.view());
        println!("r: {:?}", r);
        println!("s: {:?}", s);
        let vandermonde = Self::vandermonde2d(n, r.view(), s.view());
        let inv_vandermonde = vandermonde.inv().unwrap();
        let (dr, ds) = Self::dmatrices_2d(n, r.view(), s.view(), vandermonde.view());
        let nodes_along_edges = Self::find_nodes_along_edges(n, r.view(), s.view());
        let quad_p = Self::jacobi_gauss_lobatto(0.0, 0.0, n);
        let (_, quad_w_vec) = get_lobatto_points_interval(n + 1);
        let quad_w = Array1::from_iter(quad_w_vec);
        let (cub_r, cub_s, cub_w) = Self::cubature_points(2 * n - 1);
        let (dr_cub, ds_cub) =
            Self::dmatrices_2d(n, cub_r.view(), cub_s.view(), vandermonde.view());
        let basis1d = LobattoBasis::new(n);
        Self {
            n,
            r,
            s,
            vandermonde,
            inv_vandermonde,
            dr,
            ds,
            nodes_along_edges,
            quad_p,
            quad_w,
            cub_r,
            cub_s,
            cub_w,
            dr_cub,
            ds_cub,
            basis1d,
        }
    }
    fn compute_l(v: ArrayView2<f64>) -> Array2<f64> {
        let mut l = Array2::<f64>::zeros((v.shape()[0], v.shape()[0]));
        let v_t_inv = v.t().inv().unwrap();
        for i in 0..l.shape()[1] {
            let p = v.row(i);
            l.column_mut(i).assign(&v_t_inv.dot(&p));
        }
        l
    }
    fn find_nodes_along_edges(n: usize, r: ArrayView1<f64>, s: ArrayView1<f64>) -> Array2<usize> {
        let node_tol = 1.0e-10;
        let nfp = n + 1;
        let fmask1 = r
            .iter()
            .enumerate()
            .filter_map(|(i, _)| {
                if (s[i] + 1.0).abs() < node_tol {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();
        let fmask2 = r
            .iter()
            .enumerate()
            .filter_map(|(i, &r_val)| {
                if (r_val + s[i]).abs() < node_tol {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();
        let mut fmask3 = r
            .iter()
            .enumerate()
            .filter_map(|(i, &r_val)| {
                if (r_val + 1.0).abs() < node_tol {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();
        fmask3.reverse();
        assert_eq!(fmask1.len(), nfp);
        assert_eq!(fmask2.len(), nfp);
        assert_eq!(fmask3.len(), nfp);
        /*
        // Verification code
        println!("=== Edge Node Verification (n={}) ===", n);

        // Verify Edge 0 (bottom edge: s = -1, should be ordered by increasing r)
        println!("Edge 0 (bottom, s=-1) nodes:");
        let mut prev_r = f64::NEG_INFINITY;
        for (idx, &node_id) in fmask1.iter().enumerate() {
            let r_val = r[node_id];
            let s_val = s[node_id];
            println!("  Node {}: r={:.6}, s={:.6}", node_id, r_val, s_val);

            // Verify node is on correct edge
            assert!(
                (s_val + 1.0).abs() < node_tol,
                "Node {} not on bottom edge",
                node_id
            );

            // Verify counterclockwise ordering (r should increase)
            if idx > 0 {
                assert!(
                    r_val >= prev_r - node_tol,
                    "Edge 0 nodes not in counterclockwise order"
                );
            }
            prev_r = r_val;
        }

        // Verify Edge 1 (diagonal edge: r + s = 0, should be ordered by decreasing r)
        println!("Edge 1 (diagonal, r+s=0) nodes:");
        let mut prev_r = f64::INFINITY;
        for (idx, &node_id) in fmask2.iter().enumerate() {
            let r_val = r[node_id];
            let s_val = s[node_id];
            println!("  Node {}: r={:.6}, s={:.6}", node_id, r_val, s_val);

            // Verify node is on correct edge
            assert!(
                (r_val + s_val).abs() < node_tol,
                "Node {} not on diagonal edge",
                node_id
            );

            // Verify counterclockwise ordering (r should decrease)
            if idx > 0 {
                assert!(
                    r_val <= prev_r + node_tol,
                    "Edge 1 nodes not in counterclockwise order"
                );
            }
            prev_r = r_val;
        }

        // Verify Edge 2 (left edge: r = -1, should be ordered by decreasing s)
        println!("Edge 2 (left, r=-1) nodes:");
        let mut prev_s = f64::INFINITY;
        for (idx, &node_id) in fmask3.iter().enumerate() {
            let r_val = r[node_id];
            let s_val = s[node_id];
            println!("  Node {}: r={:.6}, s={:.6}", node_id, r_val, s_val);

            // Verify node is on correct edge
            assert!(
                (r_val + 1.0).abs() < node_tol,
                "Node {} not on left edge",
                node_id
            );

            // Verify counterclockwise ordering (s should decrease)
            if idx > 0 {
                assert!(
                    s_val <= prev_s + node_tol,
                    "Edge 2 nodes not in counterclockwise order"
                );
            }
            prev_s = s_val;
        }
        */
        let mut nodes_along_edges = Array2::<usize>::zeros((3, nfp));
        nodes_along_edges
            .slice_mut(s![0, ..])
            .assign(&Array1::from_iter(fmask1));
        nodes_along_edges
            .slice_mut(s![1, ..])
            .assign(&Array1::from_iter(fmask2));
        nodes_along_edges
            .slice_mut(s![2, ..])
            .assign(&Array1::from_iter(fmask3));
        nodes_along_edges
    }
    fn rs_to_ab(r: ArrayView1<f64>, s: ArrayView1<f64>) -> (Array1<f64>, Array1<f64>) {
        let a = r
            .iter()
            .zip(s.iter())
            .map(|(&r_val, &s_val)| {
                if s_val != 1.0 {
                    2.0 * (1.0 + r_val) / (1.0 - s_val) - 1.0
                } else {
                    -1.0
                }
            })
            .collect::<Array1<f64>>();
        let b = s.to_owned();
        (a, b)
    }
    fn dubiner_basis(a: ArrayView1<f64>, b: ArrayView1<f64>, i: i32, j: i32) -> Array1<f64> {
        2.0_f64.sqrt()
            * Self::jacobi_polynomial(a, 0.0, 0.0, i)
            * Self::jacobi_polynomial(b, 2.0 * i as f64 + 1.0, 0.0, j)
            * (1.0 - &b).powi(i)
    }
    fn warp_factor(n: usize, r: ArrayView1<f64>) -> Array1<f64> {
        let lglr = Self::jacobi_gauss_lobatto(0.0, 0.0, n);
        let req = Array1::linspace(-1.0, 1.0, n + 1);
        let veq = Self::vandermonde1d(n, req.view());
        let nr = r.len();
        let mut pmat = Array2::<f64>::zeros((n + 1, nr));
        for i in 0..n + 1 {
            pmat.row_mut(i)
                .assign(&Self::jacobi_polynomial(r.clone(), 0.0, 0.0, i as i32));
        }
        let mut lmat = Array2::zeros((veq.shape()[1], pmat.shape()[1]));
        for i in 0..pmat.shape()[1] {
            let col = pmat.column(i);
            let sol = veq.t().solve(&col).unwrap();
            lmat.column_mut(i).assign(&sol);
        }
        assert_eq!(lglr.len(), req.len());
        let warp = lmat.t().dot(&(&lglr - &req));
        let zerof = r.mapv(|x| if x.abs() < 1.0 - 1.0e-10 { 1.0 } else { 0.0 });
        let sf = 1.0 - (&zerof * &r).mapv(|x| x.powi(2));
        let warp = &warp / &sf + &warp * &(zerof - 1.0);
        warp
    }

    fn xy_to_rs(x: ArrayView1<f64>, y: ArrayView1<f64>) -> (Array1<f64>, Array1<f64>) {
        let l1 = (3.0_f64.sqrt() * &y + 1.0) / 3.0;
        let l2 = (-3.0 * &x - 3.0_f64.sqrt() * &y + 2.0) / 6.0;
        let l3 = (3.0 * &x - 3.0_f64.sqrt() * &y + 2.0) / 6.0;
        let r = -&l2 + &l3 - &l1;
        let s = -&l2 - &l3 + &l1;
        (r, s)
    }
    fn grad_simplex_2d(
        a: ArrayView1<f64>,
        b: ArrayView1<f64>,
        id: i32,
        jd: i32,
    ) -> (Array1<f64>, Array1<f64>) {
        let fa = Self::jacobi_polynomial(a, 0.0, 0.0, id);
        let gb = Self::jacobi_polynomial(b, 2.0 * id as f64 + 1.0, 0.0, jd);
        let dfa = Self::grad_jacobi_polynomial(a, 0.0, 0.0, id);
        let dgb = Self::grad_jacobi_polynomial(b, 2.0 * id as f64 + 1.0, 0.0, jd);
        let mut dmode_dr = &dfa * &gb;
        if id > 0 {
            dmode_dr = (0.5 * (1.0 - &b)).powi(id - 1) * &dmode_dr;
        }
        let mut dmode_ds = &dfa * (&gb * (0.5 * (1.0 + &a)));
        if id > 0 {
            dmode_ds = (0.5 * (1.0 - &b)).powi(id - 1) * &dmode_ds;
        }
        let mut tmp = &dgb * ((0.5 * (1.0 - &b)).powi(id));
        if id > 0 {
            tmp = tmp - 0.5 * id as f64 * &gb * ((0.5 * (1.0 - &b)).powi(id - 1));
        }
        dmode_ds = dmode_ds + &fa * &tmp;
        dmode_dr = dmode_dr * 2.0_f64.powf(id as f64 + 0.5);
        dmode_ds = dmode_ds * 2.0_f64.powf(id as f64 + 0.5);

        (dmode_dr, dmode_ds)
    }
    fn lift_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
        nodes_along_edges: ArrayView2<usize>,
        v: ArrayView2<f64>,
    ) -> Array2<f64> {
        let np = (n + 1) * (n + 2) / 2;
        let nfp = n + 1;
        let nfaces = 3;
        let mut emat = Array2::<f64>::zeros((np, nfaces * nfp));
        // face 1
        let facer = r.select(
            Axis(0),
            nodes_along_edges.slice(s![0, ..]).as_slice().unwrap(),
        );
        let v1d = Self::vandermonde1d(n, facer.view());
        let mass_edge1 = v1d.t().dot(&v1d).inv().unwrap();
        for &i in nodes_along_edges.slice(s![0, ..]) {
            let mut emat_row = emat.row_mut(i);
            emat_row.slice_mut(s![0..nfp]).assign(&mass_edge1.row(i));
        }
        // face 2
        let facer = r.select(
            Axis(0),
            nodes_along_edges.slice(s![1, ..]).as_slice().unwrap(),
        );
        let v1d = Self::vandermonde1d(n, facer.view());
        let mass_edge2 = v1d.t().dot(&v1d).inv().unwrap();
        for &i in nodes_along_edges.slice(s![1, ..]) {
            let mut emat_row = emat.row_mut(i);
            emat_row
                .slice_mut(s![nfp..2 * nfp])
                .assign(&mass_edge2.row(i));
        }
        // face 3
        let faces = s.select(
            Axis(0),
            nodes_along_edges.slice(s![2, ..]).as_slice().unwrap(),
        );
        let v1d = Self::vandermonde1d(n, faces.view());
        let mass_edge3 = v1d.t().dot(&v1d).inv().unwrap();
        for &i in nodes_along_edges.slice(s![2, ..]) {
            let mut emat_row = emat.row_mut(i);
            emat_row
                .slice_mut(s![2 * nfp..3 * nfp])
                .assign(&mass_edge3.row(i));
        }
        let lift = v.t().dot(&(v.dot(&emat)));
        lift
    }
    pub fn cubature_points(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let (r, s, weight) = match n {
            2 => {
                let r = array![0.16666666666667, 0.16666666666667, 0.66666666666667];
                let s = array![0.16666666666667, 0.66666666666667, 0.16666666666667];
                let weight = array![0.33333333333333, 0.33333333333333, 0.33333333333333];
                (r, s, weight)
            }
            3 => {
                let r = array![
                    0.33333333333333,
                    0.20000000000000,
                    0.20000000000000,
                    0.60000000000000
                ];
                let s = array![
                    0.33333333333333,
                    0.20000000000000,
                    0.60000000000000,
                    0.20000000000000
                ];
                let weight = array![
                    -0.56250000000000,
                    0.52083333333333,
                    0.52083333333333,
                    0.52083333333333
                ];
                (r, s, weight)
            }
            4 => {
                let r = array![
                    0.44594849091597,
                    0.44594849091597,
                    0.10810301816807,
                    0.09157621350977,
                    0.09157621350977,
                    0.81684757298046
                ];
                let s = array![
                    0.44594849091597,
                    0.10810301816807,
                    0.44594849091597,
                    0.09157621350977,
                    0.81684757298046,
                    0.09157621350977
                ];
                let weight = array![
                    0.22338158967801,
                    0.22338158967801,
                    0.22338158967801,
                    0.10995174365532,
                    0.10995174365532,
                    0.10995174365532
                ];
                (r, s, weight)
            }
            5 => {
                let r = array![
                    0.33333333333333,
                    0.47014206410511,
                    0.47014206410511,
                    0.05971587178977,
                    0.10128650732346,
                    0.10128650732346,
                    0.79742698535309
                ];
                let s = array![
                    0.33333333333333,
                    0.47014206410511,
                    0.05971587178977,
                    0.47014206410511,
                    0.10128650732346,
                    0.79742698535309,
                    0.10128650732346
                ];
                let weight = array![
                    0.22500000000000,
                    0.13239415278851,
                    0.13239415278851,
                    0.13239415278851,
                    0.12593918054483,
                    0.12593918054483,
                    0.12593918054483
                ];
                assert!(r.len() == s.len() && r.len() == weight.len());
                (r, s, weight)
            }
            6 => {
                let r = array![
                    0.24928674517091,
                    0.24928674517091,
                    0.50142650965818,
                    0.06308901449150,
                    0.06308901449150,
                    0.87382197101700,
                    0.31035245103378,
                    0.63650249912140,
                    0.05314504984482,
                    0.63650249912140,
                    0.31035245103378,
                    0.05314504984482
                ];
                let s = array![
                    0.24928674517091,
                    0.50142650965818,
                    0.24928674517091,
                    0.06308901449150,
                    0.87382197101700,
                    0.06308901449150,
                    0.63650249912140,
                    0.05314504984482,
                    0.31035245103378,
                    0.31035245103378,
                    0.05314504984482,
                    0.63650249912140
                ];
                let weight = array![
                    0.11678627572638,
                    0.11678627572638,
                    0.11678627572638,
                    0.05084490637021,
                    0.05084490637021,
                    0.05084490637021,
                    0.08285107561837,
                    0.08285107561837,
                    0.08285107561837,
                    0.08285107561837,
                    0.08285107561837,
                    0.08285107561837
                ];
                assert!(r.len() == s.len() && r.len() == weight.len());
                (r, s, weight)
            }
            7 => {
                let r = array![
                    0.33333333333333,
                    0.26034596607904,
                    0.26034596607904,
                    0.47930806784192,
                    0.06513010290222,
                    0.06513010290222,
                    0.86973979419557,
                    0.31286549600487,
                    0.63844418856981,
                    0.04869031542532,
                    0.63844418856981,
                    0.31286549600487,
                    0.04869031542532
                ];
                let s = array![
                    0.33333333333333,
                    0.26034596607904,
                    0.47930806784192,
                    0.26034596607904,
                    0.06513010290222,
                    0.86973979419557,
                    0.06513010290222,
                    0.63844418856981,
                    0.04869031542532,
                    0.31286549600487,
                    0.31286549600487,
                    0.04869031542532,
                    0.63844418856981
                ];
                let weight = array![
                    -0.14957004446768,
                    0.17561525743321,
                    0.17561525743321,
                    0.17561525743321,
                    0.05334723560884,
                    0.05334723560884,
                    0.05334723560884,
                    0.07711376089026,
                    0.07711376089026,
                    0.07711376089026,
                    0.07711376089026,
                    0.07711376089026,
                    0.07711376089026
                ];
                assert!(r.len() == s.len() && r.len() == weight.len());
                (r, s, weight)
            }
            8 => {
                let r = array![
                    0.33333333333333,
                    0.14431560767779,
                    0.45929258829272,
                    0.09509163426728,
                    0.08141482341455,
                    0.45929258829272,
                    0.17056930775176,
                    0.10321737053472,
                    0.65886138449648,
                    0.17056930775176,
                    0.05054722831703,
                    0.03245849762320,
                    0.89890554336594,
                    0.05054722831703,
                    0.26311282963464,
                    0.02723031417443
                ];
                let s = array![
                    0.33333333333333,
                    0.45929258829272,
                    0.09509163426728,
                    0.45929258829272,
                    0.08141482341455,
                    0.09509163426728,
                    0.17056930775176,
                    0.65886138449648,
                    0.10321737053472,
                    0.10321737053472,
                    0.05054722831703,
                    0.89890554336594,
                    0.03245849762320,
                    0.03245849762320,
                    0.72849239295540,
                    0.00839477740996
                ];
                let weight = array![
                    0.00839477740996,
                    0.26311282963464,
                    0.02723031417443,
                    0.72849239295540,
                    0.26311282963464,
                    0.02723031417443,
                    0.26311282963464,
                    0.00839477740996,
                    0.02723031417443,
                    0.00839477740996,
                    0.72849239295540,
                    0.02723031417443,
                    0.17056930775176,
                    0.17056930775176,
                    0.10321737053472,
                    0.17056930775176
                ];
                assert!(r.len() == s.len() && r.len() == weight.len());
                (r, s, weight)
            }
            _ => {
                panic!("Number of points not supported");
            }
        };
        let r_new = 2.0 * &r - 1.0;
        let s_new = 2.0 * &s - 1.0;
        let weight_scaled = &weight * 2.0;
        (r_new, s_new, weight_scaled)
    }
    pub fn validate_dmatrices(&self, n: usize) -> bool {
        println!("=== Validating Differentiation Matrices ===");
        let tolerance = 1e-12;
        let mut all_tests_passed = true;

        // Test 1: Fundamental property - Dr * V = Vr and Ds * V = Vs
        println!("Test 1: Fundamental differentiation matrix property");
        let (vr, vs) = Self::grad_vandermonde_2d(n, self.r.view(), self.s.view());

        let dr_v = self.dr.dot(&self.vandermonde);
        let ds_v = self.ds.dot(&self.vandermonde);

        let dr_error = (&dr_v - &vr).iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let ds_error = (&ds_v - &vs).iter().map(|&x| x.abs()).fold(0.0, f64::max);

        println!("  Max |Dr*V - Vr|: {:.2e}", dr_error);
        println!("  Max |Ds*V - Vs|: {:.2e}", ds_error);

        if dr_error > tolerance || ds_error > tolerance {
            println!("  ❌ FAILED: Fundamental matrix property");
            all_tests_passed = false;
        } else {
            println!("  ✅ PASSED: Fundamental matrix property");
        }

        // Test 2: Differentiation of basis functions
        println!("\nTest 2: Basis function differentiation exactness");
        let np = (n + 1) * (n + 2) / 2;
        let mut basis_test_passed = true;

        for k in 0..np {
            // Create a function that is 1 at node k and 0 at all other nodes
            let mut nodal_function = Array1::zeros(self.r.len());
            nodal_function[k] = 1.0;

            // Apply differentiation matrices
            let dr_basis = self.dr.dot(&nodal_function);
            let ds_basis = self.ds.dot(&nodal_function);

            // The result should equal the k-th column of Vr and Vs divided by the inverse Vandermonde
            let expected_dr = vr.column(k);
            let expected_ds = vs.column(k);

            let dr_basis_error = (&dr_basis - &expected_dr)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);
            let ds_basis_error = (&ds_basis - &expected_ds)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);

            if dr_basis_error > tolerance || ds_basis_error > tolerance {
                println!(
                    "  ❌ FAILED: Basis function {} differentiation, dr_error: {:.2e}, ds_error: {:.2e}",
                    k, dr_basis_error, ds_basis_error
                );
                basis_test_passed = false;
                all_tests_passed = false;
            }
        }

        if basis_test_passed {
            println!(
                "  ✅ PASSED: All {} basis functions differentiated correctly",
                np
            );
        }

        // Test 3: Polynomial exactness using the actual Dubiner basis functions
        println!("\nTest 3: Dubiner basis polynomial differentiation");
        let (a, b) = Self::rs_to_ab(self.r.view(), self.s.view());
        let mut dubiner_test_passed = true;

        for i in 0..=n {
            for j in 0..=(n - i) {
                // Create the Dubiner basis function
                let basis_func = Self::dubiner_basis(a.view(), b.view(), i as i32, j as i32);

                // Apply differentiation matrices
                let dr_dubiner = self.dr.dot(&basis_func);
                let ds_dubiner = self.ds.dot(&basis_func);

                // Expected derivatives from analytical computation
                let (expected_dr, expected_ds) =
                    Self::grad_simplex_2d(a.view(), b.view(), i as i32, j as i32);

                let dr_dubiner_error = (&dr_dubiner - &expected_dr)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);
                let ds_dubiner_error = (&ds_dubiner - &expected_ds)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);

                if dr_dubiner_error > tolerance || ds_dubiner_error > tolerance {
                    println!(
                        "  ❌ FAILED: Dubiner basis P_{{{},{}}} differentiation, dr_error: {:.2e}, ds_error: {:.2e}",
                        i, j, dr_dubiner_error, ds_dubiner_error
                    );
                    dubiner_test_passed = false;
                    all_tests_passed = false;
                }
            }
        }

        if dubiner_test_passed {
            println!("  ✅ PASSED: All Dubiner basis functions differentiated exactly");
        }

        // Test 4: Simple coordinate functions (these should be exactly representable)
        println!("\nTest 4: Coordinate function derivatives");

        // Test linear function: r (this should be in the polynomial space for n >= 1)
        if n >= 1 {
            // Project r onto the polynomial space
            let r_coeffs = self.inv_vandermonde.dot(&self.r);
            let r_projected = self.vandermonde.dot(&r_coeffs);

            // Check if r is well-represented
            let r_projection_error = (&r_projected - &self.r)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);

            if r_projection_error < tolerance {
                let dr_r = self.dr.dot(&r_projected);
                let ds_r = self.ds.dot(&r_projected);

                // For the projected r, ∂r/∂r should be approximately 1, ∂r/∂s should be approximately 0
                let expected_dr_r: Array1<f64> = Array1::ones(self.r.len());
                let expected_ds_r: Array1<f64> = Array1::zeros(self.r.len());

                let dr_r_error = (&dr_r - &expected_dr_r)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);
                let ds_r_error = (&ds_r - &expected_ds_r)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);

                println!(
                    "  r-coordinate: Max |dr(r) - 1|: {:.2e}, Max |ds(r) - 0|: {:.2e}",
                    dr_r_error, ds_r_error
                );

                if dr_r_error > 1e-10 || ds_r_error > 1e-10 {
                    println!(
                        "  ❌ FAILED: r-coordinate differentiation (within projection tolerance)"
                    );
                    all_tests_passed = false;
                } else {
                    println!("  ✅ PASSED: r-coordinate differentiation");
                }
            } else {
                println!(
                    "  ⚠️ WARNING: r-coordinate not exactly representable in polynomial space (error: {:.2e})",
                    r_projection_error
                );
            }
        }

        // Test linear function: s
        if n >= 1 {
            let s_coeffs = self.inv_vandermonde.dot(&self.s);
            let s_projected = self.vandermonde.dot(&s_coeffs);

            let s_projection_error = (&s_projected - &self.s)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);

            if s_projection_error < tolerance {
                let dr_s = self.dr.dot(&s_projected);
                let ds_s = self.ds.dot(&s_projected);

                let expected_dr_s: Array1<f64> = Array1::zeros(self.s.len());
                let expected_ds_s: Array1<f64> = Array1::ones(self.s.len());

                let dr_s_error = (&dr_s - &expected_dr_s)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);
                let ds_s_error = (&ds_s - &expected_ds_s)
                    .iter()
                    .map(|&x| x.abs())
                    .fold(0.0, f64::max);

                println!(
                    "  s-coordinate: Max |dr(s) - 0|: {:.2e}, Max |ds(s) - 1|: {:.2e}",
                    dr_s_error, ds_s_error
                );

                if dr_s_error > 1e-10 || ds_s_error > 1e-10 {
                    println!(
                        "  ❌ FAILED: s-coordinate differentiation (within projection tolerance)"
                    );
                    all_tests_passed = false;
                } else {
                    println!("  ✅ PASSED: s-coordinate differentiation");
                }
            } else {
                println!(
                    "  ⚠️ WARNING: s-coordinate not exactly representable in polynomial space (error: {:.2e})",
                    s_projection_error
                );
            }
        }

        // Test 5: Matrix properties
        println!("\nTest 5: Matrix properties");
        let np = self.r.len();

        if self.dr.shape() != [np, np] || self.ds.shape() != [np, np] {
            println!("  ❌ FAILED: Matrix dimensions incorrect");
            all_tests_passed = false;
        } else {
            println!("  ✅ PASSED: Matrix dimensions correct ({} x {})", np, np);
        }

        // Test 6: Conservation property (sum of each row should be derivative of constant = 0)
        println!("\nTest 6: Conservation property");
        let dr_row_sums: Array1<f64> = self.dr.sum_axis(Axis(1));
        let ds_row_sums: Array1<f64> = self.ds.sum_axis(Axis(1));

        let dr_conservation_error = dr_row_sums.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let ds_conservation_error = ds_row_sums.iter().map(|&x| x.abs()).fold(0.0, f64::max);

        println!("  Max |sum(Dr[i,:])|: {:.2e}", dr_conservation_error);
        println!("  Max |sum(Ds[i,:])|: {:.2e}", ds_conservation_error);

        if dr_conservation_error > tolerance || ds_conservation_error > tolerance {
            println!("  ❌ FAILED: Conservation property");
            all_tests_passed = false;
        } else {
            println!("  ✅ PASSED: Conservation property");
        }

        println!("\n=== Validation Summary ===");
        if all_tests_passed {
            println!("✅ ALL TESTS PASSED: Differentiation matrices are valid");
        } else {
            println!("❌ SOME TESTS FAILED: Differentiation matrices may have issues");
        }

        all_tests_passed
    }
    pub fn validate_modal_derivatives(&self, n_poly_order: usize, tolerance: f64) {
        println!(
            "\n=== Validating Modal Derivatives (grad_vandermonde_2d) for N={} ===",
            n_poly_order
        );

        // The derivatives will be validated at the basis's own nodal points (self.r, self.s)
        // If grad_vandermonde_2d was called with different points (e.g., cubature points),
        // this validation should use those same points.
        let test_r_view = self.r.view();
        let test_s_view = self.s.view();

        let (v2dr, v2ds) = Self::grad_vandermonde_2d(n_poly_order, test_r_view, test_s_view);

        let num_points = test_r_view.len();
        if num_points == 0 {
            println!(
                "  Skipping modal derivative validation: no test points provided (self.r is empty)."
            );
            return;
        }
        let num_basis_fns = (n_poly_order + 1) * (n_poly_order + 2) / 2;

        // Test 1: Derivative of the constant modal basis function (P_00, mode index 0)
        // Its derivatives dP/dr and dP/ds should be zero.
        println!("--- Test 1: Constant Mode (P_00) Derivative Check ---");
        let mut const_mode_passed = true;
        if num_basis_fns > 0 {
            for k_point in 0..num_points {
                if v2dr[(k_point, 0)].abs() > tolerance {
                    println!(
                        "  ❌ FAIL (P_00): dP/dr at point {} (r={:.2e}, s={:.2e}) is {:.2e} (should be ~0)",
                        k_point,
                        self.r[k_point],
                        self.s[k_point],
                        v2dr[(k_point, 0)]
                    );
                    const_mode_passed = false;
                }
                if v2ds[(k_point, 0)].abs() > tolerance {
                    println!(
                        "  ❌ FAIL (P_00): dP/ds at point {} (r={:.2e}, s={:.2e}) is {:.2e} (should be ~0)",
                        k_point,
                        self.r[k_point],
                        self.s[k_point],
                        v2ds[(k_point, 0)]
                    );
                    const_mode_passed = false;
                }
            }
        } else {
            const_mode_passed = false; // Or handle as N/A
            println!("  Skipping constant mode check: num_basis_fns is 0.");
        }
        if const_mode_passed {
            println!(
                "  ✅ PASS (P_00): Derivatives are zero within tolerance {:.2e}.",
                tolerance
            );
        }

        // Test 2: Finite difference check for a few selected modes and points
        println!("--- Test 2: Finite Difference Check (FD Epsilon = 1e-7) ---");
        let eps = 1e-7; // Epsilon for finite differences
        // Tolerance for FD might need to be looser, e.g., tolerance * 100 or 1e-6
        let fd_tolerance = tolerance.max(1e-6);

        // Select modes to test: first, P_10-like, P_01-like, and last, if they exist.
        let mut modes_to_test: Vec<usize> = vec![];
        if num_basis_fns > 0 {
            modes_to_test.push(0);
        } // P_00

        let mut current_sk = 0;
        let mut p10_sk: Option<usize> = None;
        let mut p01_sk: Option<usize> = None;

        for i_order in 0..=n_poly_order {
            for j_order in 0..=(n_poly_order - i_order) {
                if i_order == 1 && j_order == 0 {
                    p10_sk = Some(current_sk);
                }
                if i_order == 0 && j_order == 1 {
                    p01_sk = Some(current_sk);
                }
                current_sk += 1;
            }
        }
        if let Some(sk) = p10_sk {
            modes_to_test.push(sk);
        }
        if let Some(sk) = p01_sk {
            modes_to_test.push(sk);
        }
        if num_basis_fns > 1 {
            modes_to_test.push(num_basis_fns - 1);
        } // Last mode

        modes_to_test.sort();
        modes_to_test.dedup();

        // Select points to test: first, middle, last (if distinct)
        let mut points_to_test_indices: Vec<usize> = vec![0];
        if num_points > 1 {
            points_to_test_indices.push(num_points / 2);
            points_to_test_indices.push(num_points - 1);
        }
        points_to_test_indices.sort();
        points_to_test_indices.dedup();

        for &m_basis_idx in &modes_to_test {
            if m_basis_idx >= num_basis_fns {
                continue;
            } // Skip if mode index is out of bounds

            let mut mode_fd_overall_passed = true;
            println!("  Testing Mode Index: {}", m_basis_idx);

            for &k_point_idx in &points_to_test_indices {
                let r_k = self.r[k_point_idx];
                let s_k = self.s[k_point_idx];

                // Analytical derivatives from v2dr, v2ds
                let anal_dvdr = v2dr[(k_point_idx, m_basis_idx)];
                let anal_dvds = v2ds[(k_point_idx, m_basis_idx)];

                // Numerical derivative d/dr using central differences
                let r_plus_eps_arr = Array1::from(vec![r_k + eps]);
                let r_minus_eps_arr = Array1::from(vec![r_k - eps]);
                let s_arr_for_dr = Array1::from(vec![s_k]);
                let phi_r_plus =
                    Self::vandermonde2d(n_poly_order, r_plus_eps_arr.view(), s_arr_for_dr.view())
                        [(0, m_basis_idx)];
                let phi_r_minus =
                    Self::vandermonde2d(n_poly_order, r_minus_eps_arr.view(), s_arr_for_dr.view())
                        [(0, m_basis_idx)];
                let num_dvdr = (phi_r_plus - phi_r_minus) / (2.0 * eps);

                // Numerical derivative d/ds using central differences
                let r_arr_for_ds = Array1::from(vec![r_k]);
                let s_plus_eps_arr = Array1::from(vec![s_k + eps]);
                let s_minus_eps_arr = Array1::from(vec![s_k - eps]);
                let phi_s_plus =
                    Self::vandermonde2d(n_poly_order, r_arr_for_ds.view(), s_plus_eps_arr.view())
                        [(0, m_basis_idx)];
                let phi_s_minus =
                    Self::vandermonde2d(n_poly_order, r_arr_for_ds.view(), s_minus_eps_arr.view())
                        [(0, m_basis_idx)];
                let num_dvds = (phi_s_plus - phi_s_minus) / (2.0 * eps);

                let dr_abs_error = (anal_dvdr - num_dvdr).abs();
                let ds_abs_error = (anal_dvds - num_dvds).abs();

                let dr_rel_error = if anal_dvdr.abs() > 1e-9 {
                    dr_abs_error / anal_dvdr.abs()
                } else {
                    dr_abs_error
                };
                let ds_rel_error = if anal_dvds.abs() > 1e-9 {
                    ds_abs_error / anal_dvds.abs()
                } else {
                    ds_abs_error
                };

                let pass_dr = dr_rel_error < fd_tolerance || dr_abs_error < eps.sqrt(); // Looser check for FD
                let pass_ds = ds_rel_error < fd_tolerance || ds_abs_error < eps.sqrt();

                if !pass_dr {
                    mode_fd_overall_passed = false;
                    println!(
                        "    ❌ dP/dr @ pt {} (r={:.2e},s={:.2e}): Anal={:.4e}, Num={:.4e}, AbsErr={:.2e}, RelErr={:.2e}",
                        k_point_idx, r_k, s_k, anal_dvdr, num_dvdr, dr_abs_error, dr_rel_error
                    );
                }
                if !pass_ds {
                    mode_fd_overall_passed = false;
                    println!(
                        "    ❌ dP/ds @ pt {} (r={:.2e},s={:.2e}): Anal={:.4e}, Num={:.4e}, AbsErr={:.2e}, RelErr={:.2e}",
                        k_point_idx, r_k, s_k, anal_dvds, num_dvds, ds_abs_error, ds_rel_error
                    );
                }
            }
            if mode_fd_overall_passed {
                println!(
                    "    ✅ PASS: FD check for mode {} consistent within tolerance {:.2e}.",
                    m_basis_idx, fd_tolerance
                );
            }
        }
        println!("=== Modal Derivative Validation Complete ===");
    }
}

impl Basis for TriangleBasis {
    fn vandermonde2d(n: usize, r: ArrayView1<f64>, s: ArrayView1<f64>) -> Array2<f64> {
        let mut v = Array2::<f64>::zeros((r.len(), (n + 1) * (n + 2) / 2));
        let (a, b) = Self::rs_to_ab(r, s);
        let mut sk: usize = 0;
        for i in 0..n + 1 {
            for j in 0..n + 1 - i {
                v.column_mut(sk).assign(&Self::dubiner_basis(
                    a.view(),
                    b.view(),
                    i as i32,
                    j as i32,
                ));
                sk += 1;
            }
        }
        v
    }
    fn grad_vandermonde_2d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        if n == 0 {
            let num_points = r.len();
            return (
                Array2::zeros((num_points, 1)),
                Array2::zeros((num_points, 1)),
            );
        }

        let mut v2dr = Array2::<f64>::zeros((r.len(), (n + 1) * (n + 2) / 2));
        let mut v2ds = Array2::<f64>::zeros((r.len(), (n + 1) * (n + 2) / 2));
        let (a, b) = Self::rs_to_ab(r, s);
        let mut sk: usize = 0;
        for i in 0..n + 1 {
            for j in 0..n + 1 - i {
                let (v2dr_col, v2ds_col) =
                    Self::grad_simplex_2d(a.view(), b.view(), i as i32, j as i32);
                v2dr.column_mut(sk).assign(&v2dr_col);
                v2ds.column_mut(sk).assign(&v2ds_col);
                sk += 1;
            }
        }
        (v2dr, v2ds)
    }
    fn nodes2d(n: usize) -> (Array1<f64>, Array1<f64>) {
        if n == 0 {
            return (array![0.0], array![0.0]);
        }

        let alpopt = [
            0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832, 1.3648, 1.4773, 1.4959,
            1.5743, 1.5770, 1.6223, 1.6258,
        ];
        let alpha = if n < 16 { alpopt[n] } else { 5.0 / 3.0 };
        let np = (n + 1) * (n + 2) / 2;

        let mut l1 = Array1::<f64>::zeros(np);
        let mut l2 = Array1::<f64>::zeros(np);
        let mut l3 = Array1::<f64>::zeros(np);
        let mut sk: usize = 0;
        for i in 0..n + 1 {
            for j in 0..n + 1 - i {
                l1[sk] = i as f64 / n as f64;
                l3[sk] = j as f64 / n as f64;
                l2[sk] = 1.0 - l1[sk] - l3[sk];
                sk += 1;
            }
        }
        let mut x = -&l2 + &l3;
        let mut y = (-&l2 - &l3 + 2.0 * &l1) / 3.0_f64.sqrt();

        let blend1 = 4.0 * &l2 * &l3;
        let blend2 = 4.0 * &l1 * &l3;
        let blend3 = 4.0 * &l1 * &l2;

        let warpf1 = Self::warp_factor(n, (&l3 - &l2).view());
        let warpf2 = Self::warp_factor(n, (&l1 - &l3).view());
        let warpf3 = Self::warp_factor(n, (&l2 - &l1).view());

        let warp1 = blend1 * warpf1 * (1.0 + (alpha * l1).powi(2));
        let warp2 = blend2 * warpf2 * (1.0 + (alpha * l2).powi(2));
        let warp3 = blend3 * warpf3 * (1.0 + (alpha * l3).powi(2));

        x = x + &warp1 + (2.0 * PI / 3.0).cos() * &warp2 + (4.0 * PI / 3.0).cos() * &warp3;
        y = y + (2.0 * PI / 3.0).sin() * &warp2 + (4.0 * PI / 3.0).sin() * &warp3;
        (x, y)
    }
}
