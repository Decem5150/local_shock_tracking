use local_shock_tracking::disc::cg_basis::triangle::TriangleCGBasis;
use local_shock_tracking::disc::cg_basis::CGBasis2D;
use ndarray::Array1;

fn main() {
    // Create a CG basis for polynomial order 2
    let basis = TriangleCGBasis::new(2);
    
    println!("CG Triangle Basis with polynomial order 2");
    println!("Number of nodes: {}", basis.xi.len());
    println!("Node coordinates:");
    for i in 0..basis.xi.len() {
        println!("  Node {}: ({:.3}, {:.3})", i, basis.xi[i], basis.eta[i]);
    }
    
    println!("\nEdge nodes:");
    for edge in 0..3 {
        print!("  Edge {}: nodes ", edge);
        for j in 0..basis.n + 1 {
            print!("{} ", basis.edge_nodes[[edge, j]]);
        }
        println!();
    }
    
    // Test shape functions at a point
    let test_r = Array1::from(vec![0.3]);
    let test_s = Array1::from(vec![0.3]);
    let phi = TriangleCGBasis::shape_functions(2, test_r.view(), test_s.view());
    
    println!("\nShape functions at (0.3, 0.3):");
    for i in 0..phi.ncols() {
        println!("  N_{}: {:.6}", i, phi[[0, i]]);
    }
    println!("  Sum: {:.6}", phi.row(0).sum());
}