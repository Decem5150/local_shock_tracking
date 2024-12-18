use ndarray::{Array, Ix1, Ix2};
use hashbrown::HashMap;
use super::super::gauss_points::lobatto_points::get_lobatto_points_interval;
use super::super::gauss_points::legendre_points::get_legendre_points_interval;

pub struct LagrangeBasis1D {
    pub cell_gauss_points: Array<f64, Ix1>,
    pub cell_gauss_weights: Array<f64, Ix1>,
    pub phis_cell_gps: Array<f64, Ix2>,
    pub dphis_cell_gps: Array<HashMap<usize, f64>, Ix2>,
}
impl LagrangeBasis1D {
    pub fn new(cell_gp_num: usize) -> LagrangeBasis1D {
        let dofs = cell_gp_num;
        let (cell_gauss_points, cell_gauss_weights) = get_legendre_points_interval(dofs);
        let mut phis_cell_gps = Array::zeros((dofs, dofs));
        let mut dphis_cell_gps = Array::from_elem((dofs, dofs), HashMap::new());
        // compute the basis functions at the gauss points.
        for j in 0..dofs {
            for i in 0..dofs {
                let mut product = 1.0;
                for m in 0..dofs {
                    if m != j {
                        product *= (cell_gauss_points[i] - cell_gauss_points[m]) / (cell_gauss_points[j] - cell_gauss_points[m]);
                    }
                }
                phis_cell_gps[(i, j)] = product;
            }
        }
        // compute the derivatives of the basis functions at the gauss points.
        for j in 0..dofs {
            for i in 0..dofs {
                let mut sum = 0.0;
                for m in 0..dofs {
                    if m != j {
                        sum += 1.0 / (cell_gauss_points[i] - cell_gauss_points[m]);
                    }
                }
                dphis_cell_gps[(i, j)].insert(1, sum * phis_cell_gps[(i, j)]);
            }
        }
        LagrangeBasis1D {
            cell_gauss_points,
            cell_gauss_weights,
            phis_cell_gps,
            dphis_cell_gps,
        }
    }
}