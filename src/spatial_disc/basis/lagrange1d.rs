use ndarray::{Array, Ix1, Ix2};
use hashbrown::HashMap;
use super::super::gauss_points::lobatto_points::get_lobatto_points_interval;

pub struct LagrangeBasis1DLobatto {
    pub cell_gauss_points: Array<f64, Ix1>,
    pub cell_gauss_weights: Array<f64, Ix1>,
    pub phis_cell_gps: Array<f64, Ix2>,
    pub dphis_cell_gps: Array<HashMap<usize, f64>, Ix2>,
}
impl LagrangeBasis1DLobatto {
    pub fn new(number_of_cell_gp: usize) -> LagrangeBasis1DLobatto {
        let dofs = number_of_cell_gp;
        let (cell_gauss_points, cell_gauss_weights) = get_lobatto_points_interval(number_of_cell_gp);
        let mut phis_cell_gps = Array::zeros((number_of_cell_gp, number_of_cell_gp));
        let mut dphis_cell_gps = Array::from_elem((number_of_cell_gp, number_of_cell_gp), HashMap::new());
        // compute the basis functions at the gauss points.
        for j in 0..dofs {
            for i in 0..number_of_cell_gp {
                let mut product = 1.0;
                for m in 0..number_of_cell_gp {
                    if m != j {
                        product *= (cell_gauss_points[i] - cell_gauss_points[m]) / (cell_gauss_points[j] - cell_gauss_points[m]);
                    }
                }
                phis_cell_gps[(i, j)] = product;
            }
        }
        // compute the derivatives of the basis functions at the gauss points.
        for j in 0..dofs {
            for i in 0..number_of_cell_gp {
                let mut sum = 0.0;
                for m in 0..number_of_cell_gp {
                    if m != j {
                        sum += 1.0 / (cell_gauss_points[i] - cell_gauss_points[m]);
                    }
                }
                dphis_cell_gps[(i, j)].insert(1, sum * phis_cell_gps[(i, j)]);
            }
        }
        LagrangeBasis1DLobatto {
            cell_gauss_points,
            cell_gauss_weights,
            phis_cell_gps,
            dphis_cell_gps,
        }
    }
}