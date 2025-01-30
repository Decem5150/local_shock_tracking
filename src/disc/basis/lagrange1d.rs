use super::super::gauss_points::legendre_points::get_legendre_points_interval;
use ndarray::{Array, Ix1, Ix2};
use std::collections::HashMap;

pub struct LagrangeBasis1D {
    pub cell_gauss_points: Array<f64, Ix1>,
    pub cell_gauss_weights: Array<f64, Ix1>,
    pub phis_cell_gps: Array<f64, Ix2>,
    pub dphis_cell_gps: Array<f64, Ix2>,
    pub phis_bnd_gps: Array<f64, Ix2>,
    pub dphis_bnd_gps: Array<f64, Ix2>,
}

pub struct LagrangeBasis1DLobatto {
    pub dphis_cell_gps: Array<HashMap<usize, f64>, Ix2>,
    // ... other fields
}

impl LagrangeBasis1D {
    pub fn new(cell_gp_num: usize) -> LagrangeBasis1D {
        let dofs = cell_gp_num;
        let (cell_gauss_points, cell_gauss_weights) = get_legendre_points_interval(dofs);
        let mut phis_cell_gps = Array::zeros((dofs, dofs));
        let mut dphis_cell_gps = Array::zeros((dofs, dofs));
        let mut phis_bnd_gps = Array::zeros((2, dofs));
        let mut dphis_bnd_gps = Array::zeros((2, dofs));
        // compute the basis functions at the gauss points
        for i in 0..dofs {
            for j in 0..dofs {
                phis_cell_gps[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
        // compute the basis functions at the boundary gauss points (-1 and 1).
        let boundary_points = [-1.0, 1.0]; // left and right boundary points
        for i in 0..2 {
            // loop over boundary points
            for j in 0..dofs {
                // loop over the basis functions
                let mut product = 1.0;
                for m in 0..dofs {
                    // loop over the gauss points
                    if m != j {
                        // skip the current gauss point
                        product *= (boundary_points[i] - cell_gauss_points[m])
                            / (cell_gauss_points[j] - cell_gauss_points[m]);
                    }
                }
                phis_bnd_gps[(i, j)] = product;
            }
        }
        // compute the derivatives of the basis functions at the gauss points.
        for j in 0..dofs {
            // loop over the gauss points
            for i in 0..dofs {
                // loop over the basis functions
                let mut derivative = 0.0;
                for k in 0..dofs {
                    // loop over the gauss points
                    if k != j {
                        // skip the current gauss point
                        let mut product = 1.0;
                        for m in 0..dofs {
                            // loop over the gauss points
                            if m != j && m != k {
                                // skip the current gauss point and the gauss point we are computing the derivative at
                                product *= (cell_gauss_points[i] - cell_gauss_points[m])
                                    / (cell_gauss_points[j] - cell_gauss_points[m]);
                            }
                        }
                        derivative += product / (cell_gauss_points[j] - cell_gauss_points[k]);
                    }
                }
                dphis_cell_gps[(i, j)] = derivative;
            }
        }
        // compute the derivatives of the basis functions at the boundary gauss points.
        for i in 0..2 {
            // loop over boundary points
            for j in 0..dofs {
                // loop over the basis functions
                let mut derivative = 0.0;
                for k in 0..dofs {
                    // loop over the gauss points
                    if k != j {
                        // skip the current gauss point
                        let mut product = 1.0;
                        for m in 0..dofs {
                            // loop over the gauss points
                            if m != j && m != k {
                                // skip the current gauss point and the gauss point we are computing the derivative at
                                product *= (boundary_points[i] - cell_gauss_points[m])
                                    / (cell_gauss_points[j] - cell_gauss_points[m]);
                            }
                        }
                        derivative += product / (cell_gauss_points[j] - cell_gauss_points[k]);
                    }
                }
                dphis_bnd_gps[(i, j)] = derivative;
            }
        }
        LagrangeBasis1D {
            cell_gauss_points,
            cell_gauss_weights,
            phis_cell_gps,
            dphis_cell_gps,
            phis_bnd_gps,
            dphis_bnd_gps,
        }
    }
}
