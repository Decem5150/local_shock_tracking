use csv::Writer;
use ndarray::ArrayView3;
use serde::Serialize;

use crate::disc::{basis::lagrange1d::LagrangeBasis1DLobatto, mesh::mesh1d::Mesh1d};

#[derive(Serialize)]
struct PointData {
    x: f64,
    solution: f64,
    analytical_solution: f64,
}

pub fn write_to_csv(
    solutions: ArrayView3<f64>,
    mesh: &Mesh1d,
    basis: &LagrangeBasis1DLobatto,
    current_time: f64,
    filename: &str,
) -> Result<f64, csv::Error> {
    let mut writer = Writer::from_path(filename)?;
    let mut l2_error_sq = 0.0;

    for ielem in 0..mesh.elem_num {
        let elem = &mesh.elements[ielem];
        let x_left = mesh.nodes[elem.inodes[0]].x;
        let jacob_det = elem.jacob_det;

        for igp in 0..basis.cell_gauss_points.len() {
            let xi = basis.cell_gauss_points[igp];
            let x = x_left + xi * jacob_det;
            let analytical = analytical_burgers(x, current_time);
            let numerical = solutions[[ielem, igp, 0]];

            // Accumulate L² error
            let diff = numerical - analytical;
            l2_error_sq += diff.powi(2) * basis.cell_gauss_weights[igp] * jacob_det;

            // Write data
            let data = PointData {
                x,
                solution: numerical,
                analytical_solution: analytical,
            };
            writer.serialize(data)?;
        }
    }

    writer.flush()?;
    Ok(l2_error_sq.sqrt()) // Return computed L² norm
}

fn analytical_burgers(x: f64, t: f64) -> f64 {
    if t == 0.0 {
        return -(x * std::f64::consts::PI).sin();
    }

    let mut u = -(x * std::f64::consts::PI).sin();
    let tol = 1e-12;
    let max_iter = 200;

    for _ in 0..max_iter {
        let residual = u + (std::f64::consts::PI * (x - u * t)).sin();
        let derivative =
            1.0 + std::f64::consts::PI * t * (std::f64::consts::PI * (x - u * t)).cos();
        let delta = residual / derivative;
        u -= delta;

        if delta.abs() < tol {
            break;
        }
    }
    u
}
