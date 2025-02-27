use csv::Writer;
use ndarray::ArrayView3;
use serde::Serialize;

use crate::disc::{basis::lagrange1d::LagrangeBasis1DLobatto, mesh::mesh1d::Mesh1d};

#[derive(Serialize)]
struct PointData {
    x: f64,
    solution: f64,
}

pub fn write_to_csv(
    solutions: ArrayView3<f64>,
    mesh: &Mesh1d,
    basis: &LagrangeBasis1DLobatto,
    filename: &str,
) -> Result<(), csv::Error> {
    let mut writer = Writer::from_path(filename)?;
    // writer.write_record(["x", "solution"])?;
    for ielem in 0..mesh.elem_num {
        let elem = &mesh.elements[ielem];
        let x_left = mesh.nodes[elem.inodes[0]].x;
        for igp in 0..basis.cell_gauss_points.len() {
            let xi = basis.cell_gauss_points[igp];
            let x = x_left + xi * elem.jacob_det;
            let data = PointData {
                x,
                solution: solutions[[ielem, igp, 0]],
            };
            writer.serialize(data)?;
        }
    }
    writer.flush()?;
    Ok(())
}
