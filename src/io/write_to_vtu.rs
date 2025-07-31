use ndarray::Array2;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use vtkio::{
    Vtk,
    model::{
        Attribute, Attributes, ByteOrder, CellType, Cells, DataArray, DataSet, ElementType,
        IOBuffer, UnstructuredGridPiece, Version, VertexNumbers,
    },
};

use once_cell::sync::Lazy;

use crate::disc::{
    basis::triangle::TriangleBasis,
    mesh::mesh2d::{Mesh2d, Status, TriangleElement},
};

static OUTPUT_DIR: Lazy<String> = Lazy::new(|| {
    // Create outputs directory if it doesn't exist
    let outputs_dir = "outputs";
    if !Path::new(outputs_dir).exists() {
        fs::create_dir(outputs_dir).expect("Failed to create outputs directory");
    }

    // Get current timestamp
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");

    // Convert to seconds since epoch
    let timestamp = now.as_secs();

    // Create a more readable timestamp format: MMDD_HHMM_SS
    // This is a simple approach without external dependencies
    let secs_in_day = 24 * 60 * 60;
    let secs_in_hour = 60 * 60;
    let secs_in_minute = 60;

    // Approximate day of year (not perfect but good enough for unique timestamps)
    let days_since_epoch = timestamp / secs_in_day;
    let day_of_year = (days_since_epoch % 365) + 1;
    let month = ((day_of_year - 1) / 30) + 1; // Rough approximation
    let day = ((day_of_year - 1) % 30) + 1;

    let seconds_today = timestamp % secs_in_day;
    let hour = seconds_today / secs_in_hour;
    let minute = (seconds_today % secs_in_hour) / secs_in_minute;
    let second = seconds_today % secs_in_minute;

    let timestamp_str = format!(
        "{:02}{:02}_{:02}{:02}_{:02}",
        month, day, hour, minute, second
    );

    // Create timestamped directory
    let timestamped_dir = format!("{}/{}", outputs_dir, timestamp_str);
    if !Path::new(&timestamped_dir).exists() {
        fs::create_dir_all(&timestamped_dir).expect("Failed to create timestamped directory");
    }

    timestamped_dir
});

pub fn write_average(
    name: &str,
    solutions: &Array2<f64>,
    mesh: &Mesh2d<TriangleElement>,
    basis: &TriangleBasis,
    current_step: usize,
) {
    let mut vtk_points = Vec::new();
    let mut node_map = vec![None; mesh.phys_nodes.len()];
    let mut new_node_idx = 0;

    for (i, node_status) in mesh.phys_nodes.iter().enumerate() {
        if let Status::Active(node) = node_status {
            vtk_points.push(node.x);
            vtk_points.push(node.y);
            vtk_points.push(0.0); // Z-coordinate
            node_map[i] = Some(new_node_idx);
            new_node_idx += 1;
        }
    }

    let mut connectivity = Vec::with_capacity(mesh.elements.len() * 3);
    let mut cell_types = Vec::with_capacity(mesh.elements.len());
    let mut cell_averages = Vec::with_capacity(mesh.elements.len());
    let mut original_element_indices = Vec::with_capacity(mesh.elements.len());
    let mut num_cells = 0;

    for (elem_idx, elem_status) in mesh.elements.iter().enumerate() {
        if let Status::Active(element) = elem_status {
            let element_inodes = element.inodes;
            let vtk_conn_elem: Vec<u64> = element_inodes
                .iter()
                .map(|&inode| node_map[inode].unwrap() as u64)
                .collect();

            connectivity.extend(vtk_conn_elem);
            cell_types.push(CellType::Triangle);
            num_cells += 1;
            original_element_indices.push(elem_idx as u64);

            let u_e = solutions.row(elem_idx);
            let modal_coeffs = basis.inv_vandermonde.dot(&u_e);
            let u0_hat = modal_coeffs[0];

            // The first modal basis function is phi_0 = sqrt(2).
            // The average value is u_avg = (1/Area) * integral(u)
            // u = u0_hat * phi_0 + ...
            // integral(u) = u0_hat * integral(phi_0) + ... = u0_hat * sqrt(2) * Area
            // so, u_avg = u0_hat * sqrt(2)
            let avg = u0_hat / 2.0_f64.sqrt();
            cell_averages.push(avg);
        }
    }

    let filename = format!("{}/{}_{}.vtu", &*OUTPUT_DIR, name, current_step);

    let vtk_file = Vtk {
        version: Version::XML { major: 1, minor: 0 },
        title: "Solution Data".into(),
        byte_order: ByteOrder::native(),
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F64(vtk_points),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity,
                    offsets: (0..num_cells).map(|i| ((i + 1) * 3) as u64).collect(),
                },
                types: cell_types,
            },
            data: Attributes {
                point: vec![],
                cell: vec![
                    Attribute::DataArray(DataArray {
                        name: "solution".to_string(),
                        elem: ElementType::Scalars {
                            num_comp: 1,
                            lookup_table: None,
                        },
                        data: IOBuffer::F64(cell_averages),
                    }),
                    Attribute::DataArray(DataArray {
                        name: "original_element_indices".to_string(),
                        elem: ElementType::Scalars {
                            num_comp: 1,
                            lookup_table: None,
                        },
                        data: IOBuffer::U64(original_element_indices),
                    }),
                ],
            },
        }),
        file_path: None,
    };

    vtk_file
        .export(&filename)
        .expect("Failed to write VTU file");
}

pub fn write_nodal_solutions(
    name: &str,
    solutions: &Array2<f64>,
    mesh: &Mesh2d<TriangleElement>,
    basis: &TriangleBasis,
    current_step: usize,
) {
    let mut vtk_points = Vec::new();
    let mut connectivity = Vec::new();
    let mut cell_types = Vec::new();
    let mut point_solutions = Vec::new();

    let num_solution_nodes = basis.xi.len();
    let mut global_point_id = 0_usize;

    for (elem_idx, elem_status) in mesh.elements.iter().enumerate() {
        if let Status::Active(element) = elem_status {
            let element_inodes = element.inodes;

            // Get physical coordinates of the mesh nodes for this element
            let x_nodes: Vec<f64> = element_inodes
                .iter()
                .map(|&inode| mesh.phys_nodes[inode].as_ref().x)
                .collect();
            let y_nodes: Vec<f64> = element_inodes
                .iter()
                .map(|&inode| mesh.phys_nodes[inode].as_ref().y)
                .collect();

            // Map each solution node from reference coordinates to physical coordinates
            for i in 0..num_solution_nodes {
                let r = basis.xi[i];
                let s = basis.eta[i];

                // Triangular shape functions for coordinate transformation
                // Reference triangle vertices: node 0 at (0,0), node 1 at (1,0), node 2 at (0,1)
                let n1 = 1.0 - r - s; // Shape function for node 0 at (0,0)
                let n2 = r; // Shape function for node 1 at (1,0)  
                let n3 = s; // Shape function for node 2 at (0,1)

                // Map to physical coordinates
                let x_phys = n1 * x_nodes[0] + n2 * x_nodes[1] + n3 * x_nodes[2];
                let y_phys = n1 * y_nodes[0] + n2 * y_nodes[1] + n3 * y_nodes[2];

                // Add point coordinates
                vtk_points.push(x_phys);
                vtk_points.push(y_phys);
                vtk_points.push(0.0); // Z-coordinate
            }
            // Store the solution values at each point
            point_solutions.extend(solutions.row(elem_idx).iter().copied());

            // Create connectivity for this element (connecting solution nodes)
            // Solution nodes are arranged row-by-row from bottom edge to top vertex
            // We create triangular sub-cells by connecting nodes from adjacent rows
            if num_solution_nodes >= 3 {
                let base_id = global_point_id;

                // Determine the polynomial order
                let p = basis.n;

                // Create sub-triangles row by row
                let mut node_id = 0;
                for row in 0..p {
                    let nodes_in_current_row = p + 1 - row;
                    let nodes_in_next_row = p - row;

                    if nodes_in_next_row > 0 {
                        // Create triangles pointing up
                        for i in 0..nodes_in_next_row {
                            connectivity.push((base_id + node_id + i) as u64);
                            connectivity.push((base_id + node_id + i + 1) as u64);
                            connectivity
                                .push((base_id + node_id + nodes_in_current_row + i) as u64);
                            cell_types.push(CellType::Triangle);
                        }

                        // Create triangles pointing down (except for the last row)
                        for i in 0..(nodes_in_next_row - 1) {
                            connectivity.push((base_id + node_id + i + 1) as u64);
                            connectivity
                                .push((base_id + node_id + nodes_in_current_row + i + 1) as u64);
                            connectivity
                                .push((base_id + node_id + nodes_in_current_row + i) as u64);
                            cell_types.push(CellType::Triangle);
                        }
                    }

                    node_id += nodes_in_current_row;
                }
            }

            global_point_id += num_solution_nodes;
        }
    }

    let filename = format!("{}/{}_nodal_{}.vtu", &*OUTPUT_DIR, name, current_step);

    let vtk_file = Vtk {
        version: Version::XML { major: 1, minor: 0 },
        title: "Solution Point Data".into(),
        byte_order: ByteOrder::native(),
        data: DataSet::inline(UnstructuredGridPiece {
            points: IOBuffer::F64(vtk_points),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity,
                    offsets: (0..cell_types.len())
                        .map(|i| ((i + 1) * 3) as u64)
                        .collect(),
                },
                types: cell_types,
            },
            data: Attributes {
                point: vec![Attribute::DataArray(DataArray {
                    name: "solution".to_string(),
                    elem: ElementType::Scalars {
                        num_comp: 1,
                        lookup_table: None,
                    },
                    data: IOBuffer::F64(point_solutions),
                })],
                cell: vec![],
            },
        }),
        file_path: None,
    };

    vtk_file
        .export(&filename)
        .expect("Failed to write VTU point file");
}
