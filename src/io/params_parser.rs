use serde::Deserialize;
use serde_json;
use std::fs;
#[derive(Deserialize, Debug)]
pub struct SolverParamParser {
    pub cfl: f64,
    pub final_time: f64,
    pub final_step: usize,
    pub polynomial_order: usize,
    pub shock_tracking_order: usize,
}
impl SolverParamParser {
    pub fn parse(file_path: &str) -> Self {
        let file_content = fs::read_to_string(file_path).expect("Failed to read file");
        let param: SolverParamParser =
            serde_json::from_str(&file_content).expect("Failed to parse JSON");
        param
    }
}
