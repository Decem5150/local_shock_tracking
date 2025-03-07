use ndarray::{ArrayView2, ArrayView3};

use super::Disc1dBurgers;

impl Disc1dBurgers<'_> {
    pub fn detect_shock(&self, old_sol: ArrayView3<f64>, candidate_sol: ArrayView2<f64>) -> bool {
        !self.numerical_admissibility_detection(old_sol, candidate_sol)
    }
    fn numerical_admissibility_detection(
        &self,
        old_sol: ArrayView3<f64>,
        candidate_sol: ArrayView2<f64>,
    ) -> bool {
        let ndof = self.solver_param.cell_gp_num;
        let delta0: f64 = 1e-4;
        let epsilon: f64 = 1e-4;

        let mut max_sol = old_sol[[0, 0, 0]];
        let mut min_sol = old_sol[[0, 0, 0]];
        for ielem in 0..old_sol.shape()[0] {
            for idof in 0..ndof {
                max_sol = max_sol.max(old_sol[[ielem, idof, 0]]);
                min_sol = min_sol.min(old_sol[[ielem, idof, 0]]);
            }
        }
        //dbg!(&max_sol);
        //dbg!(&min_sol);
        let delta = delta0.max(epsilon * (max_sol - min_sol));
        let mut is_admissible = true;
        for idof in 0..ndof {
            //dbg!(&candidate_sol[[idof, 0]]);
            //dbg!(&delta);
            if (candidate_sol[[idof, 0]] + delta < min_sol)
                || (candidate_sol[[idof, 0]] - delta > max_sol)
            {
                let tmp = epsilon * (max_sol - min_sol);
                dbg!(&candidate_sol);
                dbg!(&tmp);
                dbg!(&delta);
                dbg!(&min_sol);
                dbg!(&max_sol);
                is_admissible = false;
                break;
            }
        }
        is_admissible
    }
}
