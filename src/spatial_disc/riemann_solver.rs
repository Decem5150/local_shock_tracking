use ndarray::{Array, Ix1};

pub fn hllc1d(left_value: &Array<f64, Ix1>, right_value: &Array<f64, Ix1>, hcr: &f64) -> Result<[f64; 4], &'static str> {
    let mut flux = [0.0_f64; 3];
    let ql = [
        left_value[0], 
        left_value[1],
        left_value[2]
        ];
    let qr = [
        right_value[0],
        right_value[1],
        right_value[2]
        ];
    let ul = ql[1] / ql[0];
    let ur = qr[1] / qr[0];
    let pl = (hcr - 1.0) * (ql[2] - 0.5 * (ql[1] * ql[1]) / ql[0]);
    let pr = (hcr - 1.0) * (qr[2] - 0.5 * (qr[1] * qr[1]) / qr[0]);
    if ql[0] < 0.0 || qr[0] < 0.0 {
        Err("Negative density found in HLLC!")
    }
    else if pl < 0.0 || pr < 0.0 {
        Err("Negative pressure found in HLLC!")
    }
    else {
        let cl = (hcr * pl / ql[0]).sqrt();
        let cr = (hcr * pr / qr[0]).sqrt();
        let mut fl = [0.0f64; 3];
        let mut fr = [0.0f64; 3];
        fl[0] = ql[1];
        fl[1] = ql[1] * ul + pl;
        fl[2] = ul * (ql[2] + pl);
        fr[0] = qr[1];
        fr[1] = qr[1] * ur + pr;
        fr[2] = ur * (qr[2] + pr);
        let p_star = {
            let zeta = (hcr - 1.0) / (2.0 * hcr);
            ((cl + cr - (zeta - 1.0) / 2.0 * (ur - ul)) / (cl / pl.powf(zeta) + cr / pr.powf(zeta))).powf(1.0 / zeta)
        };
        let sl = {
            let ql_lower = {
                if p_star <= pl {
                    1.0
                }
                else {
                    (1.0 + (hcr + 1.0) / (2.0 * hcr) * (p_star / pl - 1.0)).sqrt()
                }
            };
            ul - cl * ql_lower
        };
        let sr = {
            let qr_lower = {
                if p_star <= pr {
                    1.0
                }
                else {
                    (1.0 + (hcr + 1.0) / (2.0 * hcr) * (p_star / pr - 1.0)).sqrt()
                }
            };
            ur + cr * qr_lower
        };
        if sl >= 0.0 {
            flux[0] = fl[0];
            flux[1] = fl[1] ;
            flux[2] = fl[2];
            Ok(flux)
        } else if sr <= 0.0 {
            flux[0] = fr[0];
            flux[1] = fr[1];
            flux[2] = fr[2];
            Ok(flux)
        } else {
            let s_star = (pr - pl + ql[0] * ul * (sl - ul) - qr[0] * ur * (sr - ur)) / (ql[0] * (sl - ul) - qr[0] * (sr - ur));
            if s_star >= 0.0 {
                let mut q_star = [0.0_f64; 4];
                q_star[0] = ql[0] * (sl - ul) / (sl - s_star);
                q_star[1] = ql[0] * (sl - ul) / (sl - s_star) * s_star;
                q_star[2] = ql[0] * (sl - ul) / (sl - s_star) * (ql[2] / ql[0] + (s_star - ul) * (s_star + pl / (ql[0] * (sl - ul))));
    
                flux[0] = fl[0] + sl * (q_star[0] - ql[0]);
                flux[1] = fl[1] + sl * (q_star[1] - ql[1]);
                flux[2] = fl[2] + sl * (q_star[2] - ql[2]);
                Ok(flux)
            }
            else {
                let mut q_star = [0.0_f64; 4];
                q_star[0] = qr[0] * (sr - ur) / (sr - s_star);
                q_star[1] = qr[0] * (sr - ur) / (sr - s_star) * s_star;
                q_star[2] = qr[0] * (sr - ur) / (sr - s_star) * (qr[2] / qr[0] + (s_star - ur) * (s_star + pr / (qr[0] * (sr - ur))));
    
                flux[0] = fr[0] + sr * (q_star[0] - qr[0]);
                flux[1] = fr[1] + sr * (q_star[1] - qr[1]);
                flux[2] = fr[2] + sr * (q_star[2] - qr[2]);
                Ok(flux)
            }
        }
    }
}