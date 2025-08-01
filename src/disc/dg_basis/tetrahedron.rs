use std::f64::consts::PI;

use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, array, s};
use ndarray_linalg::{Inverse, Norm, Solve};

use crate::disc::{
    dg_basis::{Basis1D, Basis3D, lagrange1d::LobattoBasis, triangle::TriangleBasis},
    gauss_points::lobatto_points::get_lobatto_points_interval,
};

pub struct TetrahedronBasis {
    pub xi: Array1<f64>,
    pub eta: Array1<f64>,
    pub zeta: Array1<f64>,
    pub vandermonde: Array2<f64>,
    pub inv_vandermonde: Array2<f64>,
    pub dxi: Array2<f64>,
    pub deta: Array2<f64>,
    pub dzeta: Array2<f64>,
    pub face_nodes: Array2<usize>,
    pub cub_xi: Array1<f64>,
    pub cub_eta: Array1<f64>,
    pub cub_zeta: Array1<f64>,
    pub cub_w: Array1<f64>,
    pub dxi_cub: Array2<f64>,
    pub deta_cub: Array2<f64>,
    pub dzeta_cub: Array2<f64>,
    pub basis2d: TriangleBasis,
}
impl TetrahedronBasis {
    fn simplex3d_polynomial(
        a: &Array1<f64>,
        b: &Array1<f64>,
        c: &Array1<f64>,
        i: i32,
        j: i32,
        k: i32,
    ) -> Array1<f64> {
        let h1 = Self::jacobi_polynomial(a.view(), 0.0, 0.0, i);
        let h2 = Self::jacobi_polynomial(b.view(), 2.0 * i as f64 + 1.0, 0.0, j);
        let h3 = Self::jacobi_polynomial(c.view(), 2.0 * (i + j) as f64 + 2.0, 0.0, k);
        let ones = Array1::<f64>::ones(a.len());
        let p = 2.0
            * 2.0_f64.sqrt()
            * &h1
            * &h2
            * (&ones - b).powf(i as f64)
            * &h3
            * (&ones - c).powf((i + j) as f64);
        p
    }
    fn rst_to_abc(
        r: &Array1<f64>,
        s: &Array1<f64>,
        t: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let np = r.len();
        let mut a = Array1::<f64>::zeros(np);
        let mut b = Array1::<f64>::zeros(np);
        let mut c = Array1::<f64>::zeros(np);
        for i in 0..np {
            if s[i] + t[i] != 0.0 {
                a[i] = 2.0 * (1.0 + r[i]) / (-s[i] - t[i]) - 1.0;
            } else {
                a[i] = 1.0;
            }
            if t[i] != 1.0 {
                b[i] = 2.0 * (1.0 + s[i]) / (1.0 - t[i]) - 1.0;
            } else {
                b[i] = -1.0;
            }
        }
        c.assign(t);
        (a, b, c)
    }
    fn eval_warp(n: usize, xnodes: &Array1<f64>, xout: &Array1<f64>) -> Array1<f64> {
        let mut warp = Array1::<f64>::zeros(xout.len());

        let mut xeq = Array1::<f64>::zeros(n + 1);
        for i in 0..=n {
            xeq[i] = -1.0 + 2.0 * (n + 1 - i) as f64 / n as f64;
        }

        for i in 0..=n {
            let d = xnodes[i] - xeq[i];
            let mut d_arr = Array1::<f64>::from_elem(xout.len(), d);

            for j in 1..n {
                if i != j {
                    d_arr = &d_arr * (xout - xeq[j]) / (xeq[i] - xeq[j]);
                }
            }

            if i != 0 {
                d_arr = -&d_arr / (xeq[i] - xeq[0]);
            }

            if i != n {
                d_arr = &d_arr / (xeq[i] - xeq[n]);
            }

            warp = warp + d_arr;
        }

        warp
    }
    fn eval_shift(
        n: usize,
        pval: f64,
        l1: &Array1<f64>,
        l2: &Array1<f64>,
        l3: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        // 1) compute Gauss-Lobatto-Legendre node distribution
        let gauss_x = -Self::jacobi_gauss_lobatto(0.0, 0.0, n);

        // 2) compute blending function at each node for each edge
        let blend1 = l2 * l3;
        let blend2 = l1 * l3;
        let blend3 = l1 * l2;

        // 3) amount of warp for each node, for each edge
        let warpfactor1 = 4.0 * Self::eval_warp(n, &gauss_x, &(l3 - l2));
        let warpfactor2 = 4.0 * Self::eval_warp(n, &gauss_x, &(l1 - l3));
        let warpfactor3 = 4.0 * Self::eval_warp(n, &gauss_x, &(l2 - l1));

        // 4) combine blend & warp
        let warp1 = &blend1 * &warpfactor1 * (1.0 + (pval * l1).mapv(|x| x.powi(2)));
        let warp2 = &blend2 * &warpfactor2 * (1.0 + (pval * l2).mapv(|x| x.powi(2)));
        let warp3 = &blend3 * &warpfactor3 * (1.0 + (pval * l3).mapv(|x| x.powi(2)));

        // 5) evaluate shift in equilateral triangle
        let dx = &warp1 + (2.0 * PI / 3.0).cos() * &warp2 + (4.0 * PI / 3.0).cos() * &warp3;
        let dy = 0.0 * &warp1 + (2.0 * PI / 3.0).sin() * &warp2 + (4.0 * PI / 3.0).sin() * &warp3;

        (dx, dy)
    }
    fn warp_shift_face3d(
        n: usize,
        pval1: f64,
        _l1: &Array1<f64>,
        l2: &Array1<f64>,
        l3: &Array1<f64>,
        l4: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let (warp1, warp2) = Self::eval_shift(n, pval1, l2, l3, l4);
        (warp1, warp2)
    }
    fn equid_nodes3d(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        // Total number of nodes
        let np = (n + 1) * (n + 2) * (n + 3) / 6;

        // Create equidistributed nodes on reference tetrahedron
        let mut x = Array1::<f64>::zeros(np);
        let mut y = Array1::<f64>::zeros(np);
        let mut z = Array1::<f64>::zeros(np);

        let mut sk = 0;
        for i in 1..=n + 1 {
            for m in 1..=n + 2 - i {
                for q in 1..=n + 3 - i - m {
                    x[sk] = -1.0 + (q - 1) as f64 * 2.0 / n as f64;
                    y[sk] = -1.0 + (m - 1) as f64 * 2.0 / n as f64;
                    z[sk] = -1.0 + (i - 1) as f64 * 2.0 / n as f64;
                    sk += 1;
                }
            }
        }

        (x, y, z)
    }
    fn xyz_to_rst(
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let v1 = array![-1.0, -1.0 / 3.0_f64.sqrt(), -1.0 / 6.0_f64.sqrt()];
        let v2 = array![1.0, -1.0 / 3.0_f64.sqrt(), -1.0 / 6.0_f64.sqrt()];
        let v3 = array![0.0, 2.0 / 3.0_f64.sqrt(), -1.0 / 6.0_f64.sqrt()];
        let v4 = array![0.0, 0.0, 3.0 / 6.0_f64.sqrt()];

        let ones = Array1::<f64>::ones(x.len());

        // Compute RHS: [X';Y';Z'] - 0.5*(v2'+v3'+v4'-v1')*ones(1,length(X))
        let offset = 0.5 * (&v2 + &v3 + &v4 - &v1);
        let rhs_x = x - offset[0];
        let rhs_y = y - offset[1];
        let rhs_z = z - offset[2];

        // Compute matrix A = [0.5*(v2-v1)',0.5*(v3-v1)',0.5*(v4-v1)']
        let col1 = 0.5 * (&v2 - &v1);
        let col2 = 0.5 * (&v3 - &v1);
        let col3 = 0.5 * (&v4 - &v1);

        let mut a = Array2::<f64>::zeros((3, 3));
        a.column_mut(0).assign(&col1);
        a.column_mut(1).assign(&col2);
        a.column_mut(2).assign(&col3);

        // Solve A * RST = RHS for each point
        let n_points = x.len();
        let mut r = Array1::<f64>::zeros(n_points);
        let mut s = Array1::<f64>::zeros(n_points);
        let mut t = Array1::<f64>::zeros(n_points);

        for i in 0..n_points {
            let rhs = array![rhs_x[i], rhs_y[i], rhs_z[i]];
            let rst = a.solve(&rhs).unwrap();
            // Transform from symmetric tetrahedron [-1,1] to unit tetrahedron [0,1]
            r[i] = (rst[0] + 1.0) / 2.0;
            s[i] = (rst[1] + 1.0) / 2.0;
            t[i] = (rst[2] + 1.0) / 2.0;
        }

        (r, s, t)
    }
}
impl Basis1D for TetrahedronBasis {}
impl Basis3D for TetrahedronBasis {
    fn nodes3d(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let alpha_store = [
            0.0, 0.0, 0.0, 0.1002, 0.1332, 1.5608, 1.3413, 1.2577, 1.1603, 1.10153, 0.608, 0.4523,
            0.8865, 0.8717, 0.9655,
        ];
        let alpha = if n < 16 { alpha_store[n] } else { 1.0 };
        let np = (n + 1) * (n + 2) * (n + 3) / 6;
        let tol = 1e-10;

        let (r, s, t) = Self::equid_nodes3d(n);

        let ones = Array1::<f64>::ones(np);

        let l1 = (&ones + &t) / 2.0;
        let l2 = (&ones + &s) / 2.0;
        let l3 = -(&ones + &r + &s + &t) / 2.0;
        let l4 = (&ones + &r) / 2.0;

        // set vertices of tetrahedron
        let v1 = array![-1.0, -1.0 / 3.0_f64.sqrt(), -1.0 / 6.0_f64.sqrt()];
        let v2 = array![1.0, -1.0 / 3.0_f64.sqrt(), -1.0 / 6.0_f64.sqrt()];
        let v3 = array![0.0, 2.0 / 3.0_f64.sqrt(), -1.0 / 6.0_f64.sqrt()];
        let v4 = array![0.0, 0.0, 3.0 / 6.0_f64.sqrt()];

        // orthogonal axis tangents on faces 1-4
        let mut t1 = Array2::<f64>::zeros((4, 3));
        let mut t2 = Array2::<f64>::zeros((4, 3));

        t1.row_mut(0).assign(&(&v2 - &v1));
        t1.row_mut(1).assign(&(&v2 - &v1));
        t1.row_mut(2).assign(&(&v3 - &v2));
        t1.row_mut(3).assign(&(&v3 - &v1));
        t2.row_mut(0).assign(&(&v3 - 0.5 * (&v1 + &v2)));
        t2.row_mut(1).assign(&(&v4 - 0.5 * (&v1 + &v2)));
        t2.row_mut(2).assign(&(&v4 - 0.5 * (&v2 + &v3)));
        t2.row_mut(3).assign(&(&v4 - 0.5 * (&v1 + &v3)));

        for n in 0..4 {
            let norm = t1.row(n).norm_l2();
            t1.row_mut(n).mapv_inplace(|x| x / norm);
            let norm = t2.row(n).norm_l2();
            t2.row_mut(n).mapv_inplace(|x| x / norm);
        }

        // Warp and blend for each face (accumulated in shift)
        // XYZ = L3*v1+L4*v2+L2*v3+L1*v4; % form undeformed coordinates
        let mut xyz = Array2::<f64>::zeros((np, 3));
        for i in 0..np {
            for j in 0..3 {
                xyz[[i, j]] = l3[i] * v1[j] + l4[i] * v2[j] + l2[i] * v3[j] + l1[i] * v4[j];
            }
        }

        let mut shift = Array2::<f64>::zeros((np, 3));

        // TODO: Implement face warping loop here
        for iface in 0..4 {
            let (la, lb, lc, ld) = match iface {
                0 => (l1.clone(), l2.clone(), l3.clone(), l4.clone()),
                1 => (l2.clone(), l1.clone(), l3.clone(), l4.clone()),
                2 => (l3.clone(), l1.clone(), l4.clone(), l2.clone()),
                3 => (l4.clone(), l1.clone(), l3.clone(), l2.clone()),
                _ => unreachable!(),
            };

            let (warp1, warp2) = Self::warp_shift_face3d(n, alpha, &la, &lb, &lc, &ld);
            let mut blend = &lb * &lc * &ld;
            let denom = (&lb + 0.5 * &la) * (&lc + 0.5 * &la) * (&ld + 0.5 * &la);
            let ids: Vec<usize> = denom
                .indexed_iter()
                .filter(|(_, x)| **x > tol)
                .map(|(i, _)| i)
                .collect();
            for id in ids {
                blend[id] = (1.0 + (alpha * la[id]).powf(2.0)) * blend[id] / denom[id];
            }

            // compute warp & blend
            shift = shift + (&blend * &warp1) * t1.row(iface) + (&blend * &warp2) * t2.row(iface);

            // fix face warp
            // ids = find(La<tol & ( (Lb>tol) + (Lc>tol) + (Ld>tol) < 3));
            let ids: Vec<usize> = (0..np)
                .filter(|&i| {
                    la[i] < tol
                        && ((if lb[i] > tol { 1 } else { 0 })
                            + (if lc[i] > tol { 1 } else { 0 })
                            + (if ld[i] > tol { 1 } else { 0 }))
                            < 3
                })
                .collect();

            for &id in &ids {
                shift
                    .row_mut(id)
                    .assign(&(warp1[id] * &t1.row(iface) + warp2[id] * &t2.row(iface)));
            }
        }
        xyz = xyz + shift;
        // Extract X, Y, Z coordinates
        let x = xyz.column(0).to_owned();
        let y = xyz.column(1).to_owned();
        let z = xyz.column(2).to_owned();

        (x, y, z)
    }
    fn vandermonde3d(
        n: usize,
        r: ArrayView1<f64>,
        s: ArrayView1<f64>,
        t: ArrayView1<f64>,
    ) -> Array2<f64> {
        let np = (n + 1) * (n + 2) * (n + 3) / 6;
        let mut v = Array2::<f64>::zeros((r.len(), np));

        // Transform from unit tetrahedron [0,1] back to standard tetrahedron [-1,1]
        let r_std = r.mapv(|x| 2.0 * x - 1.0);
        let s_std = s.mapv(|x| 2.0 * x - 1.0);
        let t_std = t.mapv(|x| 2.0 * x - 1.0);

        // Convert to (a,b,c) coordinates for polynomial evaluation
        let (a, b, c) = Self::rst_to_abc(&r_std, &s_std, &t_std);

        // Build Vandermonde matrix
        let mut sk = 0;
        for i in 0..=n {
            for j in 0..=(n - i) {
                for k in 0..=(n - i - j) {
                    let p = Self::simplex3d_polynomial(&a, &b, &c, i as i32, j as i32, k as i32);
                    v.column_mut(sk).assign(&p);
                    sk += 1;
                }
            }
        }

        v
    }
}
/*
function warp = evalwarp(p, xnodes, xout)

% function warp = evalwarp(p, xnodes, xout)
% Purpose: compute one-dimensional edge warping function

warp = zeros(size(xout));

for i=1:p+1
  xeq(i) = -1 + 2*(p+1-i)/p;
end

for i=1:p+1
  d = (xnodes(i)-xeq(i));
  for j=2:p
    if(i~=j)
    d = d.*(xout-xeq(j))/(xeq(i)-xeq(j));
    end
  end

  if(i~=1)
    d = -d/(xeq(i)-xeq(1));
  end

  if(i~=(p+1))
    d = d/(xeq(i)-xeq(p+1));
  end

  warp = warp+d;
end
return;
*/

/*
function [dx, dy] = evalshift(p, pval, L1, L2, L3)

% function [dx, dy] = evalshift(p, pval, L1, L2, L3)
% Purpose: compute two-dimensional Warp & Blend transform

% 1) compute Gauss-Lobatto-Legendre node distribution
gaussX = -JacobiGL(0,0,p);

% 2) compute blending function at each node for each edge
blend1 = L2.*L3; blend2 = L1.*L3; blend3 = L1.*L2;

% 3) amount of warp for each node, for each edge
warpfactor1 = 4*evalwarp(p, gaussX, L3-L2);
warpfactor2 = 4*evalwarp(p, gaussX, L1-L3);
warpfactor3 = 4*evalwarp(p, gaussX, L2-L1);

% 4) combine blend & warp
warp1 = blend1.*warpfactor1.*(1 + (pval*L1).^2);
warp2 = blend2.*warpfactor2.*(1 + (pval*L2).^2);
warp3 = blend3.*warpfactor3.*(1 + (pval*L3).^2);

% 5) evaluate shift in equilateral triangle
dx = 1*warp1 + cos(2*pi/3)*warp2 + cos(4*pi/3)*warp3;
dy = 0*warp1 + sin(2*pi/3)*warp2 + sin(4*pi/3)*warp3;
return;
 */

/*
function [X,Y,Z] = EquinNdes3D(N)

% function [X,Y,Z] = EquinNdes3D(N)
% Purpose: compute the equidistributed nodes on the reference tetrahedron

% total number of nodes
Np = (N+1)*(N+2)*(N+3)/6;

% 2) create equidistributed nodes on equilateral triangle
X = zeros(Np,1); Y = zeros(Np,1); Z = zeros(Np,1);

sk = 1;
for n=1:N+1
  for m=1:N+2-n
    for q=1:N+3-n-m
      X(sk) = -1 + (q-1)*2/N; Y(sk) = -1 + (m-1)*2/N; Z(sk) = -1 + (n-1)*2/N;
      sk = sk+1;
    end
  end
end
return;
*/
/*
function [X,Y,Z] = Nodes3D(p)

% function [X,Y,Z] = Nodes3D(p)
% Purpose: compute Warp & Blend nodes
%  input:    p=polynomial order of interpolant
%  output: X,Y,Z vectors of node coordinates in equilateral tetrahedron

% choose optimized blending parameter
alphastore = [0;0;0;0.1002; 1.1332;1.5608;1.3413;1.2577;1.1603;...
                1.10153;0.6080;0.4523;0.8856;0.8717;0.9655];
if(p<=15); alpha = alphastore(p) ; else;  alpha = 1. ; end

% total number of nodes and tolerance
N = (p+1)*(p+2)*(p+3)/6; tol = 1e-10;

[r,s,t] = EquiNodes3D(p); % create equidistributed nodes
L1 = (1+t)/2; L2 = (1+s)/2; L3 = -(1+r+s+t)/2; L4 = (1+r)/2;

% set vertices of tetrahedron
v1 = [-1, -1/sqrt(3), -1/sqrt(6)]; v2 = [ 1, -1/sqrt(3),-1/sqrt(6)];
v3 = [ 0,  2/sqrt(3), -1/sqrt(6)]; v4 = [ 0,  0,         3/sqrt(6)];

% orthogonal axis tangents on faces 1-4
t1(1,:) = v2-v1;          t1(2,:) = v2-v1;
t1(3,:) = v3-v2;          t1(4,:) = v3-v1;
t2(1,:) = v3-0.5*(v1+v2); t2(2,:) = v4-0.5*(v1+v2);
t2(3,:) = v4-0.5*(v2+v3); t2(4,:) = v4-0.5*(v1+v3);

for n=1:4 % normalize tangents
   t1(n,:) = t1(n,:)/norm(t1(n,:)); t2(n,:) = t2(n,:)/norm(t2(n,:));
end

% Warp and blend for each face (accumulated in shiftXYZ)
XYZ = L3*v1+L4*v2+L2*v3+L1*v4; % form undeformed coordinates
shift = zeros(size(XYZ));
for face=1:4
  if(face==1); La = L1; Lb = L2; Lc = L3; Ld = L4; end;
  if(face==2); La = L2; Lb = L1; Lc = L3; Ld = L4; end;
  if(face==3); La = L3; Lb = L1; Lc = L4; Ld = L2; end;
  if(face==4); La = L4; Lb = L1; Lc = L3; Ld = L2; end;

  % compute warp tangential to face
  [warp1 warp2] = WarpShiftFace3D(p, alpha, alpha, La, Lb, Lc, Ld);

  blend = Lb.*Lc.*Ld;   % compute volume blending

  denom = (Lb+.5*La).*(Lc+.5*La).*(Ld+.5*La);   % modify linear blend
  ids = find(denom>tol);
  blend(ids) = (1+(alpha.*La(ids)).^2).*blend(ids)./denom(ids);

  % compute warp & blend
  shift = shift+(blend.*warp1)*t1(face,:) + (blend.*warp2)*t2(face,:);

  % fix face warp
  ids = find(La<tol & ( (Lb>tol) + (Lc>tol) + (Ld>tol) < 3));
  shift(ids,:) = warp1(ids)*t1(face,:) + warp2(ids)*t2(face,:);
end
XYZ = XYZ + shift;
X = XYZ(:,1); Y = XYZ(:,2); Z = XYZ(:,3);
return;
*/
/*
function [r, s, t] = xyztorst(X, Y, Z)

% function [r,s,t] = xyztorst(x, y, z)
% Purpose : Transfer from (x,y,z) in equilateral tetrahedron
%           to (r,s,t) coordinates in standard tetrahedron

v1 = [-1,-1/sqrt(3), -1/sqrt(6)]; v2 = [ 1,-1/sqrt(3), -1/sqrt(6)];
v3 = [ 0, 2/sqrt(3), -1/sqrt(6)]; v4 = [ 0, 0/sqrt(3),  3/sqrt(6)];

% back out right tet nodes
rhs = [X';Y';Z'] - 0.5*(v2'+v3'+v4'-v1')*ones(1,length(X));
A = [0.5*(v2-v1)',0.5*(v3-v1)',0.5*(v4-v1)'];
RST = A\[rhs];
r = RST(1,:)'; s = RST(2,:)'; t = RST(3,:)';
return;
*/
