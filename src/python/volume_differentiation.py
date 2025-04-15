import sympy as sp

i, j, m, n = sp.symbols('i j m n', integer=True, positive=True)
N = sp.Function('N')
xi, eta = sp.symbols('xi eta')
alpha = sp.symbols('alpha')
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
y1, y2, y3, y4 = sp.symbols('y1 y2 y3 y4')
#x = sp.Function('x')(xi, eta)
#y = sp.Function('y')(xi, eta)
x_coords = sp.IndexedBase('x')  # Creates x[i]
y_coords = sp.IndexedBase('y')  # Creates y[i]
# Define the bilinear shape functions for a quadrilateral element
'''
N1 = 0.25 * (1 - xi) * (1 - eta)
N2 = 0.25 * (1 + xi) * (1 - eta)
N3 = 0.25 * (1 + xi) * (1 + eta)
N4 = 0.25 * (1 - xi) * (1 + eta)
print("N1.diff(xi):")
sp.pprint(N1.diff(xi))
print("N1.diff(eta):")
sp.pprint(N1.diff(eta))
'''

N1 = sp.Function('N1')(xi, eta)
N2 = sp.Function('N2')(xi, eta)
N3 = sp.Function('N3')(xi, eta)
N4 = sp.Function('N4')(xi, eta)

# Define the mapping from reference (xi, eta) to physical coordinates (x, y)
x_expr = N1 * x1 + N2 * x2 + N3 * x3 + N4 * x4
y_expr = N1 * y1 + N2 * y2 + N3 * y3 + N4 * y4
#x_expr = sp.Sum(N(i, xi, eta) * x_coords[i], (i, 1, m))
#y_expr = sp.Sum(N(i, xi, eta) * y_coords[i], (i, 1, m))
# Compute the partial derivatives of the mapping
'''
dx_dxi   = sp.diff(x_expr, xi)
dx_deta  = sp.diff(x_expr, eta)
dy_dxi   = sp.diff(y_expr, xi)
dy_deta  = sp.diff(y_expr, eta)

# Compute the Jacobian determinant as a function of (xi, eta)
jacob_det = sp.simplify(dx_dxi * dy_deta - dx_deta * dy_dxi)
# Compute the inverse transpose Jacobian components
jacob_inv_t_00 = sp.simplify( dy_deta / jacob_det )
jacob_inv_t_01 = sp.simplify(-dx_deta / jacob_det )
jacob_inv_t_10 = sp.simplify(-dy_dxi / jacob_det )
jacob_inv_t_11 = sp.simplify( dx_dxi / jacob_det )
#print("\nJacobian determinant:")
#sp.pprint(jacob_det)

#print("\nInverse Transpose Jacobian (as a 2x2 matrix):")
jacob_inv_t = sp.Matrix([[jacob_inv_t_00, jacob_inv_t_01],
                         [jacob_inv_t_10, jacob_inv_t_11]])
#sp.pprint(jacob_inv_t)
'''
# Form the mapping vector
F = sp.Matrix([x_expr, y_expr])
# Compute the Jacobian matrix with respect to (xi, eta)
J = F.jacobian([xi, eta])
# Simplify the Jacobian matrix entries
J = sp.simplify(J)
# Compute the Jacobian determinant
jacob_det = sp.simplify(J.det())
#sp.pprint(jacob_det)

# Compute the inverse transpose of the Jacobian
jacob_inv_t = J.inv().T
#sp.pprint(jacob_inv_t)
#latex_output_jacob_inv_t = sp.latex(jacob_inv_t)

# Output the LaTeX-formatted result
#print("\nLaTeX formatted expression:")
#print(latex_output_jacob_inv_t)

u = sp.symbols('u')
ub = sp.symbols('u_b')
beta = sp.symbols('beta')
f = sp.Matrix([[alpha * u, u]])
fX = jacob_det * (f @ jacob_inv_t)
fX_simplified = sp.simplify(fX)
#sp.pprint(fX_simplified)

# Define abstract indexed symbols for quadrature points and weights

gp = sp.IndexedBase('gp')  # gp[i] represents the i-th quadrature point in both xi and eta directions
gw = sp.IndexedBase('w')   # gw[i] represents the i-th quadrature weight

# Derivatives of basis functions
dphi_dxi = sp.symbols('dphi_dxi')
dphi_deta = sp.symbols('dphi_deta')
dphi_dX = sp.Matrix([[dphi_dxi, dphi_deta]])
# Define the integrand
volume_integrand = fX @ (dphi_dX.T)
#sp.pprint(integrand)
volume_integrand_simplified = sp.simplify(volume_integrand)
#sp.pprint(volume_integrand_simplified)
# Convert the simplified integrand to a LaTeX string
#latex_output = sp.latex(integrand_simplified)

# Output the LaTeX-formatted result
#print(latex_output)

volume_integrand_x = volume_integrand.diff(x1)
volume_integrand_x_simplified = sp.simplify(volume_integrand_x)
print("volume_integrand_dx1:")
sp.pprint(volume_integrand_x_simplified)
volume_integrand_y = volume_integrand.diff(y1)
volume_integrand_y_simplified = sp.simplify(volume_integrand_y)
print("volume_integrand_dy1:")
sp.pprint(volume_integrand_y_simplified)
'''
l = sp.IndexedBase('l')
u = sp.IndexedBase('u')
enriched_u = sp.Sum(l[i] * u[i], (i, 1, n))
'''
volume_integrand_u = volume_integrand.diff(u)
volume_integrand_u_simplified = sp.simplify(volume_integrand_u)
print("volume_integrand_du:")
sp.pprint(volume_integrand_u_simplified)

# Define the integral
#volume_integral = sp.Integral(integrand, (xi, -1, 1), (eta, -1, 1))
#sp.pprint(volume_integral)
#volume_integral_x1 = volume_integral.diff(x1)

#volume_integral_x1_simplified = sp.simplify(volume_integral_x1)

#sp.pprint(volume_integral_x1_simplified)
