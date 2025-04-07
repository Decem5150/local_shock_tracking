import sympy as sp
xi, eta = sp.symbols('xi eta')
ul, ur = sp.symbols('ul ur')
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
y1, y2, y3, y4 = sp.symbols('y1 y2 y3 y4')
#X1, X2, X3, X4 = sp.symbols('X1 X2 X3 X4')
#Y1, Y2, Y3, Y4 = sp.symbols('Y1 Y2 Y3 Y4')
N1 = sp.Function('N1')(xi, eta)
N2 = sp.Function('N2')(xi, eta)
N3 = sp.Function('N3')(xi, eta)
N4 = sp.Function('N4')(xi, eta)
# Define the mapping from reference (xi, eta) to physical coordinates (x, y)
x_expr = N1 * x1 + N2 * x2 + N3 * x3
y_expr = N1 * y1 + N2 * y2 + N3 * y3
x = sp.Function('x')(xi, eta)
y = sp.Function('y')(xi, eta)
# Form the mapping vector
F = sp.Matrix([x_expr, y_expr])
# Compute the Jacobian matrix with respect to (xi, eta)
J = F.jacobian([xi, eta])
# Simplify the Jacobian matrix entries
J = sp.simplify(J)
# Compute the Jacobian determinant
jacob_det = sp.simplify(J.det())
jacob_inv_t = J.inv().T.simplify()
nx = sp.symbols('nx')
ny = sp.symbols('ny')
n = sp.Matrix([nx, ny])
Nx = sp.symbols('Nx')
Ny = sp.symbols('Ny')
N = sp.Matrix([Nx, Ny])
alpha = sp.symbols('alpha')
beta = sp.Matrix([alpha, 1.0])
physical_flux = 0.5 * (beta.dot(n) * (ul + ur) + (beta.dot(n) * sp.tanh(100.0 * beta.dot(n))) * (ul - ur))
sp.pprint(physical_flux.diff(ul).simplify())
sp.pprint(physical_flux.diff(ur).simplify())
sp.pprint(physical_flux.diff(nx).simplify())
sp.pprint(physical_flux.diff(ny).simplify())
'''
physical_flux = sp.Function('h')(ul, ur, nx, ny)
sp.pprint((jacob_det * jacob_inv_t).simplify())
transformed_normal = (jacob_det * jacob_inv_t @ N)
sp.pprint(transformed_normal.simplify())
scale_factor = sp.sqrt(transformed_normal[0]**2 + transformed_normal[1]**2)
print("scale_factor")
sp.pprint(scale_factor.simplify())
scale_factor_dx = scale_factor.simplify().diff(y1)
print("scale_factor_dx")
sp.pprint(scale_factor_dx.simplify())
reference_flux = scale_factor * physical_flux
'''
#dflux_dx = reference_flux.diff(nx)
#dflux_dy = reference_flux.diff(y1)
#sp.pprint(dflux_dx.factor())
#sp.pprint(dflux_dy)
