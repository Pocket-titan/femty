from typing import Callable, Literal, Union

from .quadrature import Quadrature
from .basis import Basis

import numpy as np
from scipy import sparse


def assemble_matrix(
    ubasis: Basis,
    vbasis: Basis,
    quad: Quadrature,
    form: Callable,
):
    nt = ubasis.num_cells
    uNbfun = 4 * ubasis.element.dim
    vNbfun = 4 * vbasis.element.dim

    X = quad.points.T
    W = quad.weights.T
    dx = np.abs(ubasis.mesh.calc_detDF(ubasis.element, X)) * np.tile(W, (nt, 1))

    basis = dict(
        u=[ubasis.element.global_basis(ubasis.mesh, X, j) for j in range(uNbfun)],
        v=[vbasis.element.global_basis(vbasis.mesh, X, j) for j in range(vNbfun)],
    )

    sz = uNbfun * vNbfun * nt
    data = np.zeros((uNbfun, vNbfun, nt))
    rows = np.zeros(sz, dtype=np.int64)
    cols = np.zeros(sz, dtype=np.int64)

    for j in range(uNbfun):
        for i in range(vNbfun):
            ixs = slice(nt * (vNbfun * j + i), nt * (vNbfun * j + i + 1))
            rows[ixs] = vbasis.dofmap.element_dofs[i]
            cols[ixs] = ubasis.dofmap.element_dofs[j]
            data[j, i, :] = np.sum(form(*basis["u"][j], *basis["v"][i], {}) * dx, axis=1)

    data = data.flatten("C")

    args = (
        np.array([rows, cols]),
        data,
        (vbasis.dofmap.num_dofs, ubasis.dofmap.num_dofs),
        (uNbfun, vNbfun),  # local_shape
    )

    K = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(vbasis.dofmap.num_dofs, ubasis.dofmap.num_dofs),
    ).tocsr()

    return K


def assemble_vector(ubasis: Basis, quad: Quadrature, form: Callable):
    nt = ubasis.num_cells
    Nbfun = 4 * ubasis.element.dim

    X = quad.points.T
    W = quad.weights.T
    dx = np.abs(ubasis.mesh.calc_detDF(ubasis.element, X)) * np.tile(W, (nt, 1))

    basis = [ubasis.element.global_basis(ubasis.mesh, X, j) for j in range(Nbfun)]

    x = ubasis.dofmap.F(X)

    sz = Nbfun * nt
    data = np.zeros(sz)
    rows = np.zeros(sz, dtype=np.int64)

    for i in range(Nbfun):
        ixs = slice(nt * i, nt * (i + 1))
        rows[ixs] = ubasis.dofmap.element_dofs[i]
        data[ixs] = np.sum(form(*basis[i], {"x": x}) * dx, axis=1)

    b = sparse.coo_matrix(
        (data, (rows, np.zeros_like(rows))),
        shape=(ubasis.dofmap.num_dofs,) + (1,),
    )

    return b.toarray().T[0]


def apply_boundary_conditions(A, b, basis, value=None):
    dim = 2
    nodal_ix = basis.mesh.find_boundary_vertices()
    nodal_rows = np.arange(dim)

    dofs = basis.dofmap.nodal_dofs[nodal_rows][:, nodal_ix]
    D = dofs.flatten()
    I = np.setdiff1d(np.arange(A.shape[0]), D)

    Aout = A[I].T[I].T
    if value is None:
        value = np.zeros((A.shape[0],))
    bout = b[I] - A[I].T[D].T @ value[D]

    return Aout, bout, value, I


# @dataclass
# class Quadrature:
#     degree: int = 2
#     points: np.ndarray = field(init=False)
#     weights: np.ndarray = field(init=False)

#     def __post_init__(self) -> None:
#         self.points, self.weights = self.gauss_quadrature_2d()

#     def gauss_quadrature_1d(self, range=[0, 1]):
#         a, b = range[0], range[1]
#         num_points = int((self.degree + 1 + 1) / 2)
#         points, weights = np.polynomial.legendre.leggauss(num_points)

#         points = points * (b - a) / 2 + (a + b) / 2
#         weights = weights * (b - a) / 2

#         return points.reshape(-1), weights

#     def gauss_quadrature_2d(self, range=[[0, 1], [0, 1]]):
#         xpoints, xweights = self.gauss_quadrature_1d(range[0])
#         ypoints, yweights = self.gauss_quadrature_1d(range[1])

#         points = np.array(list(product(xpoints, ypoints)))
#         weights = np.array([x * y for x, y in product(xweights, yweights)])

#         return points, weights


# def sparsity_pattern(dofmap: Dofmap) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Vertex at row index `i` has influence on vertex at column index `j` if they
#     are in the same element; this pattern reflects that.
#     """
#     num_cells = dofmap.domain.num_cells
#     dofs_per_cell = dofmap.element.cell.num_dofs

#     rows = np.repeat(dofmap.dof_array, dofs_per_cell)
#     cols = np.tile(
#         np.reshape(dofmap.dof_array, (num_cells, dofs_per_cell)),
#         dofs_per_cell,
#     )
#     return rows, cols.ravel()


# def linear_integral(
#     coords: np.ndarray,
#     f: Callable,
#     element: FiniteElement,
#     area: float,
#     quad: Quadrature,
# ) -> np.ndarray:
#     """
#     \int_{\Omega_e} f(x) \phi_i(x) dx
#     """
#     # Index (i, j) is the value of \phi_j at point i
#     local_basis = element.basis(quad.points[:, 0], quad.points[:, 1])
#     global_points = np.dot(local_basis, coords)
#     sampled_func = np.apply_along_axis(f, 1, global_points)
#     return (local_basis * quad.weights).transpose() @ sampled_func * area


# def stiffness_integral(
#     coords: np.ndarray,
#     element: FiniteElement,
#     jacobian: np.ndarray,
#     quad: Quadrature,
# ) -> np.ndarray:
#     """
#     A^e_{i,j} = \sum_i \sum_{p} J^{-1} d_{x_i} \phi(i,p) J^{-1} d_{x_i} \phi(p,j) weights(p) * detJ
#     """
#     gradient = element.basis(quad.points[:, 0], quad.points[:, 1], grad=True)
#     derivative_dx = gradient[:, :, 0]
#     derivative_dy = gradient[:, :, 1]

#     inv_jacobian = np.linalg.inv(jacobian)
#     area = np.linalg.det(jacobian)

#     A_x = (derivative_dx.T @ derivative_dx) * quad.weights * inv_jacobian[0, 0] ** 2
#     A_y = (derivative_dy.T @ derivative_dy) * quad.weights * inv_jacobian[1, 1] ** 2
#     A_e = (A_x + A_y) * area
#     return A_e


# def mass_integral(
#     coords: np.ndarray,
#     element: FiniteElement,
#     jacobian: np.ndarray,
#     quad: Quadrature,
# ) -> np.ndarray:
#     """
#     \int_{\Omega_e} \phi_i(x) \phi_j(x) J^{-1} detJ dx
#     """
#     basis_dx = element.basis(quad.points[:, 0], quad.points[:, 1])
#     basis_dy = element.basis(quad.points[:, 0], quad.points[:, 1])

#     area = np.linalg.det(jacobian)

#     M_x = (basis_dx.T @ basis_dx) * quad.weights
#     M_y = (basis_dy.T @ basis_dy) * quad.weights
#     M_e = (M_x + M_y) * area
#     return M_e


# def divergence_integral(
#     coords: np.ndarray,
#     element: FiniteElement,
#     jacobian: np.ndarray,
#     quad: Quadrature,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     \int_{\Omega_e} \phi_i(x) \\nabla \cdot \phi_j(x) J^{-1} detJ dx
#     """
#     gradient = element.basis(quad.points[:, 0], quad.points[:, 1], grad=True)
#     derivative_dx = gradient[:, :, 0]
#     derivative_dy = gradient[:, :, 1]
#     basis_dx = element.basis(quad.points[:, 0], quad.points[:, 1])
#     basis_dy = element.basis(quad.points[:, 0], quad.points[:, 1])

#     inv_jacobian = np.linalg.inv(jacobian)
#     area = np.linalg.det(jacobian)

#     div_x = derivative_dx * inv_jacobian[0, 0]
#     div_y = derivative_dy * inv_jacobian[1, 1]

#     B_x = -(div_x.T @ basis_dx) * quad.weights * area
#     B_y = -(div_y.T @ basis_dy) * quad.weights * area
#     return B_x, B_y


# def nonlinear_integral(
#     coords: np.ndarray,
#     element: FiniteElement,
#     jacobian: np.ndarray,
#     quad: Quadrature,
#     u: np.ndarray,
# ) -> np.ndarray:
#     gradient = element.basis(quad.points[:, 0], quad.points[:, 1], grad=True)
#     derivative_dx = gradient[:, :, 0]
#     derivative_dy = gradient[:, :, 1]
#     basis_dx = element.basis(quad.points[:, 0], quad.points[:, 1])
#     basis_dy = element.basis(quad.points[:, 0], quad.points[:, 1])

#     inv_jacobian = np.linalg.inv(jacobian)
#     area = np.linalg.det(jacobian)

#     [u_xs, u_ys] = np.split(u, 2)

#     N_x = (u_xs[None, :].T * (derivative_dx * inv_jacobian[0, 0])) @ basis_dx * quad.weights
#     N_y = (u_ys[None, :].T * (derivative_dy * inv_jacobian[1, 1])) @ basis_dy * quad.weights
#     return (N_x + N_y) * area


# def assemble_vector(
#     fs: FunctionSpace,
#     kind: Literal[
#         "velocity_x",
#         "velocity_y",
#         "pressure",
#     ],
#     degree=2,
#     f: Callable = None,
# ) -> np.ndarray:
#     """
#     Assemble the velocity (v, also called 'load vector' b) or pressure vector (p)
#     """
#     if kind == "pressure" and f is not None:
#         raise Exception("Pressure vector does not take a source term!")
#     if f is None:
#         f = lambda _: 0

#     mesh, dofmap = fs.mesh, fs.dofmap
#     cells = mesh.cells.ravel()

#     quad = Quadrature(degree)

#     b = np.zeros(dofmap.size)

#     for i, cell_number in enumerate(cells):
#         dofs = dofmap.cell_dofs(i)
#         coords = mesh.coordinates[dofs]

#         b[dofs] += linear_integral(
#             coords,
#             f,
#             dofmap.element,
#             mesh.area(cell_number),
#             quad,
#         )

#     return b


# def assemble_matrix(
#     fs: FunctionSpace,
#     kind: Literal[
#         "stiffness",
#         "mass",
#         "divergence_x",
#         "divergence_y",
#         "nonlinear",
#     ],
#     degree=2,
#     u: np.ndarray = None,
#     kinematic_viscosity=1,
# ) -> sparse.coo_matrix:
#     """
#     Assemble the stiffness (A, or K), (pressure) mass (M_p, or M) or divergence (B) matrix.
#     """
#     if kind == "stiffness":
#         integral = stiffness_integral
#     elif kind == "mass":
#         integral = mass_integral
#     elif kind == "nonlinear":
#         if u is None:
#             raise Exception("Velocities are required.")
#         integral = nonlinear_integral
#     elif "divergence" in kind:
#         idx = 0 if "_x" in kind else 1
#         integral = lambda *a, **b: divergence_integral(*a, **b)[idx]
#     else:
#         raise NotImplementedError()

#     mesh, element, dofmap = fs.mesh, fs.element, fs.dofmap
#     cells = mesh.cells.ravel()

#     quad = Quadrature(degree)

#     A = np.zeros(cells.size * element.cell.num_dofs**2)

#     for i in range(cells.size):
#         dofs = dofmap.cell_dofs(i)
#         coords = mesh.coordinates[dofs]
#         cell_number = cells[i]

#         kwargs = {}
#         if kind == "nonlinear":
#             idxs = np.concatenate([dofs, int(u.size / 2) + dofs])
#             kwargs["u"] = u[idxs]

#         A_e = integral(
#             coords,
#             element,
#             mesh.jacobian(cell_number),
#             quad,
#             **kwargs,
#         )

#         A[i * A_e.size : i * A_e.size + A_e.size] = A_e.ravel()

#     A = sparse.coo_matrix(
#         (A, sparsity_pattern(dofmap)),
#         shape=(dofmap.size, dofmap.size),
#     ).tocsr()

#     if kind == "stiffness":
#         # Re = 50
#         # kinematic_viscosity = 1/Re
#         A *= kinematic_viscosity

#     return A


# def apply_boundary_conditions(
#     vertices: np.ndarray,
#     A: sparse.spmatrix = None,
#     b: np.ndarray = None,
#     value: Union[float, np.ndarray] = 0,
# ) -> None:
#     """
#     Set the values of `vertices` in matrix `A` and vector `b` to `value`.
#     """
#     if A is not None:
#         if sparse.isspmatrix_csr(A):
#             for dof in vertices:
#                 A.data[A.indptr[dof] : A.indptr[dof + 1]] = 0.0
#                 A[dof, dof] = 1.0

#             A.eliminate_zeros()
#         else:
#             raise TypeError("Matrix must be of csr format.")

#     if type(value) is np.ndarray:
#         assert len(value) == len(vertices)

#     if b is not None:
#         b[vertices] = value
