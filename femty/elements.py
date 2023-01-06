#%%
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import binom

from .mesh import Mesh


@dataclass(init=False)
class Cell(ABC):
    type: str
    vertices: ArrayLike
    num_dofs: int


class ReferenceQuadrilateral(Cell):
    type = "quadrilateral"
    vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    num_dofs = 4


class ReferenceTriangle(Cell):
    type = "triangle"
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    num_dofs = 3


def lagrange_points(cell: Cell, p: int) -> ArrayLike:
    if isinstance(cell, ReferenceTriangle):
        num_points = int(binom(p + 2, p))
        pts = np.zeros((num_points, 2))
        k = 0
        for i in range(0, p + 1):
            for j in range(0, p + 1 - i):
                pts[k] = [j / p, i / p]
                k += 1

    elif isinstance(cell, ReferenceQuadrilateral):
        x = np.zeros((p + 1, 1))
        for i in range(0, p + 1):
            x[i] = i / p

        pts = np.vstack([*map(np.ravel, np.meshgrid(x, x))]).T

    else:
        raise NotImplementedError()

    return pts


class BasisFunction(np.ndarray):
    def __new__(subtype, value, grad=None, *rest):
        obj = np.asarray(value).view(subtype)
        obj.grad = grad
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grad = getattr(obj, "grad", None)

    def __getitem__(self, key):
        return np.array(self)[key]

    @property
    def to_tuple(self):
        return tuple([np.array(self), self.grad])


def is_valid_space(space: str) -> bool:
    letter, degree = space[0], space[1:]
    return letter in ["P", "Q"] and degree.isnumeric()


class FiniteElement:
    space: str
    degree: int
    cell: Cell
    dofs: np.ndarray
    nodes: ArrayLike = None
    dim = 1

    def __init__(self, space: str, nodes: ArrayLike = None) -> None:
        if not is_valid_space(space):
            raise NotImplementedError(f"Space {space} is not implemented")
        self.space = space

        self.degree = int(space[1:])
        self.cell = {
            "P": ReferenceTriangle,
            "Q": ReferenceQuadrilateral,
        }[space[0]]()

        if nodes is None:
            self.nodes = lagrange_points(self.cell, self.degree)
        else:
            self.nodes = nodes

        if space[0] == "P":
            # see: https://coast.nd.edu/jjwteach/www/www/60130/New%20Lecture%20Notes_PDF/CE60130_Lecture%2015%20with_footer-v04.pdf
            # for the required coordinate transformation
            raise NotImplementedError(
                "Only Q elements are implemented. To fix this, I would need to implement a"
                + " coordinate transformation and then row-reduce the Vandermonde matrix."
            )

    # TODO: this should be a method of the specific FiniteElement implementation, not the base class
    # but we only use quadrilaterals so it's fine
    def local_basis(self, X: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        x, y = X

        if i == 0:
            phi = (1.0 - x) * (1.0 - y)
            dphi = np.array([-1.0 + y, -1.0 + x])
        elif i == 1:
            phi = x * (1.0 - y)
            dphi = np.array([1.0 - y, -x])
        elif i == 2:
            phi = (1.0 - x) * y
            dphi = np.array([-y, 1.0 - x])
        elif i == 3:
            phi = x * y
            dphi = np.array([y, x])
        else:
            raise IndexError()

        return phi, dphi

    def global_basis(self, mesh: Mesh, X: np.ndarray, i: int):
        phi, dphi = self.local_basis(X, i)
        invDF = mesh.calc_invDF(self, X)

        value = np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3]))
        grad = np.einsum("ijkl,il->jkl", invDF, dphi)

        return (BasisFunction(value, grad=grad),)

    @property
    def dofs(self):
        return self.cell.vertices


class VectorElement(FiniteElement):
    element: FiniteElement
    dim = 2

    def __init__(self, element: FiniteElement) -> None:
        self.element = element
        self._dofs = np.array(
            [
                element.dofs[int(np.floor(float(i) / float(self.dim)))]
                for i in range(self.dim * element.dofs.shape[0])
            ]
        )

    def global_basis(self, mesh: Mesh, X: np.ndarray, i: int):
        ind = int(np.floor(float(i) / float(self.dim)))
        n = i - self.dim * ind
        fields = []
        for field in self.element.global_basis(mesh, X, ind)[0].to_tuple:
            if field is None:
                fields.append(None)
            else:
                tmp = np.zeros((self.dim,) + field.shape)
                tmp[n] = field
                fields.append(tmp)

        return (BasisFunction(*fields),)

    @property
    def space(self):
        return self.element.space

    @property
    def nodes(self):
        return self.element.nodes

    @property
    def degree(self):
        return self.element.degree

    @property
    def cell(self):
        return self.element.cell

    @property
    def dofs(self):
        return self._dofs


class Dofmap:
    """
    Keeps track of the degrees of freedom (the corner vertices in each element).
    """

    def __init__(
        self,
        domain: Mesh,
        element: FiniteElement,
    ) -> None:
        self.domain = domain
        self.element = element

        self.dof_array = domain.vertices.ravel()
        self.size = np.unique(self.dof_array).size

    def cell_dofs(self, i: int) -> np.ndarray:
        num_dofs = self.element.cell.num_dofs
        return self.dof_array[i * num_dofs : (i + 1) * num_dofs]


class FunctionSpace:
    def __init__(self, domain, element, quad) -> None:
        self.domain = domain
        self.element = element
        self.quad = quad


# %%
