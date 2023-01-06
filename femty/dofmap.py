#%%
from dataclasses import dataclass, field

import numpy as np

from .elements import FiniteElement
from .mesh import Mesh


@dataclass
class Dofmap:
    mesh: Mesh
    element: FiniteElement
    element_dofs: np.ndarray = field(init=False)
    nodal_dofs: np.ndarray = field(init=False)
    num_dofs: int = field(init=False)

    def __post_init__(self) -> None:
        self.nodal_dofs = np.reshape(
            np.arange(
                self.element.dim * self.mesh.num_vertices, dtype=np.int64
            ),
            (self.element.dim, self.mesh.num_vertices),
            order="F",
        )

        self.element_dofs = np.zeros((0, self.mesh.num_cells), dtype=np.int64)

        for itr in range(self.mesh.vertices.T.shape[0]):
            self.element_dofs = np.vstack(
                (
                    self.element_dofs,
                    self.nodal_dofs[:, self.mesh.vertices.T[itr]],
                )
            )

        self.num_dofs = np.max(self.element_dofs) + 1

    def Fmap(self, X, i):
        p = self.mesh.coordinates.T
        t = self.mesh.vertices.T

        out = np.zeros((t.shape[1], X.shape[1]))

        for itr in range(t.shape[0]):
            phi, _ = self.element.local_basis(X, itr)
            out += p[i, t[itr, :]][:, None] * phi

        return out

    def F(self, X):
        return np.array([self.Fmap(X, i) for i in range(X.shape[0])])
