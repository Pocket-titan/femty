#%%
from dataclasses import dataclass, field

import numpy as np

from .dofmap import Dofmap
from .elements import FiniteElement
from .mesh import Mesh
from .quadrature import Quadrature


@dataclass
class Basis:
    mesh: Mesh
    element: FiniteElement
    quad: Quadrature
    dofmap: Dofmap = field(init=False)

    def __post_init__(self) -> None:
        self.dofmap = Dofmap(self.mesh, self.element)

    @property
    def num_cells(self):
        return self.mesh.num_cells
