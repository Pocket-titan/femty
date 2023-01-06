#%%
from typing import Callable, Literal, Union

import numpy as np
from scipy import sparse


def compute_cells(num_cells: int, nx: int, ny: int):
    """
    Compute the connectivity (corner vertices) of the cells.
    """
    cells = np.reshape(np.fromiter(range(num_cells), dtype=int), (nx, ny))
    vertices = np.zeros((num_cells, 4), dtype=int)

    for cell in cells.ravel():
        i = cell % nx
        j = cell // nx

        vertices[cell] = [
            j * (nx + 1) + i,
            j * (nx + 1) + i + 1,
            (j + 1) * (nx + 1) + i,
            (j + 1) * (nx + 1) + i + 1,
        ]

    return cells, vertices


def find_outer_vertices(
    cells: np.ndarray,
    vertices: np.ndarray,
    on_edges: list[bool],
    shrink: int = 0,
    where: Literal["left", "right", "top", "bottom", "all"] = "all",
) -> np.ndarray:
    """
    Find the outer vertices of the given mesh cells (2d array), optionally shrinking
    by `shrink` cells on each side (so `shrink=1` gives the edges 1 layer in).
    """
    on_left_edge, on_right_edge, on_top_edge, on_bottom_edge = on_edges

    left = cells[:, 0 : 1 + (shrink if not on_left_edge else 0)]
    right = cells[:, -1 - (shrink if not on_right_edge else 0) :]
    top = cells[0 : 1 + (shrink if not on_top_edge else 0), :]
    bottom = cells[-1 - (shrink if not on_bottom_edge else 0) :, :]

    edges = {
        k: find_cell_edge(v.flatten(), vertices, k)
        for k, v in zip(
            ["left", "right", "top", "bottom"],
            [left, right, top, bottom],
        )
    }

    # Because of the indexing difference between 'ij' and 'xy'...
    SWAP_TOP_AND_BOTTOM = True
    if SWAP_TOP_AND_BOTTOM:
        edges["top"], edges["bottom"] = edges["bottom"], edges["top"]

    if where in ["left", "right", "top", "bottom"]:
        return np.unique(edges[where])

    return np.unique(np.concatenate([*edges.values()]))


def find_cell_edge(
    cell: Union[int, np.ndarray],
    vertices: np.ndarray,
    edge: str,
) -> np.ndarray:
    if edge == "left":
        idxs = [0, 2]
    if edge == "right":
        idxs = [1, 3]
    if edge == "top":
        idxs = [0, 1]
    if edge == "bottom":
        idxs = [2, 3]

    return vertices[cell][..., idxs]


class Mesh:
    dim = 2
    _J = None

    def __init__(self, nx: int, ny: int) -> None:
        self.num_vertices = (nx + 1) * (ny + 1)
        self.num_cells = nx * ny
        self.nx = nx
        self.ny = ny

        x = np.linspace(0, 1, nx + 1)
        y = np.linspace(0, 1, ny + 1)
        grid = np.array(np.meshgrid(x, y, indexing="ij")).transpose()

        cells, vertices = compute_cells(self.num_cells, nx, ny)
        # Global cell numbers (== local cell numbers; indices)
        self.cells = cells
        # Global vertex numbers, index with (global) cells
        self.vertices = vertices
        # Global vertex coordinates, index with (global) vertices
        self.coordinates = grid.reshape((nx + 1) * (ny + 1), 2)

    # TODO: this is constant & the same for each element; cache it
    def jacobian(self, i: int):
        vertex_coords = self.coordinates[self.vertices[i]]
        dx = vertex_coords[1, 0] - vertex_coords[0, 0]
        dy = vertex_coords[2, 1] - vertex_coords[0, 1]

        jacobian = np.array([[dx, 0], [0, dy]])
        return jacobian

    # TODO: this is constant & the same for each element; cache it
    def area(self, i: int):
        return np.linalg.det(self.jacobian(i))

    def find_boundary_vertices(
        self,
        where: Literal["left", "right", "top", "bottom"] = "all",
    ) -> np.ndarray:
        return find_outer_vertices(
            self.cells,
            self.vertices,
            [False, False, False, False],
            shrink=0,
            where=where,
        )

    def calc_J(self, element, X: np.ndarray, i: int, j: int):
        p = self.coordinates.T
        t = self.vertices.T

        out = np.zeros((t.shape[1], X.shape[1]))

        for itr in range(t.shape[0]):
            _, dphi = element.local_basis(X, itr)
            out += p[i, t[itr, :]][:, None] * dphi[j]

        return out

    def calc_detDF(self, element, X: np.ndarray):
        J = [[self.calc_J(element, X, i, j) for j in range(self.dim)] for i in range(self.dim)]
        return J[0][0] * J[1][1] - J[0][1] * J[1][0]

    def calc_invDF(self, element, X: np.ndarray):
        J = [[self.calc_J(element, X, i, j) for j in range(self.dim)] for i in range(self.dim)]
        invDF = np.empty((self.dim, self.dim) + J[0][0].shape)
        invDF[0, 0] = J[1][1]  # noqa
        invDF[0, 1] = -J[0][1]
        invDF[1, 0] = -J[1][0]
        invDF[1, 1] = J[0][0]  # noqa

        detDF = J[0][0] * J[1][1] - J[0][1] * J[1][0]

        invDF /= detDF
        return invDF

    @property
    def J(self):
        if self._J is None:
            raise Exception()
        return self._J


# %%
