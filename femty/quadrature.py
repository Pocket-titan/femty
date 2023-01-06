from itertools import product
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Quadrature:
    degree: int = 2
    points: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.points, self.weights = self.gauss_quadrature_2d()

    def gauss_quadrature_1d(self, range=[0, 1]):
        a, b = range[0], range[1]
        num_points = int((self.degree + 1 + 1) / 2)
        points, weights = np.polynomial.legendre.leggauss(num_points)

        points = points * (b - a) / 2 + (a + b) / 2
        weights = weights * (b - a) / 2

        return points.reshape(-1), weights

    def gauss_quadrature_2d(self, range=[[0, 1], [0, 1]]):
        xpoints, xweights = self.gauss_quadrature_1d(range[0])
        ypoints, yweights = self.gauss_quadrature_1d(range[1])

        points = np.array(list(product(xpoints, ypoints)))
        weights = np.array([x * y for x, y in product(xweights, yweights)])

        return points, weights
