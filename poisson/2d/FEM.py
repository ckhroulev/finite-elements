#!/usr/bin/env python
import numpy as np

class FEMElement:
    def __init__(self):
        self.x = None
        self.y = None

    def reset(self, pts):
        self.x = pts[:,0]
        self.y = pts[:,1]

    def det_J(self, point):
        """Determinant of the Jacobian."""
        return np.linalg.det(self.J(point))

    def J_inverse(self, point):
        """Inverse of the Jacobian."""
        return np.linalg.inv(self.J(point))

    def xy(self, point):
        """Isoparametric mapping from the reference element."""
        x = self.x
        y = self.y
        xx = 0
        yy = 0
        for j in xrange(self.n_chi()):
            xx += x[j] * self.chi(j, point)
            yy += y[j] * self.chi(j, point)

        return [xx, yy]

class P1(FEMElement):
    """P1 element"""

    def reset(self, pts):
        FEMElement.reset(self, pts)

        x = self.x
        y = self.y
        self._J = np.matrix([[x[1] - x[0], y[1] - y[0]],
                             [x[2] - x[0], y[2] - y[0]]])

        self._Jinv = np.linalg.inv(self._J)

        self._detJ = np.linalg.det(self._J)

    def n_chi(self):
        """Return the number of element basis functions."""
        return 3

    def chi(self, r, (xi, eta)):
        """P1 element basis functions."""
        if   r == 0:
            return 1 - xi - eta
        elif r == 1:
            return xi
        else:
            return eta

        raise ValueError("invalid argument (r)")

    def dchi(self, r, point):
        """Derivatives of P1 element basis functions.

        These are constant on the whole element."""
        if   r == 0:
            return np.matrix([-1, -1]).T
        elif r == 1:
            return np.matrix([1, 0]).T
        else:
            return np.matrix([0, 1]).T

        raise ValueError("invalid argument (r)")

    def J(self, point):
        """Jacobian of the linear map from the reference triangle.

        Constant on the whole element."""
        return self._J

    def J_inverse(self, point):
        """Inverse of the Jacobian."""
        return self._Jinv

    def det_J(self, point):
        return self._detJ

class Q1(FEMElement):
    """The Q1 element."""

    def n_chi(self):
        """Number of element basis functions."""
        return 4

    def chi(self, r, (xi, eta)):
        """Q1 element basis functions. See (1.26), page 23 of Elman and others."""
        if   r == 0:
            return (xi - 1)*(eta - 1) / 4.0
        elif r == 1:
            return (xi + 1)*(1 - eta) / 4.0
        elif r == 2:
            return (xi + 1)*(eta + 1) / 4.0
        elif r == 3:
            return (1 - xi)*(eta + 1) / 4.0

        raise ValueError("invalid argument (r)")

    def dchi(self, r, (xi, eta)):
        """Derivatives of Q1 element basis functions."""
        if   r == 0:
            return np.matrix([eta - 1, xi - 1]).T / 4.0
        elif r == 1:
            return np.matrix([1 - eta, - (xi + 1)]).T / 4.0
        elif r == 2:
            return np.matrix([eta + 1, xi + 1]).T / 4.0
        elif r == 3:
            return np.matrix([- (eta + 1), 1 - xi]).T / 4.0

        raise ValueError("invalid argument (r)")

    def J(self, (xi, eta)):
        """Jacobian of the transformation from the reference element."""
        x = self.x
        y = self.y
        return 0.25 * np.matrix([[eta*(-x[3]+x[2]-x[1]+x[0]) - x[3]+x[2]+x[1]-x[0], eta*(-y[3]+y[2]-y[1]+y[0]) - y[3]+y[2]+y[1]-y[0]],
                                 [ xi*(-x[3]+x[2]-x[1]+x[0]) + x[3]+x[2]-x[1]-x[0],  xi*(-y[3]+y[2]-y[1]+y[0]) + y[3]+y[2]-y[1]-y[0]]])

class Q1Aligned(Q1):
    """Q1 rectangular elements aligned with coordinate axes."""
    def reset(self, pts):
        FEMElement.reset(self, pts)
        x = self.x
        y = self.y

        dx = x[1] - x[0]
        dy = y[2] - y[0]

        self._J = np.matrix([[0.5*dx, 0],
                             [0, 0.5*dy]])

        self._J_inv = np.matrix([[2.0 / dx, 0],
                                 [0, 2.0 / dy]])

        self._detJ = 0.25 * dy * dy

    def J(self, point):
        return self._J

    def J_inverse(self, point):
        return self._J_inv

    def det_J(self, point):
        return self._detJ

class GaussTri1:
    """One-point Gaussian quadrature for the reference triangle."""
    def points(self):
        return np.array([[1, 1]]) / 3.0

    def weights(self):
        return [1.0]

class GaussQuad2x2:
    """Gaussian 2x2 quadrature for the reference quad."""
    def points(self):
        return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3)

    def weights(self):
        return [1.0, 1.0, 1.0, 1.0]

class GaussQuad1:
    """Gaussian 1-point quadrature for the reference quad."""
    def points(self):
        return np.array([[0, 0]])

    def weights(self):
        return [4.0]
