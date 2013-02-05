#!/usr/bin/env python
import numpy as np

class Q13D:
    """3D Q1 hexahedral finite element with equal spacing in x and y directions.

    Pre-computes values of element basis functions and derivatives of element basis functions
    when created and uses table lookup during the assembly.

    Computes J^{-1} and det(J)*w once per element and uses table lookup later.

    Q13D.chi[i,j] is the element basis function i at the quadrature point j.

    Q13D.detJxW[j] is the determinant of the Jacobian at the quadrature point j
    times the corresponding quadrature weight.

    Q13D.dphi_xy[i,j,k] is the derivative of the shape function i at the quadrature point j
    with respect to variable k. (In the (x,y,z) system.)
    """
    def __init__(self, quadrature):
        self.quadrature = quadrature
        self.chi = None
        self._dchi = None
        self.dphi_xy = None
        self.detJxW = None
        self.xis   = [-1,  1,  1, -1, -1,  1, 1, -1]
        self.etas  = [-1, -1,  1,  1, -1, -1, 1,  1]
        self.zetas = [-1, -1, -1, -1,  1,  1, 1,  1]

        # pre-compute values of element basis functions at all quadrature points
        self.chi = np.zeros((self.n_chi(), self.quadrature.n_pts))

        for i in xrange(self.n_chi()):
            for j in xrange(self.quadrature.n_pts):
                self.chi[i,j] = self._chi(i, self.quadrature.pts[j])

        # pre-compute values of derivatives of element basis functions at all quadrature points
        self._dchi = np.zeros((self.n_chi(), self.quadrature.n_pts, 3))

        for i in xrange(self.n_chi()):
            for j in xrange(self.quadrature.n_pts):
                self._dchi[i,j,:] = self.__dchi(i, self.quadrature.pts[j])

    def n_quadrature_points(self):
        return self.quadrature.n_pts

    def n_chi(self):
        """Number of element basis functions."""
        return 8

    def _dz(self, k, z):
        """Compute the gradient of z (in the xi,eta,zeta coordinate system) at the quadrature point k."""
        dz_dxi   = 0
        dz_deta  = 0
        dz_dzeta = 0
        for i in xrange(self.n_chi()):
            dz_dxi   += z[i] * self._dchi[i, k, 0]
            dz_deta  += z[i] * self._dchi[i, k, 1]
            dz_dzeta += z[i] * self._dchi[i, k, 2]

        return [dz_dxi, dz_deta, dz_dzeta]

    def _chi(self, i, pt):
        """Element basis function i at a point pt."""
        xi,eta,zeta = pt
        return 0.125 * (1.0 + self.xis[i]*xi) * (1.0 + self.etas[i]*eta) * (1.0 + self.zetas[i]*zeta)

    def __dchi(self, i, pt):
        """Derivatives of the element basis function i at a point pt."""
        xi,eta,zeta = pt
        dchi_dxi   = 0.125 *   self.xis[i] * (1.0 + self.etas[i] * eta) * (1.0 + self.zetas[i] * zeta)
        dchi_deta  = 0.125 *  self.etas[i] * (1.0 +  self.xis[i] *  xi) * (1.0 + self.zetas[i] * zeta)
        dchi_dzeta = 0.125 * self.zetas[i] * (1.0 +  self.xis[i] *  xi) * (1.0 +  self.etas[i] *  eta)

        return [dchi_dxi, dchi_deta, dchi_dzeta]

    def reset(self, pts):
        x = pts[:,0]
        y = pts[:,1]
        z = pts[:,2]

        # pre-compute J^{-1} and det(J)*w at all quadrature points
        dx = x[1] - x[0]
        dy = y[2] - y[0]
        self.detJxW = np.zeros(self.quadrature.n_pts)
        self.dphi_xy = np.zeros((self.n_chi(), self.quadrature.n_pts, 3))
        for i in xrange(self.quadrature.n_pts):
            dz = self._dz(i, z)

            jac = np.array([[0.5*dx,    0.0, dz[0]],
                            [0.0,    0.5*dy, dz[1]],
                            [0.0,       0.0, dz[2]]])

            J_inv = np.matrix([[1.0/jac[0,0],          0.0, -jac[0,2]/(jac[0,0]*jac[2,2])],
                               [0.0,          1.0/jac[1,1], -jac[1,2]/(jac[1,1]*jac[2,2])],
                               [0.0,                   0.0,                 1.0/jac[2,2]]])

            for j in xrange(self.n_chi()):
                self.dphi_xy[j,i,:] = (J_inv * np.matrix(self._dchi[j,i,:]).T).T

            self.detJxW[i] = jac[0,0] * jac[1,1] * jac[2,2] * self.quadrature.weights[i]

class Q13DEquallySpaced(Q13D):
    """Elements equallly spaced in all 3 directions."""
    def reset(self, pts):
        if self.detJxW is not None:
            return

        x = pts[:,0]
        y = pts[:,1]
        z = pts[:,2]

        dx = x[1] - x[0]
        dy = y[2] - y[0]
        dz = z[4] - z[0]

        self.detJxW  = np.array(self.quadrature.weights * 8.0/(dx*dy*dz))
        self.dphi_xy = np.zeros((self.n_chi(), self.quadrature.n_pts, 3))
        for i in xrange(self.n_chi()):
            for j in xrange(self.quadrature.n_pts):
                self.dphi_xy[i,j,0] = self._dchi[i,j,0] * (2.0/dx)
                self.dphi_xy[i,j,1] = self._dchi[i,j,1] * (2.0/dy)
                self.dphi_xy[i,j,2] = self._dchi[i,j,2] * (2.0/dz)

class Gauss1:
    def __init__(self):
        self.pts = np.array([[0, 0, 0]])
        self.n_pts = 1
        self.weights = np.array([8])

class Gauss2x2x2:
    def __init__(self):
        # coordinates of quadrature points (sans the 1/sqrt(3)):
        xis   = [-1,  1,  1, -1, -1,  1, 1, -1]
        etas  = [-1, -1,  1,  1, -1, -1, 1,  1]
        zetas = [-1, -1, -1, -1,  1,  1, 1,  1]

        self.n_pts = 8
        self.pts = np.vstack((xis, etas, zetas)).T / np.sqrt(3)
        self.weights = np.ones(8)

if __name__ == "__main__":
    quadrature = Gauss2x2x2()
    E = Q13D(quadrature)

    # Test the Q13D class:

    def p(number):
        """Custom print function."""
        print ("%05.5f" % float(number)).rjust(8," "),

    # print values of element basis functions to compare to Maxima
    print "chi"
    for j in xrange(quadrature.n_pts):
        for i in xrange(E.n_chi()):
            p(E.chi[i,j])
        print ""

    # print derivatives of element basis functions to compare to Maxima
    for k in [0, 1, 2]:
        print "dchi[%d]" % k
        for j in xrange(quadrature.n_pts):
            for i in xrange(E.n_chi()):
                p(E._dchi[i,j,k])
            print ""

    xs = [-1,  1,  1, -1, -1,  1, 1, -1]
    ys = [-1, -1,  1,  1, -1, -1, 1,  1]
    zs = [-1, -1, -1, -1,  1,  1, 1,  1]

    # twice the size (8 times the volume):
    pts = np.vstack((xs, ys, zs)).T * 2

    # Initialize the element (and pre-compute J^{-1} and det(J))
    E.reset(pts)

    # print derivatives of element basis functions to compare to Maxima
    for k in [0, 1, 2]:
        print "dphi_xy[%d]" % k
        for j in xrange(quadrature.n_pts):
            for i in xrange(E.n_chi()):
                p(E.dphi_xy[i,j,k])
            print ""

    # print det(J)*w at quadrature points
    print "det(J)*w"
    for j in xrange(quadrature.n_pts):
        p(E.detJxW[j])
    print ""
