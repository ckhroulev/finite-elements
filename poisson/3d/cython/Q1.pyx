# -*- mode: cython -*-
#cython: boundscheck=False
#cython: embedsignature=True
#cython: wraparound=False
import numpy as np
cimport numpy as np

ctypedef np.float64_t double_t
ctypedef np.int32_t   int_t

cdef class Q13D:
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

    cdef np.ndarray _dchi, weights, xis, etas, zetas
    cdef public np.ndarray chi, detJxW, dphi_xy
    cdef public int n_pts, n_chi

    def __init__(self, quadrature):
        self.n_chi = 8
        self.n_pts = quadrature.n_pts
        self.weights = quadrature.weights
        self.chi = None
        self._dchi = None
        self.dphi_xy = None
        self.detJxW = None
        self.xis   = np.array([-1,  1,  1, -1, -1,  1, 1, -1])
        self.etas  = np.array([-1, -1,  1,  1, -1, -1, 1,  1])
        self.zetas = np.array([-1, -1, -1, -1,  1,  1, 1,  1])

        # pre-compute values of element basis functions at all quadrature points
        self.chi = np.zeros((self.n_chi, self.n_pts))

        for i in xrange(self.n_chi):
            for j in xrange(self.n_pts):
                self.chi[i,j] = self._chi(i, quadrature.pts[j])

        # pre-compute values of derivatives of element basis functions at all quadrature points
        self._dchi = np.zeros((self.n_chi, self.n_pts, 3))

        for i in xrange(self.n_chi):
            for j in xrange(self.n_pts):
                self._dchi[i,j,:] = self.__dchi(i, quadrature.pts[j])

    def _dz(self, k, z):
        """Compute the gradient of z (in the xi,eta,zeta coordinate system) at the quadrature point k."""
        dz_dxi   = 0
        dz_deta  = 0
        dz_dzeta = 0
        for i in xrange(self.n_chi):
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

    def reset(self, double_t[:,:] pts):
        cdef double_t[:] x = pts[:,0]
        cdef double_t[:] y = pts[:,1]
        cdef double_t[:] z = pts[:,2]
        cdef double_t dx, dy
        cdef np.ndarray[dtype=double_t, ndim=2, mode='c'] jac, J_inv

        # pre-compute J^{-1} and det(J)*w at all quadrature points
        dx = x[1] - x[0]
        dy = y[2] - y[0]
        self.detJxW = np.zeros(self.n_pts)
        self.dphi_xy = np.zeros((self.n_chi, self.n_pts, 3))
        for i in xrange(self.n_pts):
            dz = self._dz(i, z)

            jac = np.array([[0.5*dx,    0.0, dz[0]],
                            [0.0,    0.5*dy, dz[1]],
                            [0.0,       0.0, dz[2]]])

            J_inv = np.matrix([[1.0/jac[0,0],          0.0, -jac[0,2]/(jac[0,0]*jac[2,2])],
                               [0.0,          1.0/jac[1,1], -jac[1,2]/(jac[1,1]*jac[2,2])],
                               [0.0,                   0.0,                 1.0/jac[2,2]]])

            for j in xrange(self.n_chi):
                self.dphi_xy[j,i,:] = (J_inv * np.matrix(self._dchi[j,i,:]).T).T

            self.detJxW[i] = jac[0,0] * jac[1,1] * jac[2,2] * self.weights[i]

cdef class Q13DEquallySpaced(Q13D):
    """Elements equallly spaced in all 3 directions."""
    def reset(self, double_t[:,:] pts):
        cdef double_t[:] x = pts[:,0]
        cdef double_t[:] y = pts[:,1]
        cdef double_t[:] z = pts[:,2]
        cdef double_t dx, dy, dz
        cdef double_t[:,:,:] dphi_xy, dchi
        cdef int n_chi = self.n_chi
        cdef int n_pts = self.n_pts

        if self.detJxW is not None:
            return

        dx = x[1] - x[0]
        dy = y[2] - y[0]
        dz = z[4] - z[0]

        self.detJxW  = np.array(self.weights * 8.0/(dx*dy*dz))
        self.dphi_xy = np.zeros((n_chi, n_pts, 3))
        dphi_xy = self.dphi_xy
        dchi = self._dchi
        for i in xrange(n_chi):
            for j in xrange(n_pts):
                dphi_xy[i,j,0] = dchi[i,j,0] * (2.0/dx)
                dphi_xy[i,j,1] = dchi[i,j,1] * (2.0/dy)
                dphi_xy[i,j,2] = dchi[i,j,2] * (2.0/dz)

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
