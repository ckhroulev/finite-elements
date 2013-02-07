# -*- mode: cython -*-
#cython: boundscheck=False
#cython: embedsignature=True
#cython: wraparound=False
from scipy.sparse import lil_matrix
import numpy as np
cimport numpy as np

ctypedef np.float64_t double_t
ctypedef np.int32_t   int_t

def poisson(double_t[:, :] pts,
            int_t[:,:] elements,
            int_t[:] bc_nodes,
            double_t[:] f,
            double_t[:] u_bc,
            element,
            float scaling=1.0):
    """Set up a Galerkin FEM system approximating the Poisson equation with Dirichlet B.C
    on a domain described by a mesh in (pts, elements).

    This code can be used with different element types and different quadratures.

    This code pre-computes J^{-1}*dchi and det(J)*w at quadrature points.

    Parameters:
    - pts: list of node coordinates (2D array)
    - elements: list of lists, each element contains indices in pts of nodes that belong
      to a given quad or triangle
    - bc_nodes: 1D array of 0 and 1; bc_nodes[k] == 1 iff pts[k] is a Dirichlet B.C. node
    - f: forcing term: array with len(pts) elements
    - u_bc: Dirichlet boundary condition: array with len(pts) elements
    - element: constructor of the FEM element class (callable)
    - quadrature class instance
    """
    cdef int n_nodes    = pts.shape[0]
    cdef int n_elements = elements.shape[0]
    cdef int n_quadrature_points = element.n_pts
    cdef int n_chi = element.n_chi
    cdef int row, col
    cdef double_t[:,:] chi    = element.chi
    cdef double_t[:] det_JxW  = element.detJxW
    cdef double_t[:,:,:] dphi = element.dphi_xy
    cdef np.ndarray[dtype=double_t, ndim=1, mode='c'] b
    cdef np.ndarray[dtype=double_t, ndim=2, mode='c'] nodes = np.zeros((n_chi, 3))

    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    # for each element...
    for k in xrange(n_elements):
        # initialize the current element
        for n in xrange(n_chi):
            nodes[n,:] = pts[elements[k,n],:]
        element.reset(nodes)

        det_JxW = element.detJxW
        dphi    = element.dphi_xy

        # for each shape function $\phi_i$...
        for i in xrange(n_chi):
            row = elements[k, i]

            # skip rows corresponding to Dirichlet nodes
            if bc_nodes[row]:
                continue

            # for each shape function $\phi_j$...
            for j in xrange(n_chi):
                col = elements[k, j]

                for q in xrange(n_quadrature_points):
                    # stiffness matrix:
                    A[row, col] += det_JxW[q] * (dphi[i,q,0]*dphi[j,q,0] +
                                                 dphi[i,q,1]*dphi[j,q,1] +
                                                 dphi[i,q,2]*dphi[j,q,2])

                    # right hand side:
                    b[row] += det_JxW[q] * chi[i,q] * (f[col]  * chi[j,q])

    # enforce Dirichlet boundary conditions:
    for k in xrange(n_nodes):
        if bc_nodes[k]:
            A[k,k] = 1.0 * scaling
            b[k] = u_bc[k] * scaling

    return A, b
