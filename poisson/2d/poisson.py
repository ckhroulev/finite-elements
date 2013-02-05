#!/usr/bin/env python
from scipy.sparse import lil_matrix
import numpy as np

def poisson(pts, elements, bc_nodes, f, u_bc, element, quadrature):
    """Set up a Galerkin FEM system approximating the Poisson equation with Dirichlet B.C
    on a domain described by a mesh in (pts, elements).

    This code can be used with different element types and different quadratures.

    This code is not optimized to preserve the clear correspondence between
    mathematical expressions and their implementations.

    Parameters:
    - pts: list of node coordinates (2D array)
    - elements: list of lists, each element contains indices in pts of nodes that belong
      to a given quad or triangle
    - bc_nodes: 1D array of 0 and 1; bc_nodes[k] == 1 iff pts[k] is a Dirichlet B.C. node
    - f: forcing function: array with len(pts) elements
    - u_bc: Dirichlet boundary condition: array with len(pts) elements
    - element: constructor of the FEM element class (callable)
    - quadrature class instance
    """
    n_nodes    = pts.shape[0]
    n_elements = elements.shape[0]
    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    # Get quadrature points and weights
    q_pts = quadrature.points()
    q_w   = quadrature.weights()
    n_quadrature_points = len(q_pts)

    E = element()

    # for each element...
    for k in xrange(n_elements):
        # initialize the current element
        E.reset(pts[elements[k]])

        # compute the stiffness matrix:
        # for each shape function $\phi_i$...
        for i in xrange(E.n_chi()):
            row = elements[k, i]

            # for each shape function $\phi_j$...
            for j in xrange(E.n_chi()):
                col = elements[k, j]

                for q in xrange(n_quadrature_points):
                    # The following two lines use the chain rule to express
                    # derivatives with respect to x and y in terms of derivatives
                    # with respect to xi and eta..
                    dchi_i = E.J_inverse(q_pts[q]) * E.dchi(i, q_pts[q])
                    dchi_j = E.J_inverse(q_pts[q]) * E.dchi(j, q_pts[q])

                    A[row, col] += q_w[q] * E.det_J(q_pts[q]) * (dchi_i.T * dchi_j)

                    b[row] += q_w[q] * E.det_J(q_pts[q]) * E.chi(i, q_pts[q]) * (f[col] * E.chi(j, q_pts[q]))

    for k in xrange(n_nodes):
        if bc_nodes[k]:
            A[k,:] = 0.0
            A[k,k] = 1.0
            b[k] = u_bc[k]

    return A, b
