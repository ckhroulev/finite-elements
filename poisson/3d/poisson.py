#!/usr/bin/env python
from scipy.sparse import lil_matrix
import numpy as np

def poisson(pts, elements, bc_nodes, f, u_bc, element, scaling=1.0):
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
    n_nodes    = pts.shape[0]
    n_elements = elements.shape[0]
    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    n_quadrature_points = element.n_quadrature_points()
    n_chi = element.n_chi()
    chi = element.chi

    # for each element...
    for k in xrange(n_elements):
        # initialize the current element
        element.reset(pts[elements[k]])
        det_JxW = element.detJxW
        dphi = element.dphi_xy

        # for each shape function $\phi_i$...
        for i in xrange(n_chi):
            row = elements[k, i]

            if bc_nodes[row]:
                continue

            # for each shape function $\phi_j$...
            for j in xrange(n_chi):
                col = elements[k, j]

                for q in xrange(n_quadrature_points):
                    # stiffness matrix:
                    A[row, col] += det_JxW[q] * (dphi[i,q,0]*dphi[j,q,0] + dphi[i,q,1]*dphi[j,q,1] + dphi[i,q,2]*dphi[j,q,2])

                    # right hand side:
                    b[row] += det_JxW[q] * chi[i,q] * (f[col]  * chi[j,q])

    # enforce Dirichlet boundary conditions:
    for k in xrange(n_nodes):
        if bc_nodes[k]:
            A[k,k] = 1.0 * scaling
            b[k] = u_bc[k] * scaling

    return A, b
