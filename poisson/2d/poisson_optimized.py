#!/usr/bin/env python
from scipy.sparse import lil_matrix
import numpy as np

def poisson(pts, elements, bc_nodes, f, u_bc, element, quadrature, scaling=1.0):
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

    # Get quadrature points and weights
    q_pts = quadrature.points()
    q_w   = quadrature.weights()
    n_quadrature_points = len(q_pts)

    E = element()
    # pre-compute values of shape functions at all quadrature points:
    chi = np.zeros((n_quadrature_points, E.n_chi()))
    for j in xrange(E.n_chi()):
        for q in xrange(n_quadrature_points):
            chi[j, q] = E.chi(j, q_pts[q])

    # pre-compute values of derivatives of shape functions at all quadrature points:
    dchi = np.zeros((n_quadrature_points, E.n_chi(), 2))
    for q in xrange(n_quadrature_points):
        for j in xrange(E.n_chi()):
            dchi[q,j,:] = E.dchi(j, q_pts[q]).T

    # for each element...
    for k in xrange(n_elements):
        # initialize the current element
        E.reset(pts[elements[k]])

        # compute dchi with respect to x,y (using chain rule)
        # and det(J)*w at all quadrature points:
        dchi_xy = np.zeros((E.n_chi(), n_quadrature_points,  2))
        det_JxW = np.zeros(n_quadrature_points)
        for q in xrange(n_quadrature_points):
            J_inv = E.J_inverse(q_pts[q])
            for j in xrange(E.n_chi()):
                dchi_xy[j,q,:] = (J_inv * np.matrix(dchi[q,j,:]).T).T

            det_JxW[q] = E.det_J(q_pts[q]) * q_w[q]

        # for each shape function $\phi_i$...
        for i in xrange(E.n_chi()):
            row = elements[k, i]

            # for each shape function $\phi_j$...
            for j in xrange(E.n_chi()):
                col = elements[k, j]

                for q in xrange(n_quadrature_points):
                    # stiffness matrix:
                    A[row, col] += det_JxW[q] * np.dot(dchi_xy[i,q], dchi_xy[j,q])

                    # right hand side:
                    b[row] += det_JxW[q] * chi[i,q] * (f[col]  * chi[q,j])

    # enforce Dirichlet boundary conditions:
    for k in xrange(n_nodes):
        if bc_nodes[k]:
            A[k,:] = 0.0
            A[k,k] = 1.0 * scaling
            b[k] = u_bc[k] * scaling

    return A, b
