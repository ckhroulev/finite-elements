#!/usr/bin/env python
from py_distmesh2d import huniform, dcircle, drectangle, ddiff, distmesh2d, fixmesh
import numpy as np

# Triangular mesh (using distmesh2d):
bbox = [[-1, 1], [-1, 1]]

def tri_circle(hole=False):
    def circle(pts):
        return dcircle(pts, 0, 0, 1)

    def circle_with_a_hole(pts):
        return ddiff(dcircle(pts, 0, 0, 1), dcircle(pts, 0.5, 0, 0.4))

    if hole:
        pts, tri = distmesh2d(circle_with_a_hole, huniform, 0.1, bbox, [])
    else:
        pts, tri = distmesh2d(circle, huniform, 0.15, bbox, [])
    pts, tri = fixmesh(pts, tri)

    bc_nodes = np.zeros(pts.shape[0])
    for k,p in enumerate(pts):
        if np.fabs(circle(np.array([p]))) < 1e-6:
            bc_nodes[k] = 1

    return pts, tri, bc_nodes

def tri_ell():
    """L-shaped domain from 'Finite Elements and Fast Iterative Solvers' by Elman, Silvester, and Wathen."""
    pfix_ell = [[1,1], [1, -1], [0, -1], [0, 0], [-1, 0], [-1, 1]]
    def ell(pts):
        return ddiff(drectangle(pts, -1, 1, -1, 1), drectangle(pts, -2, 0, -2, 0))

    pts, tri = distmesh2d(ell, huniform, 0.1, bbox, pfix_ell)
    pts, tri = fixmesh(pts, tri)

    bc_nodes = np.zeros(pts.shape[0])
    for k,p in enumerate(pts):
        if np.fabs(ell(np.array([p]))) < 1e-6:
            bc_nodes[k] = 1

    return pts, tri, bc_nodes

def tri_square():
    """[-1, 1] * [-1, 1] square"""
    pfix_square = [[-1, -1], [1, -1], [1, 1], [-1, 1]]

    def square(pts):
        return drectangle(pts, -1, 1, -1, 1)

    pts, tri = distmesh2d(square, huniform, 0.2, bbox, pfix_square)
    pts, tri = fixmesh(pts, tri)

    bc_nodes = np.zeros(pts.shape[0])
    for k,p in enumerate(pts):
        if np.fabs(square(np.array([p]))) < 1e-6:
            bc_nodes[k] = 1

    return pts, tri, bc_nodes

# Quad mesh (by hand):
def quad_rectangle(x0, x1, y0, y1, Mx, My):
    """Creates a uniform rectangular Q1 mesh covering [x0,x1]*[y0,y1]."""

    x = np.linspace(x0, x1, Mx)
    y = np.linspace(y0, y1, My)

    xx,yy = np.meshgrid(x, y)

    def ii(i, j):
        return j*Mx + i

    quads = []
    for j in xrange(My-1):
        for i in xrange(Mx-1):
            quads.append([ii(i,j), ii(i+1,j), ii(i+1,j+1), ii(i,j+1)])

    # triangulation (for plotting)
    tri = []
    for j in xrange(My-1):
        for i in xrange(Mx-1):
            tri.append([ii(i,j), ii(i+1,j), ii(i+1,j+1)])
            tri.append([ii(i,j), ii(i+1,j+1), ii(i,j+1)])

    return np.vstack((xx.flatten(), yy.flatten())).T, np.array(quads), np.array(tri)

