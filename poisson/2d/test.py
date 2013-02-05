#!/usr/bin/env python

from scipy.sparse.linalg import spsolve
import pylab as plt
import numpy as np
import FEM, create_mesh, plot_mesh
import time

def f(pts):
    """The forcing term"""
    return np.ones(pts.shape[0])

def u_bc(pts):
    """Dirichlet boundary conditions"""
    return np.zeros(pts.shape[0])

def set_bc_nodes_square(pts):
    """Sets up B.C. nodes in the [-1,1]x[-1,1] square case."""
    n_nodes = pts.shape[0]
    bc_nodes = np.zeros(n_nodes)

    eps = 1e-6
    for k in xrange(n_nodes):
        x,y = pts[k]
        if (np.fabs(x - 1) < eps or #np.fabs(x - (-1)) < eps or
            np.fabs(y - 1) < eps):# or np.fabs(y - (-1)) < eps):
            bc_nodes[k] = 1

    return bc_nodes

if __name__ == "__main__":
    # Try the quad mesh with Q1 elements:
    if True:
        import poisson_optimized as poisson
        # import poisson
        pts, quads, tri = create_mesh.quad_rectangle(0, 1, 0, 1, 21, 21)
        bc_nodes = set_bc_nodes_square(pts)

        # forcing term at mesh nodes
        f_pts = f(pts)
        # Dirichlet boundary conditions at *all* mesh nodes (most of these values are not used)
        u_bc_pts = u_bc(pts)

        # Set up the system and solve:
        tic = time.clock()
        A, b = poisson.poisson(pts, quads, bc_nodes, f_pts, u_bc_pts, FEM.Q1Aligned, FEM.GaussQuad2x2())
        toc = time.clock()
        print "Matrix assembly took %f s" % (toc - tic)

        if pts.shape[0] < 50:
            print "cond(A) = %3.3f" % np.linalg.cond(A.todense())

        tic = time.clock()
        x = spsolve(A.tocsr(), b)
        toc = time.clock()
        print "Sparse solve took %f s" % (toc - tic)

        # Plot:
        plt.figure()
        plot_mesh.plot_mesh(pts, quads)
        plt.figure()
        plot_mesh.plot_mesh(pts, tri, x)

    # Try the triangular mesh with P1 elements:
    if False:
        import poisson_optimized as poisson
        pts, tri, bc_nodes = create_mesh.tri_circle()

        # forcing term at mesh nodes
        f_pts = f(pts)
        # Dirichlet boundary conditions at *all* mesh nodes (most of these values are not used)
        u_bc_pts = u_bc(pts)

        tic = time.clock()
        A, b = poisson.poisson(pts, tri, bc_nodes, f_pts, u_bc_pts, FEM.P1, FEM.GaussTri1())
        toc = time.clock()
        print "Matrix assembly took %f s" % (toc - tic)

        print "cond(A) = %3.3f" % np.linalg.cond(A.todense())

        tic = time.clock()
        x = spsolve(A.tocsr(), b)
        toc = time.clock()
        print "Sparse solve took %f s" % (toc - tic)

        # Plot:
        plt.figure()
        plot_mesh.plot_mesh(pts, tri)
        plt.figure()
        plot_mesh.plot_mesh(pts, tri, x)

    # Triangular mesh covering the L-shaped domain
    if False:
        import poisson_optimized as poisson
        pts, tri, bc_nodes = create_mesh.tri_ell()

        # forcing term at mesh nodes
        f_pts = f(pts)
        # Dirichlet boundary conditions at *all* mesh nodes (most of these values are not used)
        u_bc_pts = u_bc(pts)

        tic = time.clock()
        A, b = poisson.poisson(pts, tri, bc_nodes, f_pts, u_bc_pts, FEM.P1, FEM.GaussTri1())
        toc = time.clock()
        print "Matrix assembly took %f s" % (toc - tic)

        print "cond(A) = %3.3f" % np.linalg.cond(A.todense())

        tic = time.clock()
        x = spsolve(A.tocsr(), b)
        toc = time.clock()
        print "Sparse solve took %f s" % (toc - tic)

        # Plot:
        plt.figure()
        plot_mesh.plot_mesh(pts, tri)
        plt.figure()
        plot_mesh.plot_mesh(pts, tri, x)

    plt.show()
