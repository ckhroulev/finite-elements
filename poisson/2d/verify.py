#!/usr/bin/env python

from scipy.sparse.linalg import spsolve
import pylab as plt
import numpy as np
import FEM, create_mesh, plot_mesh
import poisson_optimized as poisson
import time

from test import set_bc_nodes_square

def f(pts):
    """zero forcing term"""
    return np.zeros(pts.shape[0])

def u_exact(pts):
    """exact solution (and the Dirichlet B.C.)"""
    try:
        x = pts[:,0]
        y = pts[:,1]
    except:
        x = pts[0]
        y = pts[1]

    return 2.0*(1+y) / ((3+x)**2 + (1+y)**2)


errors = []
Ns = np.array([5, 11, 21, 41, 81])
for n in Ns:
    print "Running with N = %d" % n

    tic = time.clock()
    pts,quads,tri = create_mesh.quad_rectangle(-1, 1, -1, 1, n, n)
    bc_nodes = set_bc_nodes_square(pts)
    toc = time.clock()
    print "Mesh generation took %f s" % (toc - tic)

    f_pts = f(pts)
    u_bc_pts = u_exact(pts)

    # Set up the system and solve:
    tic = time.clock()
    A,b = poisson.poisson(pts, quads, bc_nodes, f_pts, u_bc_pts, FEM.Q1, FEM.GaussQuad2x2())
    toc = time.clock()
    print "Matrix assembly took %f s" % (toc - tic)

    tic = time.clock()
    x = spsolve(A.tocsr(), b)
    toc = time.clock()
    print "Sparse solve took %f s\n" % (toc - tic)

    x_exact = u_exact(pts)

    errors.append(np.max(np.fabs(x - x_exact)))

dxs = 2.0 / np.array(Ns-1)

p = np.polyfit(np.log10(dxs), np.log10(errors), 1)
plt.hold(True)
plt.plot(np.log10(dxs), np.log10(errors), 'o-', color='black')
plt.plot(np.log10(dxs), np.polyval(p, np.log10(dxs)), '--', color='green',lw=2)
plt.axis('tight')
plt.title("Convergence rate: O(dx^%3.3f)" % p[0])
plt.xlabel("dx")
plt.ylabel("max error")
loc, _ = plt.yticks()
plt.yticks(loc, ["10^(%1.1f)" % x for x in loc])
plt.xticks(np.log10(dxs), ["%1.4f" % x for x in dxs])
plt.savefig("poisson-convergence.pdf")
