#!/usr/bin/env python

from scipy.sparse.linalg import spsolve
import pylab as plt
import numpy as np
import Q1
import time

def f(pts):
    """The forcing term"""
    return np.ones(pts.shape[0])

def u_bc(pts):
    """Dirichlet boundary conditions"""
    return np.zeros(pts.shape[0])

def hex_cube(x0, x1, y0, y1, z0, z1, Mx, My, Mz):
    """Creates a uniform hexahedral Q1 mesh covering [x0,x1]*[y0,y1]*[z0,z1]."""

    x = np.linspace(x0, x1, Mx)
    y = np.linspace(y0, y1, My)
    z = np.linspace(z0, z1, Mz)

    def ii(i, j, k):
        return (Mx*My)*k + (j*Mx + i)

    pts = np.zeros((Mx*My*Mz, 3), dtype='f8')
    for k in xrange(Mz):
        for j in xrange(My):
            for i in xrange(Mx):
                pts[ii(i,j,k), :] = (x[i], y[j], z[k])

    hexes = np.zeros(((Mx-1)*(My-1)*(Mz-1), 8), dtype='i')
    n = 0
    for k in xrange(Mz-1):
        for j in xrange(My-1):
            for i in xrange(Mx-1):
                hexes[n] = [ii(i,j,k), ii(i+1,j,k), ii(i+1,j+1,k), ii(i,j+1,k),
                            ii(i,j,k+1), ii(i+1,j,k+1), ii(i+1,j+1,k+1), ii(i,j+1,k+1)]
                n += 1

    return x, y, z, pts, hexes

def set_bc_nodes_cube(pts):
    """Sets up B.C. nodes in the [-1,1]x[-1,1]x[-1,1] cube case."""
    n_nodes = pts.shape[0]
    bc_nodes = np.zeros(n_nodes, dtype='i')

    eps = 1e-6
    for k in xrange(n_nodes):
        x,y,z = pts[k]
        if (np.fabs(x - 1) < eps or
            np.fabs(x + 1) < eps or
            np.fabs(y - 1) < eps or
            np.fabs(y + 1) < eps or
            np.fabs(z - 1) < eps or
            np.fabs(z + 1) < eps):
            bc_nodes[k] = True

    return bc_nodes

if __name__ == "__main__":
    # Try the quad mesh with Q1 elements:
    import poisson
    N = 11
    Mx = N
    My = N
    Mz = N

    tic = time.clock()
    x, y, z, pts, hexes = hex_cube(0, 1, 0, 1, 0, 1, Mx, My, Mz)
    bc_nodes = set_bc_nodes_cube(pts)
    toc = time.clock()
    print "Mesh generation took %f s" % (toc - tic)

    # forcing term at mesh nodes
    f_pts = f(pts)
    # Dirichlet boundary conditions at *all* mesh nodes (most of these values are not used)
    u_bc_pts = u_bc(pts)

    # Set up the system and solve:
    tic = time.clock()
    E = Q1.Q13DEquallySpaced(Q1.Gauss2x2x2())
    A, b = poisson.poisson(pts, hexes, bc_nodes, f_pts, u_bc_pts, E, scaling=1.0)
    toc = time.clock()
    print "Matrix assembly took %f s" % (toc - tic)

    tic = time.clock()
    u = spsolve(A.tocsr(), b)
    toc = time.clock()
    print "Sparse solve took %f s" % (toc - tic)

    Z = u.reshape((Mz, My, Mx))

    # Plot:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.colors

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    norm = matplotlib.colors.Normalize(vmin=Z.min(), vmax=Z.max())
    clevels = np.linspace(Z.min(), Z.max(), 21)
    for k,level in enumerate(z):
        ax.contourf(x, y, Z[k], zdir='z', levels=clevels, offset=level, cmap=cm.RdBu, alpha=0.5, norm=norm)
    ax.set_zlim(z.min(), z.max())
    plt.show()
