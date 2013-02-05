from pylab import figure, plot, hold, triplot, tripcolor, tricontour, clabel, axis, axes, show, text, spy, colorbar, xticks, yticks
import numpy as np

def plot_tri_mesh(pts, tri):
    triplot(pts[:,0], pts[:,1], tri, "k-", lw=2)

    if len(pts) < 200:
        for k,p in enumerate(pts):
            text(p[0], p[1], "%d" % k, color="black",
                 fontsize=10,
                 bbox=dict(boxstyle = "round", fc = "white"),
                 horizontalalignment='center', verticalalignment='center')

    if len(tri) < 400:
        for k in xrange(tri.shape[0]):
            center = pts[tri[k]].sum(axis=0) / 3.0

            text(center[0], center[1], "%d" % k, color="black",
                 fontsize=8,
                 horizontalalignment='center', verticalalignment='center')

def plot_quad_mesh(pts, quads):
    """Plot a quadrilateral mesh."""

    def plot_quad(pts):
        """Plot one quad."""
        plot(np.r_[pts[:,0], pts[0,0]],
             np.r_[pts[:,1], pts[0,1]],
             lw=1.5, color="black")

    hold(True)
    for k,q in enumerate(quads):
        plot_quad(pts[q])

    if len(quads) < 400:
        for k,q in enumerate(quads):
            center = pts[q].sum(axis=0) / 4.0
            text(center[0], center[1], "%d" % k, color="black",
                 fontsize=10,
                 horizontalalignment='center', verticalalignment='center')

    if len(pts) < 200:
        for k,p in enumerate(pts):
            text(p[0], p[1], "%d" % k, color="black",
                 fontsize=10,
                 bbox=dict(boxstyle = "round", fc = "white"),
                 horizontalalignment='center', verticalalignment='center')

def plot_mesh(pts, tri, z=None):
    if z is not None:
        x = pts[:,0]
        y = pts[:,1]
        tripcolor(x, y, tri, z)
        colorbar()
        cs = tricontour(x, y, tri, z, colors="black", linestyles="dashed")
        clabel(cs)
    else:
        if len(tri[0]) == 3:
            plot_tri_mesh(pts, tri)
        else:
            plot_quad_mesh(pts, tri)

    axis('tight')
    xticks([])
    yticks([])
    axes().set_aspect('equal')
