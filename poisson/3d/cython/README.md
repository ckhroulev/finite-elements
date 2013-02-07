Scripts solving the Dirichlet problem for the Poisson equation in 3D.

`Q1.pyx` computes values of element basis functions, derivatives of shape functions, and `det(J)` for `Q1` 3D elements with equal spacing in `x` and `y` directions. 

`poisson.pyx` assembles the linear system corresponding to a Dirichlet problem on a cube.
