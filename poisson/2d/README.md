This directory contains scripts solving the Dirichlet problem for the Poisson in 2 dimensions.

  - `test.py` assembles and solves the system corresponding to `$\nabla u = 1$` on a square with zero Dirichlet B.C. on 2 sides and 0 Neumann ("natural") B.C. on the other two sides.
  - `verify.py` plots the convergence graph for the Laplace equation on a square with Dirichlet B.C. coming from an exact solution (on all 4 sides). (**Note that it uses the wrong (`$\ell^{\infty}$`) norm to compute errors.**)
  - `FEM.py` implements `Q1` and `P1` finite elements in 2D, as well as some quadratures for use with these.
  - `poisson.py` is a naive implementation of the matrix assembly. It is very slow, but should be easy to understand.
  - `poisson_optimized.py` is a much faster matrix assembly code functionally equivalent to `poisson.py`. It moves computations out of inner loops as much as possible. It should be relatively easy to understand, too.

Both `poisson.py` and `poisson_optimized.py` can be used with various element and quadrature types. (It would be very easy to add `P2` or `Q2` elements, or a 3x3 quadrature for the square.
