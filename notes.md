
# I describe your problem back to you to see if I understand

A basic gravity model tries to explain trade between nations as a function of other things, like distance, common language, and so on.
The effect of these explanatory variables is modeled parametrically using a Generalized Linear Model.
The `GME` package provides a python module for estimating the parameters of a gravity model data.
The estimation is done by solving for the Poisson pseudo-maximum-likelihood estimator (PPML). 
The `statsmodels` package is used to solve the PPML problem.
By default, `statsmodels` uses iterative reweighted least squares (IRLS) to solve the PPML problem.

I do not work with economic models, but IRLS looks broadly similar to a solution method known as a Picard iteration method for solving nonlinear PDEs (see for example [doi:10.1029/94WR02046](https://doi.org/10.1029/94WR02046)).
Picard iteration methods converge linearly but are usually stable.
If the function is well behaved in the vicinity of the solution, Newton's method converges quadratically, but can be unstable far from the solution.

The majority of the work in each iteration of IRLS is setting up and solving a weighted linear least squares problem.
The least squares problem is dense: there are `mn` exogenous variables, where `m` is the number of endogenous variables and `n` is the number of explanatory variables plus "fixed effect" variables for the importer and exporter in each pair.
Internally, `statsmodels` uses `numpy.linalg.lstsq`, which in turn uses LAPACK's `dgelsy`.  This has a complexity of `O(mn^2)`.
For a problem provided to me described as a medium size problem, `m = 179,646` and `n = 476`.
I tried to run this problem on my laptop, but it only has 8 GB of RAM, so I made the problem size smaller by only keeping three years, in which case `m = 89,731` and `n = 473`.
The IRLS algorithm converged in 17 iterations and took about two minutes.

Hundreds of thousands of such estimations need to be run, but I do not know how these different instances are related.
Are they independent?
Are they already running in parallel?

# A brief description of aspects of PETSc that should be relevant to this problem

- PETSc provides interfaces and implementations for data structures and algorithms for solving equations.
- "Data structures": PETSc implements many different matrix formats.  **I suspect the most egregious performance issue in `GME` is the use of a dense matrix format for the matrix of exogenous variables, which is very sparse.**  PETSc has good sparse matrix implementations, and algorithms that can exploit the efficiency of sparsity (for example, using sparse direct factorization and/or iterative methods to solve the normal equations).
- "Solving equations": PETSc has objects that handle optimization problems, nonlinear systems of equations, and linear systems of equations.  This means that PETSc could take over the estimation at various levels of granularity.  It could be asked to solve the likelihood maximization problem (allowing for additional regularization, for example, or for problems with inequality constraints), the PPML nonlinear equations (using algorithms that could potentially handle the nonlinearity better than IRLS), or just the least squares problems.
- The "E" in PETSc stands for "extensible": it can use outside libraries as plugins (`suitesparse` or `Elemental` would be relevant to `GME`).  It also emphasizes "composable solvers" through preconditioning for better performance.
- It is callable and extensible from python using `petsc4py`
- It supports several kinds of parallelism (for using multiple computational resources to solve a single estimation problem faster): it can use multiple CPUs or GPUs to accelerate solvers.
