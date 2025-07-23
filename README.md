gulinalg 
========

Linear algebra functions as Generalized Ufuncs.


Uses ILP64 (64-bit) LAPACK from MKL or OpenBLAS, has optional OpenMP support
to parallelize the outer gufunc loop via the `workers` argument.

Build the package
--------------------

To build with MKL, do

```
$ pip install numpy meson meson-python ninja
$ pip install mkl mkl-devel
$ pip install . --no-build-isolation -Csetup-args='-Dopenmp=gnu' -Csetup-args='-Dblas=mkl'
```

To disable the OpenMP support, remove `-Csetup-args='-Dopenmp=gnu'` from the
pip invocation.

To build with OpenBLAS, install `scipy-openblas64` instead of MKL,

```
$ pip install scipy-openblas64   # instead of mkl mkl-devel
```

generate the `pkg-config` file,

```
$ python -c'import scipy_openblas64 as sc; print(sc.get_pkg_config())' > openblas.pc
$ export PKG_CONFIG_PATH=$PWD
```

and build the package

```
$ pip install . --no-build-isolation -Csetup-args='-Dopenmp=gnu' -Csetup-args='-Dblas=scipy-openblas64'
```

Test the package
----------------

```
$ python -P -c'import gulinalg as g; g.test(verbosity=2)'
```

or use the standard `pytest` invocations.


--------------------------------------------------------------------------------

NumPy/SciPy analogs
===================

This packages shares ancestry with `numpy.linalg`. In fact, about a half of
the `gulinalg` functionality is also available from `numpy.linalg`. We recommend
that existing users of `gulinalg` migrate to `numpy.linalg` using the following equivalence (NumPy functions below are from `np.linalg` namespace unless
explicitly namespaced with `np.`)

| `gulinalg`              | `np.linalg`                       |
|:------------------------|:----------------------------------|
| `matrix_multiply(a, b)` | `a @ b`, `np.matmul(a, b)`        |
| `matvec_multiply(a, b)` | `np.matvec(a, b)` (`numpy >= 2.2.0`) |
| `inner1d(a, b)`         | `vecdot(a.conj(), b)`             |
| `dotc1d(a, b)`          | `vecdot(a, b)`                    |
| `cholesky(a, b)`        | `cholesky(a, b)`                  |
| `qr(a)`                 |  `qr(a)`                          |
| `svd(a)`                |  `svd(a)`                         |
| `solve(a, b)`           | `solve(a, b)`                     |
| `inv(a, b)`             |  `inv(a, b)`                      |
| `det(a, b)`             |  `det(a, b)`                      |
| `slogdet(a, b)`         |  `slogdet(a, b)`                  |
| `eig(a, b)`             |  `eig(a, b)`                      |
| `eigh(a, b)`            |  `eigh(a, b)`                     |
| `eigvals(a, b)`         |  `eigvals(a, b)`                  |
| `eigvalsh(a, b)`        |  `eigvalsh(a, b)`                 |



A few things to keep in mind when porting from `gulinalg` to `np.linalg`:

1. `np.linalg.vecdot` complex conjugates its first argument. Therefore for
    complex-valued arguments it is equivalent to `gulinalg.dotc1d`, and for
    real-valued arguments it is equivalent to `gulinalg.inner1d`.
2. For matrix factorization, `gulinalg` functions return tuples of arrays, while
   `np.linalg` analogs return namedtuples.
3. `np.matvec` and `np.vecmat` functions are new in NumPy version 2.2.0.
4. For matrix factorizationsWhere the result is not unique (e.g., the
  `QR`factorization is unique only up to a matching permutation of the rows of
   the `Q` matrix and the columns of the `R` matrix),
  `np.linalg` and `gulinalg` functions may return different, if equivalent, results.


Several `gulinalg` functions have no direct NumPy equivalents but are simple
to replicate manually, at least for 1D arguments:

| `gulinalg`                | `np.linalg`                  |
|:--------------------------|:-----------------------------|
| `update_rank1(a, b, c)`   | `np.outer(a, b) + c`         |
| `innerwt(a, b, c)`        | `np.linalg.vecdot(a * b, c)` |
| `quadratic_form(a, b, c`) | `a.T @ b @ c`                |

Not all of `gulinalg` is available from NumPy or is easy to replicate in pure Python. In these cases,
equivalent functionality is available from SciPy starting from version 1.17, with slight API
differences. Namely,

| `gulinalg`                         | `scipy.linalg`                            |
|:-----------------------------------|:------------------------------------------|
| `poinv(a)`                         | `inv(a, assume_a="pos"`                   |
| `chosolve(a, b)`                   | `solve(a, b, assume_a="pos"`              |
| `solve_triangular(a, b, UPLO="U")` | `solve(a, b, assume_a="upper triangular"` |

Also note that starting from version 1.17, `scipy.linalg` functions internally attempt at detecting
the matrix structure and pick an appropriate algorithm. For instance, `solve(a, b)` with `a`
being an upper triangular matrix will automatically select a triangular solve algorithm. Supplying
an explicit `assume_a` argument bypasses the structure detection. Strictly speaking, structure
detection is not free, thus bypassing it may improve performance. Whether the improvement is substantial
strongly depends on the matrix size, structure and the depth of the batch, and this should be
evaluated on a case-by-case basis.


--------------------------------------------------------------------------------


Below is the documentation for the older version of the package (v0.1.6).
=========================================================================

This version is using `numpy.distutils`, and is therefore limited to
python versions `<= 3.11` and compatible NumPy versions.



Notes about building
====================

This module is built using NumPy's configuration for LAPACK. This means that
you need a setup similar to the one used to build the NumPy you are using. If
you are building your own version of NumPy that should be the case.

OpenMP support
==============

A subset of functions currently have openMP support via a `workers` argument
that can be used to set the number of threads to use in the outer gufunc loop.

On windows MSVC-style flags will be set, otherwise GCC-style flags (-fopenmp)
are set. By default OpenMP is enabled, but if compilation of a simple test
function fails, OpenMP will be disabled,

The user can force OpenMP to always be disabled if desired by defining the
environment variable GULINALG_DISABLE_OPENMP.

On linux, linking against intel's OpenMP implementation instead of the GNU
implementation can be selected by defining GULINALG_INTEL_OPENMP. This will
cause libiomp5 and libpthread to be linked during compilation (instead of GCC's
libgomp). This should be done, for example, on MKL-based conda environments
where the intel-openmp package has been installed. For OpenBLAS-based conda
environments, the GULINALG_INTEL_OPENMP variable should not be defined.

If Intel's icc compiler is being used instead of gcc, the user should define
the GULINALG_USING_ICC environment variable. Use of icc on windows systems is
not currently supported.

Build Status
============

Travis CI: [![Build Status](https://travis-ci.org/Quansight/gulinalg.svg?branch=master)](https://travis-ci.org/Quansight/gulinalg)
