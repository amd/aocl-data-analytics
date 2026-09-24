# AOCL Data Analytics — Experimental R Interface

> **Experimental.** This R interface is an early-stage implementation and is not yet
> part of the official AOCL-DA release. APIs may change without notice.
> Currently developed for Linux only and doesn't support Windows.
> Requires AOCL-DA 6.0.

AMD's AOCL Data Analytics Library provides optimized building blocks for data analysis and classical machine learning applications.
The package leverages the [AMD Optimizing CPU Libraries (AOCL)](https://www.amd.com/en/developer/aocl.html) to provide outstanding performance, not just on AMD processors but on other x86 hardware too.

The AOCL Data Analytics R interface provides experimental R APIs giving access to all algorithms in the library, including approximate nearest neighbors, basic statistics, cubic spline interpolation, DBSCAN, decision forests, decision trees, k-means, kernel functions, kernel PCA, linear models, logistic regression, nearest neighbors, nonlinear least squares, nonnegative matrix factorization, PCA, SVMs (SVC, SVR, NuSVC, NuSVR), t-SNE, train/test splitting, and more.

## Installation

The R package build is driven by CMake. AOCL-DA must first be built as a shared library (`-DBUILD_SHARED_LIBS=On`) using the LP64 (32-bit integer) variant. Refer to the top-level `README.md` for full build instructions.

### Build

```bash
cd r_interface
cmake -B build
cmake --build build --target rpackage
R CMD INSTALL build/aoclda_*_R_x86_64-pc-linux-gnu.tar.gz
```

To point at a custom AOCL-DA install location:

```bash
cmake -B build -DDA_LIB_DIR=/path/to/lib/LP64 -DDA_INCLUDE_DIR=/path/to/include/LP64
```

Prerequisites: `R`, `R-devel`, `patchelf`, and R packages `Rcpp` (>= 1.1.2) and `R6`.

### Linear algebra libraries and performance

> **Warning — BLAS/LAPACK symbol clashes.** R ships with, and links against, its own
> BLAS and LAPACK. The AOCL-DA package bundles AOCL's optimized BLIS and libFLAME, which
> export the same BLAS/LAPACK symbols. With both loaded in a single process, the dynamic
> linker may bind calls to R's slower reference BLAS instead of AOCL's — degrading
> performance and, in rare cases, causing symbol conflicts.
>
> For best performance, preload the AOCL BLIS (and libFLAME) libraries with `LD_PRELOAD`
> so all BLAS/LAPACK calls resolve to the optimized AOCL implementations:
>
> ```bash
> LD_PRELOAD=/path/to/aocl/lib/libblis-mt.so:/path/to/aocl/lib/libflame.so R
> ```

### Documentation

To regenerate the man pages from roxygen2 comments:

```bash
cmake --build build --target rdocs
```

This requires the `roxygen2` R package. The generated `.Rd` files are written to `man/`.

To view the documentation as plain text, HTML, or PDF:

```bash
# Plain text
Rscript -e "tools::Rd2txt('man/nearest_neighbors.Rd')"

# Single page HTML
Rscript -e "tools::Rd2HTML('man/nearest_neighbors.Rd', out='nearest_neighbors.html')"

# Full PDF reference manual (requires LaTeX)
R CMD Rd2pdf --no-preview --output=aoclda-manual.pdf .
```

## Requirements

| R Package    | Purpose               | Minimum version |
|--------------|-----------------------|-----------------|
| `Rcpp`       | Build (C++ bindings)  | 1.1.2           |
| `R6`         | Build (OOP classes)   | —               |
| `testthat`   | Testing               | 3.0.0           |
| `rmarkdown`  | Rendering examples    | —               |
| `knitr`      | Rendering examples    | —               |
| `roxygen2`   | Generating docs       | —               |

## R APIs

The package comes with numerous example scripts in the `examples/` directory. To render an example:

```bash
Rscript -e "rmarkdown::render('examples/pca_ex.Rmd')"
```

A quick usage example:

```r
library(aoclda)

X <- matrix(rnorm(500), nrow = 100, ncol = 5)

# k-means clustering
km <- kmeans$new(n_clusters = 3, seed = 42)
km$fit(X)
labels <- km$labels

# PCA
pca <- PCA$new(n_components = 2)
pca$fit(X)
scores <- pca$transform(X)
```

## License

Copyrighted code in AOCL Data Analytics is subject to the licenses set forth in the source code file headers of such code. See the top-level [License](../License) file for full details.

## Further information

Full documentation of the C++ and Python APIs can be found in the [AMD technical information portal](https://docs.amd.com/r/en-US/68552-AOCL-api-guide/AOCL-Data-Analytics).

The AOCL Data Analytics source code and compilation instructions are available at https://github.com/amd/aocl-data-analytics/.

AOCL-DA is developed and maintained by [AMD](https://www.amd.com/). For support or queries, you can email us on
[toolchainsupport@amd.com](mailto:toolchainsupport@amd.com).
