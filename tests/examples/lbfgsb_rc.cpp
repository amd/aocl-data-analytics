#include "lbfgsb_driver.hpp"
#include <cmath>
#include <iostream>

int main() {
    int n = 2, m = 2, iprint = 100;
    double factr = 1.0e07, pgtol = 1.0e-05;

    int itask, icsave, lsavei[4], isave[44];
    double f, dsave[29];
    int *nbd = nullptr, *iwa = nullptr;
    double *x = nullptr, *l = nullptr, *u = nullptr, *g = nullptr, *wa = nullptr;

    int i;

    nbd = (int *)malloc(n * sizeof(int));
    x = (double *)malloc(n * sizeof(double));
    g = (double *)malloc(n * sizeof(double));
    l = (double *)malloc(n * sizeof(double));
    u = (double *)malloc(n * sizeof(double));
    iwa = (int *)malloc(3 * n * sizeof(int));
    int nwa = 2 * m * n + 5 * n + 11 * m * m + 8 * m;
    wa = (double *)malloc(nwa * sizeof(double));

    // Set bounds
    for (i = 0; i < n; i += 2) {
        nbd[i] = 2;
        l[i] = 1.0;
        u[i] = 100.0;
    }
    for (i = 1; i < n; i += 2) {
        nbd[i] = 2;
        l[i] = -100.0;
        u[i] = 100.0;
    }

    // Starting point
    for (i = 0; i < n; i++)
        x[i] = 3.0;

    std::cout << "Solving sample problem." << std::endl
              << "(f = 0.0 at the optimal solution.)" << std::endl;

    itask = 2;

    /* task = 'START => itask = 2
     * TASK = 'NEW_X' => itask = 1
     */
    bool compute_fg = true;
    while (itask == 2 || itask == 1 || compute_fg) {
        setulb_d_(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, &itask, &iprint,
                  &icsave, lsavei, isave, dsave);
        compute_fg = itask == 4 ||  // 'FG'
                     itask == 21 || // 'FG_START'
                     itask == 20;   // 'FG_LNSRCH
        if (compute_fg) {
            f = pow(1.0 - x[0], 2.0);
            f += 100.0 * pow(x[1] - x[0] * x[0], 2.0);

            // compute gradient
            g[0] = 2 * (x[0] - 1.0) - 400.0 * (x[1] - x[0] * x[0]) * x[0];
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
        }
    }
    std::cout << std::endl;

    std::cout << "Final solution, f = " << f << std::endl;
    std::cout << "x = [" << x[0] << ", " << x[1] << "]" << std::endl;

    if (nbd)
        free(nbd);
    if (iwa)
        free(iwa);
    if (wa)
        free(wa);
    if (x)
        free(x);
    if (l)
        free(l);
    if (u)
        free(u);
    if (g)
        free(g);
}