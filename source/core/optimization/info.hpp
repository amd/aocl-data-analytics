#ifndef OPTIMIZATION_INFO_H
#define OPTIMIZATION_INFO_H

namespace optim {
using namespace optim;

// Elements in the vector info<T> containing information from the solver
enum info_t {
    info_objective = 0,
    info_grad_norm = 1,
    info_iter = 2,
    info_number // leave last
};              // TODO fill in
} // namespace optim

#endif // OPTIMIZATION_INFO_H