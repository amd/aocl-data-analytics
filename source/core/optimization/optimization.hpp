#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include "callbacks.hpp"
#include "lbfgsb_driver.hpp"
#include <functional>
#include <iostream>
#include <vector>

/* all constraint types
 * mainly used to check wether a specific type of constraint is defined in a bool array
 */
enum constraint_type { cons_bounds = 0, cons_number };

enum opt_solvers { solver_undefined = 0, solver_lbfgsb };

enum opt_status {
    opt_status_success = 0,
    opt_status_memory_err,
    opt_status_invalid_input,
    opt_status_not_implemented
};

template <typename T> class da_optimization {
  private:
    int n = 0;

    // which constraints are defined, indexed by the enum constraint_type
    std::vector<bool> def_constraints;

    opt_solvers solver = solver_undefined;

    // bound constraints; allocated if def_constraints[cons_bound] == true
    std::vector<T> l, u;

    // pointers to callbacks
    objfun_t<T> objfun;
    objgrd_t<T> objgrd;

    // Internal solvers data
    lbfgsb_data<T> *soldata_lbfgsb = nullptr;

    // solutions
    T f;
    std::vector<T> g;

  public:
    da_optimization() : def_constraints(cons_number, false){};
    ~da_optimization();
    opt_status declare_vars(int n);
    opt_status add_bound_const(std::vector<T> &l, std::vector<T> &u);
    opt_status
    user_objective(std::function<void(int n, T *x, T *f, void *usrdata)> usrfun);
    opt_status
    user_gradient(std::function<void(int n, T *x, T *grad, void *usrdata)> usrgrd);
    opt_status select_solver(opt_solvers sol);

    // solver interfaces (only lbfgsb for now)
    opt_status solve(std::vector<T> &x, void *usrdata);
};

template <typename T> da_optimization<T>::~da_optimization() {
    if (soldata_lbfgsb) {
        free_lbfgsb_data(&soldata_lbfgsb);
    }
}

template <typename T> opt_status da_optimization<T>::declare_vars(int n) {
    if (this->n != 0)
        return opt_status_invalid_input;

    this->n = n;

    return opt_status_success;
}

template <typename T>
opt_status da_optimization<T>::add_bound_const(std::vector<T> &l, std::vector<T> &u) {
    if (l.size() != n || u.size() != n)
        return opt_status_invalid_input;

    this->def_constraints[cons_bounds] = true;

    this->l = std::vector<T>(l);
    this->u = std::vector<T>(u);

    return opt_status_success;
}

template <typename T>
opt_status da_optimization<T>::user_objective(
    std::function<void(int n, T *x, T *f, void *usrdata)> usrfun) {
    if (objfun != nullptr) {
        std::cout << "Objective function was already defined. Exit" << std::endl;
        return opt_status_invalid_input;
    }

    objfun = usrfun;
    return opt_status_success;
}

template <typename T>
opt_status da_optimization<T>::user_gradient(
    std::function<void(int n, T *x, T *grad, void *usrdata)> usrgrd) {
    if (objgrd != nullptr) {
        std::cout << "Gradient function was already defined. Exit" << std::endl;
        return opt_status_invalid_input;
    }

    objgrd = usrgrd;
    return opt_status_success;
}

template <typename T> opt_status da_optimization<T>::select_solver(opt_solvers sol) {

    switch (sol) {
    case solver_lbfgsb:
        // Allocate internal memory for lbfgsb
        soldata_lbfgsb = new lbfgsb_data<T>;
        // derivative based solver, need to allocate gradient memory
        g.resize(n);
        // TODO m and bigbnd need to be options
        init_lbfgsb_data<T>(soldata_lbfgsb, n, 2, (T)1.0e20, l, u);
        break;

    default:
        break;
    }
    solver = sol;

    return opt_status_success;
}

template <typename T>
opt_status da_optimization<T>::solve(std::vector<T> &x, void *usrdata) {

    T f;

    switch (solver) {
    case solver_lbfgsb:
        lbfgsb_fc(soldata_lbfgsb, n, x.data(), l.data(), u.data(), &f, g.data(), objfun,
                  objgrd, usrdata);
        break;

    default:
        break;
    }
    return opt_status_success;
}

#endif