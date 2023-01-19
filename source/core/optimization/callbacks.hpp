#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP
#include <functional>

/*
 * Generic function pointers
 */
template <typename T> struct meta_objcb {
    static_assert(std::is_floating_point<T>::value,
                  "Objective function arguments must be floating point");
    using type = std::function<void(int n, T *x, T *val, void *usrdata)>;
};
template <typename T> using objfun_t = typename meta_objcb<T>::type;
template <typename T> using objgrd_t = typename meta_objcb<T>::type;

#endif