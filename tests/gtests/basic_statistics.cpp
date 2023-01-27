#include "aoclda.h"
#include "da_mean.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <list>

template <typename T> class FPTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FPTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(FPTest, FPTypes);

TYPED_TEST(FPTest, mean) {
    da_int n = 4;
    TypeParam *x = nullptr;

    x = (TypeParam *)malloc(n * sizeof(TypeParam));

    // Starting point
    for (da_int i = 0; i < n; i++)
        x[i] = (TypeParam)i;

    TypeParam mean;

    // int err = 1;
    da_status err = da_mean(n, x, 1, &mean);

    if (err == da_status_success) {
        std::cout << "mean = " << mean << std::endl;
    } else {
        std::cout << "error computing mean" << std::endl;
    }

    if (x)
        free(x);

    ASSERT_EQ(1.5, mean);
}

TYPED_TEST(FPTest, mean2) {
    da_int n = 4;
    TypeParam *x = nullptr;

    x = (TypeParam *)malloc(n * sizeof(TypeParam));

    // Starting point
    for (da_int i = 0; i < n; i++)
        x[i] = 0.0;

    TypeParam mean;

    // int err = 1;
    da_status err = da_mean(n, x, 1, &mean);

    if (err == da_status_success) {
        std::cout << "mean = " << mean << std::endl;
    } else {
        std::cout << "error computing mean" << std::endl;
    }

    if (x)
        free(x);

    ASSERT_EQ(0.0, mean);
}

TEST(doublestats, mean) {
    da_int n = 4;
    double *x = nullptr;

    x = (double *)malloc(n * sizeof(double));

    // Starting point
    for (da_int i = 0; i < n; i++)
        x[i] = (double)i;

    double mean;

    // int err = 1;
    da_status err = da_mean_d(n, x, 1, &mean);

    if (err == da_status_success) {
        std::cout << "mean = " << mean << std::endl;
    } else {
        std::cout << "error computing mean" << std::endl;
    }

    if (x)
        free(x);

    ASSERT_EQ(1.5, mean);
}

TEST(floatstats, mean) {
    da_int n = 4;
    float *x = nullptr;

    x = (float *)malloc(n * sizeof(float));

    // Starting point
    for (da_int i = 0; i < n; i++)
        x[i] = 0.25;

    float mean;

    // int err = 1;
    da_status err = da_mean_s(n, x, 1, &mean);

    if (err == da_status_success) {
        std::cout << "mean = " << mean << std::endl;
    } else {
        std::cout << "error computing mean" << std::endl;
    }

    if (x)
        free(x);

    ASSERT_EQ(0.25, mean);
}