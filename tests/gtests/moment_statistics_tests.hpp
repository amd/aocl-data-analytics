#include "aoclda.h"
#include "utest_utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <list>

template <typename T> class MomentStatisticsTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> struct MomentsParamType {
    da_int n;
    da_int p;
    da_int ldx;
    da_int k;
    std::vector<T> x;
    std::vector<T> expected_column_means;
    std::vector<T> expected_row_means;
    T expected_overall_mean;
    std::vector<T> expected_column_harmonic_means;
    std::vector<T> expected_row_harmonic_means;
    T expected_overall_harmonic_mean;
    std::vector<T> expected_column_geometric_means;
    std::vector<T> expected_row_geometric_means;
    T expected_overall_geometric_mean;
    std::vector<T> expected_column_variances;
    std::vector<T> expected_row_variances;
    T expected_overall_variance;
    std::vector<T> expected_column_skewnesses;
    std::vector<T> expected_row_skewnesses;
    T expected_overall_skewness;
    std::vector<T> expected_column_kurtoses;
    std::vector<T> expected_row_kurtoses;
    T expected_overall_kurtosis;
    std::vector<T> expected_column_moments;
    std::vector<T> expected_row_moments;
    T expected_overall_moment;
    da_status expected_status;
    T epsilon;
};

template <typename T> void GetZeroData(std::vector<MomentsParamType<T>> &params) {
    // Test with data matrix full of zeros
    MomentsParamType<T> param;
    param.n = 4;
    param.p = 4;
    param.ldx = param.n;
    std::vector<double> x(param.n * param.p, 0);
    param.x = convert_vector<double, T>(x);
    param.k = 4;
    std::vector<double> expected_column_means(param.p, 0);
    std::vector<double> expected_row_means(param.n, 0);
    param.expected_column_means = convert_vector<double, T>(expected_column_means);
    param.expected_row_means = convert_vector<double, T>(expected_row_means);
    param.expected_overall_mean = (T)0;
    std::vector<double> expected_column_harmonic_means(param.p, 0);
    std::vector<double> expected_row_harmonic_means(param.n, 0);
    param.expected_column_harmonic_means =
        convert_vector<double, T>(expected_column_harmonic_means);
    param.expected_row_harmonic_means =
        convert_vector<double, T>(expected_row_harmonic_means);
    param.expected_overall_harmonic_mean = (T)0;
    std::vector<double> expected_column_geometric_means(param.p, 0);
    std::vector<double> expected_row_geometric_means(param.n, 0);
    param.expected_column_geometric_means =
        convert_vector<double, T>(expected_column_geometric_means);
    param.expected_row_geometric_means =
        convert_vector<double, T>(expected_row_geometric_means);
    param.expected_overall_geometric_mean = (T)0;
    std::vector<double> expected_column_variances(param.p, 0);
    std::vector<double> expected_row_variances(param.n, 0);
    param.expected_column_variances =
        convert_vector<double, T>(expected_column_variances);
    param.expected_row_variances = convert_vector<double, T>(expected_row_variances);
    param.expected_overall_variance = (T)0;
    std::vector<double> expected_column_skewnesses(param.p, 0);
    std::vector<double> expected_row_skewnesses(param.n, 0);
    param.expected_column_skewnesses =
        convert_vector<double, T>(expected_column_skewnesses);
    param.expected_row_skewnesses = convert_vector<double, T>(expected_row_skewnesses);
    param.expected_overall_skewness = (T)0;
    std::vector<double> expected_column_kurtoses(param.p, -3);
    std::vector<double> expected_row_kurtoses(param.n, -3);
    param.expected_column_kurtoses = convert_vector<double, T>(expected_column_kurtoses);
    param.expected_row_kurtoses = convert_vector<double, T>(expected_row_kurtoses);
    param.expected_overall_kurtosis = (T)-3;
    std::vector<double> expected_column_moments(param.p, 0);
    std::vector<double> expected_row_moments(param.n, 0);
    param.expected_column_moments = convert_vector<double, T>(expected_column_moments);
    param.expected_row_moments = convert_vector<double, T>(expected_row_moments);
    param.expected_overall_moment = (T)0;

    param.expected_status = da_status_success;
    param.epsilon = std::numeric_limits<T>::epsilon();
    ;
    params.push_back(param);
}

template <typename T> void GetOnesData(std::vector<MomentsParamType<T>> &params) {
    // Test with data matrix full of ones
    MomentsParamType<T> param;
    param.n = 7;
    param.p = 7;
    param.ldx = param.n;
    std::vector<double> x(param.n * param.p, 1);
    param.x = convert_vector<double, T>(x);
    param.k = 8;
    std::vector<double> expected_column_means(param.p, 1);
    std::vector<double> expected_row_means(param.n, 1);
    param.expected_column_means = convert_vector<double, T>(expected_column_means);
    param.expected_row_means = convert_vector<double, T>(expected_row_means);
    param.expected_overall_mean = (T)1;
    std::vector<double> expected_column_harmonic_means(param.p, 1);
    std::vector<double> expected_row_harmonic_means(param.n, 1);
    param.expected_column_harmonic_means =
        convert_vector<double, T>(expected_column_harmonic_means);
    param.expected_row_harmonic_means =
        convert_vector<double, T>(expected_row_harmonic_means);
    param.expected_overall_harmonic_mean = (T)1;
    std::vector<double> expected_column_geometric_means(param.p, 1);
    std::vector<double> expected_row_geometric_means(param.n, 1);
    param.expected_column_geometric_means =
        convert_vector<double, T>(expected_column_geometric_means);
    param.expected_row_geometric_means =
        convert_vector<double, T>(expected_row_geometric_means);
    param.expected_overall_geometric_mean = (T)1;
    std::vector<double> expected_column_variances(param.p, 0);
    std::vector<double> expected_row_variances(param.n, 0);
    param.expected_column_variances =
        convert_vector<double, T>(expected_column_variances);
    param.expected_row_variances = convert_vector<double, T>(expected_row_variances);
    param.expected_overall_variance = (T)0;
    std::vector<double> expected_column_skewnesses(param.p, 0);
    std::vector<double> expected_row_skewnesses(param.n, 0);
    param.expected_column_skewnesses =
        convert_vector<double, T>(expected_column_skewnesses);
    param.expected_row_skewnesses = convert_vector<double, T>(expected_row_skewnesses);
    param.expected_overall_skewness = (T)0;
    std::vector<double> expected_column_kurtoses(param.p, -3);
    std::vector<double> expected_row_kurtoses(param.n, -3);
    param.expected_column_kurtoses = convert_vector<double, T>(expected_column_kurtoses);
    param.expected_row_kurtoses = convert_vector<double, T>(expected_row_kurtoses);
    param.expected_overall_kurtosis = (T)-3;
    std::vector<double> expected_column_moments(param.p, 0);
    std::vector<double> expected_row_moments(param.n, 0);
    param.expected_column_moments = convert_vector<double, T>(expected_column_moments);
    param.expected_row_moments = convert_vector<double, T>(expected_row_moments);
    param.expected_overall_moment = (T)0;

    param.expected_status = da_status_success;
    param.epsilon = 0;
    params.push_back(param);
}

template <typename T> void GetTallThinData(std::vector<MomentsParamType<T>> &params) {
    // Test with normal tall thin data matrix
    MomentsParamType<T> param;
    param.n = 5;
    param.p = 3;
    param.ldx = param.n;
    std::vector<double> x{1.8, 2.1,  3.3,  4.9,  5.1,  6.2,  7.2, 8.2,
                          9.9, 10.4, 11.0, 12.6, 13.8, 14.1, 15.7};
    param.x = convert_vector<double, T>(x);
    param.k = 6;
    std::vector<double> expected_column_means{3.4400000000000004, 8.379999999999999,
                                              13.440000000000001};
    std::vector<double> expected_row_means{6.333333333333333, 7.3, 8.433333333333334,
                                           9.633333333333333, 10.4};
    param.expected_column_means = convert_vector<double, T>(expected_column_means);
    param.expected_row_means = convert_vector<double, T>(expected_row_means);
    param.expected_overall_mean = (T)8.419999999999998;
    std::vector<double> expected_column_harmonic_means{
        2.88195002621003, 8.073704810580168, 13.250150944893514};
    std::vector<double> expected_row_harmonic_means{
        3.713997579669221, 4.32, 6.030813953488372, 7.978432287413975, 8.42870542191032};
    param.expected_column_harmonic_means =
        convert_vector<double, T>(expected_column_harmonic_means);
    param.expected_row_harmonic_means =
        convert_vector<double, T>(expected_row_harmonic_means);
    param.expected_overall_harmonic_mean = (T)5.491317375094326;
    std::vector<double> expected_column_geometric_means{
        3.1532209196170395, 8.226996535935449, 13.346033675337713};
    std::vector<double> expected_row_geometric_means{
        4.969953132169934, 5.754056368993306, 7.2011572214028075, 8.810829470582446,
        9.408081174967142};
    param.expected_column_geometric_means =
        convert_vector<double, T>(expected_column_geometric_means);
    param.expected_row_geometric_means =
        convert_vector<double, T>(expected_row_geometric_means);
    param.expected_overall_geometric_mean = (T)7.021813816968474;
    std::vector<double> expected_column_variances{2.348, 3.1420000000000003,
                                                  3.0829999999999993};
    std::vector<double> expected_row_variances{21.173333333333332, 27.57,
                                               27.60333333333334, 21.21333333333333,
                                               28.089999999999996};
    param.expected_column_variances =
        convert_vector<double, T>(expected_column_variances);
    param.expected_row_variances = convert_vector<double, T>(expected_row_variances);
    param.expected_overall_variance = (T)20.307428571428574;
    std::vector<double> expected_column_skewnesses{
        0.0673265881163833, -0.0127915008072743, -0.1674052876561382};
    std::vector<double> expected_row_skewnesses{
        5.3188312013560390e-02, 3.4975260081736619e-02, 8.1428318452524656e-02,
        -1.0600903466702061e-01, -5.8453190991673130e-16};
    param.expected_column_skewnesses =
        convert_vector<double, T>(expected_column_skewnesses);
    param.expected_row_skewnesses = convert_vector<double, T>(expected_row_skewnesses);
    param.expected_overall_skewness = (T)0.04545762262656644;
    std::vector<double> expected_column_kurtoses{-1.7192128064335448, -1.5276548282625777,
                                                 -0.9537719483099689};
    std::vector<double> expected_row_kurtoses{-1.4999999999999998, -1.5000000000000002,
                                              -1.4999999999999998, -1.5000000000000009,
                                              -1.4999999999999993};
    param.expected_column_kurtoses = convert_vector<double, T>(expected_column_kurtoses);
    param.expected_row_kurtoses = convert_vector<double, T>(expected_row_kurtoses);
    param.expected_overall_kurtosis = (T)-1.2255225724617373;
    std::vector<double> expected_column_moments{11.171069067519996, 38.06081606528,
                                                68.94169886271999};
    std::vector<double> expected_row_moments{6336.101916795611, 13978.323597999994,
                                             14062.783149659814, 6395.864148104248,
                                             14776.240752666665};
    param.expected_column_moments = convert_vector<double, T>(expected_column_moments);
    param.expected_row_moments = convert_vector<double, T>(expected_row_moments);
    param.expected_overall_moment = (T)25444.255086817273;

    param.expected_status = da_status_success;
    param.epsilon = 100 * sqrt(std::numeric_limits<T>::epsilon());
    params.push_back(param);
}

template <typename T> void GetShortFatData(std::vector<MomentsParamType<T>> &params) {
    // Test with normal short fat data matrix
    MomentsParamType<T> param;
    param.n = 3;
    param.p = 5;
    param.ldx = param.n;
    std::vector<double> x{1.8,  6.2, 11.0, 2.1,  7.2, 12.6, 3.3, 8.2,
                          13.8, 4.9, 9.9,  14.1, 5.1, 10.4, 15.7};
    param.x = convert_vector<double, T>(x);
    param.k = 6;
    std::vector<double> expected_row_means{3.4400000000000004, 8.379999999999999,
                                           13.440000000000001};
    std::vector<double> expected_column_means{6.333333333333333, 7.3, 8.433333333333334,
                                              9.633333333333333, 10.4};
    param.expected_column_means = convert_vector<double, T>(expected_column_means);
    param.expected_row_means = convert_vector<double, T>(expected_row_means);
    param.expected_overall_mean = (T)8.419999999999998;
    std::vector<double> expected_row_harmonic_means{2.88195002621003, 8.073704810580168,
                                                    13.250150944893514};
    std::vector<double> expected_column_harmonic_means{
        3.713997579669221, 4.32, 6.030813953488372, 7.978432287413975, 8.42870542191032};
    param.expected_column_harmonic_means =
        convert_vector<double, T>(expected_column_harmonic_means);
    param.expected_row_harmonic_means =
        convert_vector<double, T>(expected_row_harmonic_means);
    param.expected_overall_harmonic_mean = (T)5.491317375094326;
    std::vector<double> expected_row_geometric_means{
        3.1532209196170395, 8.226996535935449, 13.346033675337713};
    std::vector<double> expected_column_geometric_means{
        4.969953132169934, 5.754056368993306, 7.2011572214028075, 8.810829470582446,
        9.408081174967142};
    param.expected_column_geometric_means =
        convert_vector<double, T>(expected_column_geometric_means);
    param.expected_row_geometric_means =
        convert_vector<double, T>(expected_row_geometric_means);
    param.expected_overall_geometric_mean = (T)7.021813816968474;
    std::vector<double> expected_row_variances{2.348, 3.1420000000000003,
                                               3.0829999999999993};
    std::vector<double> expected_column_variances{21.173333333333332, 27.57,
                                                  27.60333333333334, 21.21333333333333,
                                                  28.089999999999996};
    param.expected_column_variances =
        convert_vector<double, T>(expected_column_variances);
    param.expected_row_variances = convert_vector<double, T>(expected_row_variances);
    param.expected_overall_variance = (T)20.307428571428574;
    std::vector<double> expected_row_skewnesses{0.0673265881163833, -0.0127915008072743,
                                                -0.1674052876561382};
    std::vector<double> expected_column_skewnesses{
        5.3188312013560390e-02, 3.4975260081736619e-02, 8.1428318452524656e-02,
        -1.0600903466702061e-01, -5.8453190991673130e-16};
    param.expected_column_skewnesses =
        convert_vector<double, T>(expected_column_skewnesses);
    param.expected_row_skewnesses = convert_vector<double, T>(expected_row_skewnesses);
    param.expected_overall_skewness = (T)0.04545762262656644;
    std::vector<double> expected_row_kurtoses{-1.7192128064335448, -1.5276548282625777,
                                              -0.9537719483099689};
    std::vector<double> expected_column_kurtoses{-1.4999999999999998, -1.5000000000000002,
                                                 -1.4999999999999998, -1.5000000000000009,
                                                 -1.4999999999999993};
    param.expected_column_kurtoses = convert_vector<double, T>(expected_column_kurtoses);
    param.expected_row_kurtoses = convert_vector<double, T>(expected_row_kurtoses);
    param.expected_overall_kurtosis = (T)-1.2255225724617373;
    std::vector<double> expected_row_moments{11.171069067519996, 38.06081606528,
                                             68.94169886271999};
    std::vector<double> expected_column_moments{6336.101916795611, 13978.323597999994,
                                                14062.783149659814, 6395.864148104248,
                                                14776.240752666665};
    param.expected_column_moments = convert_vector<double, T>(expected_column_moments);
    param.expected_row_moments = convert_vector<double, T>(expected_row_moments);
    param.expected_overall_moment = (T)25444.255086817273;

    param.expected_status = da_status_success;
    param.epsilon = 100 * sqrt(std::numeric_limits<T>::epsilon());
    params.push_back(param);
}

template <typename T> void GetSubarrayData(std::vector<MomentsParamType<T>> &params) {
    // Subarray test
    MomentsParamType<T> param;
    param.n = 3;
    param.p = 5;
    param.ldx = param.n + 3;
    std::vector<double> x{1.8,  6.2, 11.0, 0,   0,    0,    2.1,  7.2, 12.6, 0,
                          0,    0,   3.3,  8.2, 13.8, 0,    0,    0,   4.9,  9.9,
                          14.1, 0,   0,    0,   5.1,  10.4, 15.7, 0,   0,    0};
    param.x = convert_vector<double, T>(x);
    param.k = 6;
    std::vector<double> expected_row_means{3.4400000000000004, 8.379999999999999,
                                           13.440000000000001};
    std::vector<double> expected_column_means{6.333333333333333, 7.3, 8.433333333333334,
                                              9.633333333333333, 10.4};
    param.expected_column_means = convert_vector<double, T>(expected_column_means);
    param.expected_row_means = convert_vector<double, T>(expected_row_means);
    param.expected_overall_mean = (T)8.419999999999998;
    std::vector<double> expected_row_harmonic_means{2.88195002621003, 8.073704810580168,
                                                    13.250150944893514};
    std::vector<double> expected_column_harmonic_means{
        3.713997579669221, 4.32, 6.030813953488372, 7.978432287413975, 8.42870542191032};
    param.expected_column_harmonic_means =
        convert_vector<double, T>(expected_column_harmonic_means);
    param.expected_row_harmonic_means =
        convert_vector<double, T>(expected_row_harmonic_means);
    param.expected_overall_harmonic_mean = (T)5.491317375094326;
    std::vector<double> expected_row_geometric_means{
        3.1532209196170395, 8.226996535935449, 13.346033675337713};
    std::vector<double> expected_column_geometric_means{
        4.969953132169934, 5.754056368993306, 7.2011572214028075, 8.810829470582446,
        9.408081174967142};
    param.expected_column_geometric_means =
        convert_vector<double, T>(expected_column_geometric_means);
    param.expected_row_geometric_means =
        convert_vector<double, T>(expected_row_geometric_means);
    param.expected_overall_geometric_mean = (T)7.021813816968474;
    std::vector<double> expected_row_variances{2.348, 3.1420000000000003,
                                               3.0829999999999993};
    std::vector<double> expected_column_variances{21.173333333333332, 27.57,
                                                  27.60333333333334, 21.21333333333333,
                                                  28.089999999999996};
    param.expected_column_variances =
        convert_vector<double, T>(expected_column_variances);
    param.expected_row_variances = convert_vector<double, T>(expected_row_variances);
    param.expected_overall_variance = (T)20.307428571428574;
    std::vector<double> expected_row_skewnesses{0.0673265881163833, -0.0127915008072743,
                                                -0.1674052876561382};
    std::vector<double> expected_column_skewnesses{
        5.3188312013560390e-02, 3.4975260081736619e-02, 8.1428318452524656e-02,
        -1.0600903466702061e-01, -5.8453190991673130e-16};
    param.expected_column_skewnesses =
        convert_vector<double, T>(expected_column_skewnesses);
    param.expected_row_skewnesses = convert_vector<double, T>(expected_row_skewnesses);
    param.expected_overall_skewness = (T)0.04545762262656644;
    std::vector<double> expected_row_kurtoses{-1.7192128064335448, -1.5276548282625777,
                                              -0.9537719483099689};
    std::vector<double> expected_column_kurtoses{-1.4999999999999998, -1.5000000000000002,
                                                 -1.4999999999999998, -1.5000000000000009,
                                                 -1.4999999999999993};
    param.expected_column_kurtoses = convert_vector<double, T>(expected_column_kurtoses);
    param.expected_row_kurtoses = convert_vector<double, T>(expected_row_kurtoses);
    param.expected_overall_kurtosis = (T)-1.2255225724617373;
    std::vector<double> expected_row_moments{11.171069067519996, 38.06081606528,
                                             68.94169886271999};
    std::vector<double> expected_column_moments{6336.101916795611, 13978.323597999994,
                                                14062.783149659814, 6395.864148104248,
                                                14776.240752666665};
    param.expected_column_moments = convert_vector<double, T>(expected_column_moments);
    param.expected_row_moments = convert_vector<double, T>(expected_row_moments);
    param.expected_overall_moment = (T)25444.255086817273;

    param.expected_status = da_status_success;
    param.epsilon = 100 * sqrt(std::numeric_limits<T>::epsilon());
    params.push_back(param);
}

template <typename T> void GetSingleRowData(std::vector<MomentsParamType<T>> &params) {
    // Single row test
    MomentsParamType<T> param;
    param.n = 1;
    param.p = 5;
    param.ldx = param.n;
    std::vector<double> x{4.7, 2.6, 7.4, 9.5, 4.6};
    param.x = convert_vector<double, T>(x);
    param.k = 7;
    std::vector<double> expected_row_means{5.76};
    std::vector<double> expected_column_means{4.7, 2.6, 7.4, 9.5, 4.6};
    param.expected_column_means = convert_vector<double, T>(expected_column_means);
    param.expected_row_means = convert_vector<double, T>(expected_row_means);
    param.expected_overall_mean = (T)5.76;
    std::vector<double> expected_row_harmonic_means{4.7385687125292835};
    std::vector<double> expected_column_harmonic_means{4.7, 2.6, 7.4, 9.5, 4.6};
    param.expected_column_harmonic_means =
        convert_vector<double, T>(expected_column_harmonic_means);
    param.expected_row_harmonic_means =
        convert_vector<double, T>(expected_row_harmonic_means);
    param.expected_overall_harmonic_mean = (T)4.7385687125292835;
    std::vector<double> expected_row_geometric_means{5.240308712574584};
    std::vector<double> expected_column_geometric_means{4.7, 2.6, 7.4, 9.5, 4.6};
    param.expected_column_geometric_means =
        convert_vector<double, T>(expected_column_geometric_means);
    param.expected_row_geometric_means =
        convert_vector<double, T>(expected_row_geometric_means);
    param.expected_overall_geometric_mean = (T)5.240308712574584;
    std::vector<double> expected_row_variances{7.283};
    std::vector<double> expected_column_variances{0, 0, 0, 0, 0};
    param.expected_column_variances =
        convert_vector<double, T>(expected_column_variances);
    param.expected_row_variances = convert_vector<double, T>(expected_row_variances);
    param.expected_overall_variance = (T)7.283;
    std::vector<double> expected_row_skewnesses{0.31880822305631984};
    std::vector<double> expected_column_skewnesses{0, 0, 0, 0, 0};
    param.expected_column_skewnesses =
        convert_vector<double, T>(expected_column_skewnesses);
    param.expected_row_skewnesses = convert_vector<double, T>(expected_row_skewnesses);
    param.expected_overall_skewness = (T)0.31880822305631984;
    std::vector<double> expected_row_kurtoses{-1.1991174876238382};
    std::vector<double> expected_column_kurtoses{-3, -3, -3, -3, -3};
    param.expected_column_kurtoses = convert_vector<double, T>(expected_column_kurtoses);
    param.expected_row_kurtoses = convert_vector<double, T>(expected_row_kurtoses);
    param.expected_overall_kurtosis = (T)-1.1991174876238382;
    std::vector<double> expected_row_moments{1423.3050598873851};
    std::vector<double> expected_column_moments{0, 0, 0, 0, 0};
    param.expected_column_moments = convert_vector<double, T>(expected_column_moments);
    param.expected_row_moments = convert_vector<double, T>(expected_row_moments);
    param.expected_overall_moment = (T)1423.3050598873851;

    param.expected_status = da_status_success;
    param.epsilon = 10 * sqrt(std::numeric_limits<T>::epsilon());
    params.push_back(param);
}

template <typename T> void GetSingleColumnData(std::vector<MomentsParamType<T>> &params) {
    // Single row test
    MomentsParamType<T> param;
    param.n = 5;
    param.p = 1;
    param.ldx = param.n;
    std::vector<double> x{4.7, 2.6, 7.4, 9.5, 4.6};
    param.x = convert_vector<double, T>(x);
    param.k = 7;
    std::vector<double> expected_column_means{5.76};
    std::vector<double> expected_row_means{4.7, 2.6, 7.4, 9.5, 4.6};
    param.expected_column_means = convert_vector<double, T>(expected_column_means);
    param.expected_row_means = convert_vector<double, T>(expected_row_means);
    param.expected_overall_mean = (T)5.76;
    std::vector<double> expected_column_harmonic_means{4.7385687125292835};
    std::vector<double> expected_row_harmonic_means{4.7, 2.6, 7.4, 9.5, 4.6};
    param.expected_column_harmonic_means =
        convert_vector<double, T>(expected_column_harmonic_means);
    param.expected_row_harmonic_means =
        convert_vector<double, T>(expected_row_harmonic_means);
    param.expected_overall_harmonic_mean = (T)4.7385687125292835;
    std::vector<double> expected_column_geometric_means{5.240308712574584};
    std::vector<double> expected_row_geometric_means{4.7, 2.6, 7.4, 9.5, 4.6};
    param.expected_column_geometric_means =
        convert_vector<double, T>(expected_column_geometric_means);
    param.expected_row_geometric_means =
        convert_vector<double, T>(expected_row_geometric_means);
    param.expected_overall_geometric_mean = (T)5.240308712574584;
    std::vector<double> expected_column_variances{7.283};
    std::vector<double> expected_row_variances{0, 0, 0, 0, 0};
    param.expected_column_variances =
        convert_vector<double, T>(expected_column_variances);
    param.expected_row_variances = convert_vector<double, T>(expected_row_variances);
    param.expected_overall_variance = (T)7.283;
    std::vector<double> expected_column_skewnesses{0.31880822305631984};
    std::vector<double> expected_row_skewnesses{0, 0, 0, 0, 0};
    param.expected_column_skewnesses =
        convert_vector<double, T>(expected_column_skewnesses);
    param.expected_row_skewnesses = convert_vector<double, T>(expected_row_skewnesses);
    param.expected_overall_skewness = (T)0.31880822305631984;
    std::vector<double> expected_column_kurtoses{-1.1991174876238382};
    std::vector<double> expected_row_kurtoses{-3, -3, -3, -3, -3};
    param.expected_column_kurtoses = convert_vector<double, T>(expected_column_kurtoses);
    param.expected_row_kurtoses = convert_vector<double, T>(expected_row_kurtoses);
    param.expected_overall_kurtosis = (T)-1.1991174876238382;
    std::vector<double> expected_column_moments{1423.3050598873851};
    std::vector<double> expected_row_moments{0, 0, 0, 0, 0};
    param.expected_column_moments = convert_vector<double, T>(expected_column_moments);
    param.expected_row_moments = convert_vector<double, T>(expected_row_moments);
    param.expected_overall_moment = (T)1423.3050598873851;

    param.expected_status = da_status_success;
    param.epsilon = 10 * sqrt(std::numeric_limits<T>::epsilon());
    params.push_back(param);
}

template <typename T> void Get1by1Data(std::vector<MomentsParamType<T>> &params) {
    // 1 by 1 test
    MomentsParamType<T> param;
    param.n = 1;
    param.p = 1;
    param.ldx = param.n;
    std::vector<double> x{4.7};
    param.x = convert_vector<double, T>(x);
    param.k = 7;
    std::vector<double> expected_column_means{4.7};
    std::vector<double> expected_row_means{4.7};
    param.expected_column_means = convert_vector<double, T>(expected_column_means);
    param.expected_row_means = convert_vector<double, T>(expected_row_means);
    param.expected_overall_mean = (T)4.7;
    std::vector<double> expected_column_harmonic_means{4.7};
    std::vector<double> expected_row_harmonic_means{4.7};
    param.expected_column_harmonic_means =
        convert_vector<double, T>(expected_column_harmonic_means);
    param.expected_row_harmonic_means =
        convert_vector<double, T>(expected_row_harmonic_means);
    param.expected_overall_harmonic_mean = (T)4.7;
    std::vector<double> expected_column_geometric_means{4.7};
    std::vector<double> expected_row_geometric_means{4.7};
    param.expected_column_geometric_means =
        convert_vector<double, T>(expected_column_geometric_means);
    param.expected_row_geometric_means =
        convert_vector<double, T>(expected_row_geometric_means);
    param.expected_overall_geometric_mean = (T)4.7;
    std::vector<double> expected_column_variances{0};
    std::vector<double> expected_row_variances{0};
    param.expected_column_variances =
        convert_vector<double, T>(expected_column_variances);
    param.expected_row_variances = convert_vector<double, T>(expected_row_variances);
    param.expected_overall_variance = (T)0;
    std::vector<double> expected_column_skewnesses{0};
    std::vector<double> expected_row_skewnesses{0};
    param.expected_column_skewnesses =
        convert_vector<double, T>(expected_column_skewnesses);
    param.expected_row_skewnesses = convert_vector<double, T>(expected_row_skewnesses);
    param.expected_overall_skewness = (T)0;
    std::vector<double> expected_column_kurtoses{-3};
    std::vector<double> expected_row_kurtoses{-3};
    param.expected_column_kurtoses = convert_vector<double, T>(expected_column_kurtoses);
    param.expected_row_kurtoses = convert_vector<double, T>(expected_row_kurtoses);
    param.expected_overall_kurtosis = (T)-3;
    std::vector<double> expected_column_moments{0};
    std::vector<double> expected_row_moments{0};
    param.expected_column_moments = convert_vector<double, T>(expected_column_moments);
    param.expected_row_moments = convert_vector<double, T>(expected_row_moments);
    param.expected_overall_moment = (T)0;

    param.expected_status = da_status_success;
    param.epsilon = std::numeric_limits<T>::epsilon();
    params.push_back(param);
}

template <typename T> void GetMomentsData(std::vector<MomentsParamType<T>> &params) {

    GetZeroData(params);
    GetOnesData(params);
    GetTallThinData(params);
    GetShortFatData(params);
    GetSubarrayData(params);
    GetSingleRowData(params);
    GetSingleColumnData(params);
    Get1by1Data(params);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MomentStatisticsTest, FloatTypes);
TYPED_TEST(MomentStatisticsTest, MomentsFunctionality) {

    std::vector<MomentsParamType<TypeParam>> params;
    GetMomentsData(params);

    for (auto &param : params) {
        std::vector<TypeParam> column_stat(param.p);
        std::vector<TypeParam> row_stat(param.n);
        TypeParam overall_stat;
        std::vector<TypeParam> column_stat2(param.p);
        std::vector<TypeParam> row_stat2(param.n);
        TypeParam overall_stat2;
        std::vector<TypeParam> column_stat3(param.p);
        std::vector<TypeParam> row_stat3(param.n);
        TypeParam overall_stat3;

        ASSERT_EQ(da_mean(da_axis_col, param.n, param.p, param.x.data(), param.ldx,
                          column_stat.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_means.data(), column_stat.data(),
                        param.epsilon);
        ASSERT_EQ(da_mean(da_axis_row, param.n, param.p, param.x.data(), param.ldx,
                          row_stat.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_means.data(), row_stat.data(),
                        param.epsilon);
        ASSERT_EQ(da_mean(da_axis_all, param.n, param.p, param.x.data(), param.ldx,
                          &overall_stat),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_mean, overall_stat, param.epsilon);

        ASSERT_EQ(da_harmonic_mean(da_axis_col, param.n, param.p, param.x.data(),
                                   param.ldx, column_stat.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_harmonic_means.data(),
                        column_stat.data(), param.epsilon);
        ASSERT_EQ(da_harmonic_mean(da_axis_row, param.n, param.p, param.x.data(),
                                   param.ldx, row_stat.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_harmonic_means.data(),
                        row_stat.data(), param.epsilon);
        ASSERT_EQ(da_harmonic_mean(da_axis_all, param.n, param.p, param.x.data(),
                                   param.ldx, &overall_stat),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_harmonic_mean, overall_stat, param.epsilon);

        ASSERT_EQ(da_geometric_mean(da_axis_col, param.n, param.p, param.x.data(),
                                    param.ldx, column_stat.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_geometric_means.data(),
                        column_stat.data(), param.epsilon);
        ASSERT_EQ(da_geometric_mean(da_axis_row, param.n, param.p, param.x.data(),
                                    param.ldx, row_stat.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_geometric_means.data(),
                        row_stat.data(), param.epsilon);
        ASSERT_EQ(da_geometric_mean(da_axis_all, param.n, param.p, param.x.data(),
                                    param.ldx, &overall_stat),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_geometric_mean, overall_stat, param.epsilon);

        ASSERT_EQ(da_variance(da_axis_col, param.n, param.p, param.x.data(), param.ldx,
                              column_stat.data(), column_stat2.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_means.data(), column_stat.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_variances.data(),
                        column_stat2.data(), param.epsilon);
        ASSERT_EQ(da_variance(da_axis_row, param.n, param.p, param.x.data(), param.ldx,
                              row_stat.data(), row_stat2.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_means.data(), row_stat.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_variances.data(), row_stat2.data(),
                        param.epsilon);
        ASSERT_EQ(da_variance(da_axis_all, param.n, param.p, param.x.data(), param.ldx,
                              &overall_stat, &overall_stat2),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_mean, overall_stat, param.epsilon);
        EXPECT_NEAR(param.expected_overall_variance, overall_stat2, param.epsilon);

        ASSERT_EQ(da_skewness(da_axis_col, param.n, param.p, param.x.data(), param.ldx,
                              column_stat.data(), column_stat2.data(),
                              column_stat3.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_means.data(), column_stat.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_variances.data(),
                        column_stat2.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_skewnesses.data(),
                        column_stat3.data(), param.epsilon);
        ASSERT_EQ(da_skewness(da_axis_row, param.n, param.p, param.x.data(), param.ldx,
                              row_stat.data(), row_stat2.data(), row_stat3.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_means.data(), row_stat.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_variances.data(), row_stat2.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_skewnesses.data(), row_stat3.data(),
                        param.epsilon);
        ASSERT_EQ(da_skewness(da_axis_all, param.n, param.p, param.x.data(), param.ldx,
                              &overall_stat, &overall_stat2, &overall_stat3),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_mean, overall_stat, param.epsilon);
        EXPECT_NEAR(param.expected_overall_variance, overall_stat2, param.epsilon);
        EXPECT_NEAR(param.expected_overall_skewness, overall_stat3, param.epsilon);

        ASSERT_EQ(da_kurtosis(da_axis_col, param.n, param.p, param.x.data(), param.ldx,
                              column_stat.data(), column_stat2.data(),
                              column_stat3.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_means.data(), column_stat.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_variances.data(),
                        column_stat2.data(), param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_kurtoses.data(),
                        column_stat3.data(), param.epsilon);
        ASSERT_EQ(da_kurtosis(da_axis_row, param.n, param.p, param.x.data(), param.ldx,
                              row_stat.data(), row_stat2.data(), row_stat3.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_means.data(), row_stat.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_variances.data(), row_stat2.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_kurtoses.data(), row_stat3.data(),
                        param.epsilon);
        ASSERT_EQ(da_kurtosis(da_axis_all, param.n, param.p, param.x.data(), param.ldx,
                              &overall_stat, &overall_stat2, &overall_stat3),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_mean, overall_stat, param.epsilon);
        EXPECT_NEAR(param.expected_overall_variance, overall_stat2, param.epsilon);
        EXPECT_NEAR(param.expected_overall_kurtosis, overall_stat3, param.epsilon);

        ASSERT_EQ(da_moment(da_axis_col, param.n, param.p, param.x.data(), param.ldx,
                            param.k, 0, column_stat.data(), column_stat2.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.p, param.expected_column_means.data(), column_stat.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.p, param.expected_column_moments.data(),
                        column_stat2.data(), param.epsilon);
        ASSERT_EQ(da_moment(da_axis_row, param.n, param.p, param.x.data(), param.ldx,
                            param.k, 0, row_stat.data(), row_stat2.data()),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.n, param.expected_row_means.data(), row_stat.data(),
                        param.epsilon);
        EXPECT_ARR_NEAR(param.n, param.expected_row_moments.data(), row_stat2.data(),
                        param.epsilon);
        ASSERT_EQ(da_moment(da_axis_all, param.n, param.p, param.x.data(), param.ldx,
                            param.k, 0, &overall_stat, &overall_stat2),
                  param.expected_status);
        EXPECT_NEAR(param.expected_overall_mean, overall_stat, param.epsilon);
        EXPECT_NEAR(param.expected_overall_moment, overall_stat2, param.epsilon);
    }
}

TYPED_TEST(MomentStatisticsTest, IllegalArgsMoments) {

    std::vector<double> x_d{4.7, 1.2, -0.3, 4.5};
    std::vector<TypeParam> x = convert_vector<double, TypeParam>(x_d);
    da_int n = 2, p = 2, ldx = 2, k = 2;
    std::vector<TypeParam> dummy1(10, 0);
    std::vector<TypeParam> dummy2(10, 0);
    std::vector<TypeParam> dummy3(10, 0);

    // Test with illegal value of k
    da_int k_illegal = -3;
    ASSERT_EQ(da_moment(da_axis_all, n, p, x.data(), ldx, k_illegal, 0, dummy1.data(),
                        dummy2.data()),
              da_status_invalid_input);

    // Test with illegal value of ldx
    da_int ldx_illegal = 1;
    ASSERT_EQ(da_mean(da_axis_row, n, p, x.data(), ldx_illegal, dummy1.data()),
              da_status_invalid_leading_dimension);
    ASSERT_EQ(da_harmonic_mean(da_axis_row, n, p, x.data(), ldx_illegal, dummy1.data()),
              da_status_invalid_leading_dimension);
    ASSERT_EQ(da_geometric_mean(da_axis_row, n, p, x.data(), ldx_illegal, dummy1.data()),
              da_status_invalid_leading_dimension);
    ASSERT_EQ(da_variance(da_axis_row, n, p, x.data(), ldx_illegal, dummy1.data(),
                          dummy2.data()),
              da_status_invalid_leading_dimension);
    ASSERT_EQ(da_skewness(da_axis_row, n, p, x.data(), ldx_illegal, dummy1.data(),
                          dummy2.data(), dummy3.data()),
              da_status_invalid_leading_dimension);
    ASSERT_EQ(da_kurtosis(da_axis_row, n, p, x.data(), ldx_illegal, dummy1.data(),
                          dummy2.data(), dummy3.data()),
              da_status_invalid_leading_dimension);
    ASSERT_EQ(da_moment(da_axis_row, n, p, x.data(), ldx_illegal, k, 0, dummy1.data(),
                        dummy2.data()),
              da_status_invalid_leading_dimension);

    // Test with illegal p
    da_int p_illegal = 0;
    ASSERT_EQ(da_mean(da_axis_row, n, p_illegal, x.data(), ldx, dummy1.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_harmonic_mean(da_axis_row, n, p_illegal, x.data(), ldx, dummy1.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_geometric_mean(da_axis_row, n, p_illegal, x.data(), ldx, dummy1.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_variance(da_axis_row, n, p_illegal, x.data(), ldx, dummy1.data(),
                          dummy2.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_skewness(da_axis_row, n, p_illegal, x.data(), ldx, dummy1.data(),
                          dummy2.data(), dummy3.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_kurtosis(da_axis_row, n, p_illegal, x.data(), ldx, dummy1.data(),
                          dummy2.data(), dummy3.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_moment(da_axis_row, n, p_illegal, x.data(), ldx, k, 0, dummy1.data(),
                        dummy2.data()),
              da_status_invalid_array_dimension);

    // Test with illegal n
    da_int n_illegal = 0;
    ASSERT_EQ(da_mean(da_axis_col, n_illegal, p, x.data(), ldx, dummy1.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_harmonic_mean(da_axis_col, n_illegal, p, x.data(), ldx, dummy1.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_geometric_mean(da_axis_col, n_illegal, p, x.data(), ldx, dummy1.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_variance(da_axis_col, n_illegal, p, x.data(), ldx, dummy1.data(),
                          dummy2.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_skewness(da_axis_col, n_illegal, p, x.data(), ldx, dummy1.data(),
                          dummy2.data(), dummy3.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_kurtosis(da_axis_col, n_illegal, p, x.data(), ldx, dummy1.data(),
                          dummy2.data(), dummy3.data()),
              da_status_invalid_array_dimension);
    ASSERT_EQ(da_moment(da_axis_col, n_illegal, p, x.data(), ldx, k, 0, dummy1.data(),
                        dummy2.data()),
              da_status_invalid_array_dimension);

    // Test illegal geometric mean
    ASSERT_EQ(da_geometric_mean(da_axis_row, n, p, x.data(), ldx, dummy1.data()),
              da_status_negative_data);
    ASSERT_EQ(da_geometric_mean(da_axis_col, n, p, x.data(), ldx, dummy2.data()),
              da_status_negative_data);
    ASSERT_EQ(da_geometric_mean(da_axis_all, n, p, x.data(), ldx, dummy3.data()),
              da_status_negative_data);

    // Test illegal pointers
    TypeParam *x_null = nullptr;
    ASSERT_EQ(da_mean(da_axis_col, n, p, x_null, ldx, dummy1.data()),
              da_status_invalid_pointer);
    ASSERT_EQ(da_harmonic_mean(da_axis_col, n, p, x_null, ldx, dummy1.data()),
              da_status_invalid_pointer);
    ASSERT_EQ(da_geometric_mean(da_axis_col, n, p, x_null, ldx, dummy1.data()),
              da_status_invalid_pointer);
    ASSERT_EQ(da_variance(da_axis_col, n, p, x_null, ldx, dummy1.data(), dummy2.data()),
              da_status_invalid_pointer);
    ASSERT_EQ(da_skewness(da_axis_col, n, p, x_null, ldx, dummy1.data(), dummy2.data(),
                          dummy3.data()),
              da_status_invalid_pointer);
    ASSERT_EQ(da_kurtosis(da_axis_col, n, p, x_null, ldx, dummy1.data(), dummy2.data(),
                          dummy3.data()),
              da_status_invalid_pointer);
    ASSERT_EQ(
        da_moment(da_axis_col, n, p, x_null, ldx, k, 0, dummy1.data(), dummy2.data()),
        da_status_invalid_pointer);
}