/*
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <list>

template <typename T> class KernelFunctionTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

template <typename T> struct KernelFunctionParamType {
    da_int m = 0; // number of rows of X
    da_int n = 0; // number of rows of Y
    da_int p = 0; // number of cols of X and Y
    da_int ldx = 0;
    da_int ldy = 0;
    da_int ldd = 0;
    da_int ldd_itself = 0;
    da_int kernel_size = 0;
    da_int kernel_itself_size = 0;

    std::string test_name;

    T gamma;
    da_int degree;
    T coef0;

    std::vector<T> x;
    std::vector<T> y;

    std::vector<T> rbf_kernel_expected;
    std::vector<T> rbf_kernel_with_itself_expected;

    std::vector<T> linear_kernel_expected;
    std::vector<T> linear_kernel_with_itself_expected;

    std::vector<T> polynomial_kernel_expected;
    std::vector<T> polynomial_kernel_with_itself_expected;

    std::vector<T> sigmoid_kernel_expected;
    std::vector<T> sigmoid_kernel_with_itself_expected;

    da_status expected_status = da_status_success;
    da_order order = column_major;
    T epsilon = 100 * std::numeric_limits<T>::epsilon();
    // Polynomial kernel have some tests with large values, hence more lenient epsilon
    T epsilon_polynomial = sqrt(std::numeric_limits<T>::epsilon());
};

template <typename T> void GetTallData(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 6 by 5 and Y is 2 by 5
    KernelFunctionParamType<T> param;
    param.test_name = "Tall matrix";
    param.m = 6;
    param.n = 2;
    param.p = 5;
    param.ldx = param.m;
    param.ldy = param.n;
    param.ldd = param.m;
    param.ldd_itself = param.ldd;
    param.kernel_size = param.ldd * param.n;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 0.5;
    param.degree = 7;
    param.coef0 = 2;

    std::vector<double> x{0.55, 0.65, 0.79, 0.09, 0.98, 0.64, 0.72, 0.44, 0.53, 0.02,
                          0.8,  0.14, 0.6,  0.89, 0.57, 0.83, 0.46, 0.94, 0.54, 0.96,
                          0.93, 0.78, 0.78, 0.52, 0.42, 0.38, 0.07, 0.87, 0.12, 0.41};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.26, 0.62, 0.77, 0.61, 0.46, 0.62, 0.57, 0.94, 0.02, 0.68};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{
        0.8749837186979339, 0.6950301868806096, 0.7855666299462941, 0.47357262453850596,
        0.7507369161321245, 0.6292982306935997, 0.8847059049434833, 0.907964498239863,
        0.8146473164114145, 0.6925325766037563, 0.7670141724615499, 0.75088707853109};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{
        1.0000000000000000, 0.8392471827448608, 0.8314367920628749, 0.6021191076526999,
        0.8358968990483183, 0.7942952781730198, 0.8392471827448608, 1.0000000000000000,
        0.8926595623172259, 0.6817788094932781, 0.7697419116461471, 0.8662774852301183,
        0.8314367920628749, 0.8926595623172259, 1.0000000000000000, 0.4770662065149722,
        0.9295543511462733, 0.7425983196756992, 0.6021191076526999, 0.6817788094932781,
        0.4770662065149722, 1.0000000000000000, 0.3499552464360404, 0.7377870840531275,
        0.8358968990483183, 0.7697419116461471, 0.9295543511462733, 0.3499552464360404,
        1.0000000000000000, 0.6270995368449068, 0.7942952781730198, 0.8662774852301184,
        0.7425983196756992, 0.7377870840531275, 0.6270995368449068, 1.0000000000000000};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{1.2896, 1.472,  1.4072, 0.8826,
                                               1.5294, 1.0112, 1.9454, 2.384,
                                               2.0883, 1.9074, 2.1956, 1.8326};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{
        1.6489, 1.8863, 1.6897, 1.3485, 1.8626, 1.4698, 1.8863, 2.4742, 2.1734,
        1.8854, 2.1928, 1.9692, 1.6897, 2.1734, 2.0997, 1.3411, 2.1942, 1.6279,
        1.3485, 1.8854, 1.3411, 2.0627, 1.1988, 1.6029, 1.8626, 2.1928, 2.1942,
        1.1988, 2.4348, 1.6264, 1.4698, 1.9692, 1.6279, 1.6029, 1.6264, 1.7513};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{
        905.211064176028,   1147.6584306405023, 1055.8374555060823, 516.8304082180927,
        1234.6279022558322, 619.9864281385471,  2051.4341548275343, 3376.293391864469,
        2422.491929696561,  1961.3932709015874, 2737.622918688313,  1794.0121704874616};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{
        1433.9606608842407, 1912.8758216821034, 1508.0495328968277, 978.1693068383559,
        1859.610013298783,  1144.4324370625866, 1912.8758216821034, 3724.712962815662,
        2669.6907306661437, 1910.8294442750512, 2728.9740848038855, 2109.613726299531,
        1508.0495328968277, 2669.6907306661437, 2454.422709199248,  968.735008989019,
        2733.2955695598503, 1397.058636636973,  978.1693068383559,  1910.8294442750512,
        968.735008989019,   2352.0826623352054, 801.8844692274317,  1354.1916431552233,
        1859.610013298783,  2728.9740848038855, 2733.2955695598503, 801.8844692274317,
        3568.9084075311252, 1394.4542208427795, 1144.4324370625866, 2109.613726299531,
        1397.058636636973,  1354.1916431552233, 1394.4542208427795, 1626.1184737921214};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{
        0.989963069309474,  0.9916295432860046, 0.9910716794832725, 0.984959392115766,
        0.9920946326248296, 0.9867623948326573, 0.9947779605482564, 0.9966289794673341,
        0.9954717490234584, 0.9945762530149062, 0.9959315334703542, 0.994156229061938};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{
        0.9929819369796267, 0.9944609163803843, 0.9932615659984216, 0.990534454166752,
        0.9943284488286025, 0.9916111853349231, 0.9944609163803843, 0.9969192874559449,
        0.9958403934500023, 0.9944559427863934, 0.9959201490715824, 0.9949004646096297,
        0.9932615659984216, 0.9958403934500023, 0.9955229630530381, 0.990464484516954,
        0.9959258452392736, 0.9928335329623567, 0.990534454166752,  0.9944559427863934,
        0.990464484516954,  0.995354601968639,  0.989014299548725,  0.992652779451855,
        0.9943284488286025, 0.9959201490715824, 0.9959258452392736, 0.989014299548725,
        0.9967956828302027, 0.9928228138007401, 0.9916111853349231, 0.9949004646096297,
        0.9928335329623567, 0.992652779451855,  0.9928228138007401, 0.993662851810401};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    param.epsilon_polynomial = 10 * sqrt(std::numeric_limits<T>::epsilon());

    params.push_back(param);
}

template <typename T> void GetFatData(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 4 by 5 and Y is 5 by 5
    KernelFunctionParamType<T> param;
    param.test_name = "Fat matrix";
    param.m = 4;
    param.n = 5;
    param.p = 5;
    param.ldx = param.m;
    param.ldy = param.n;
    param.ldd = param.m;
    param.ldd_itself = param.ldd;
    param.kernel_size = param.ldd * param.n;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 0.021;
    param.degree = 4;
    param.coef0 = -10;

    std::vector<double> x{0.36, 0.67, 0.57, 0.16, 0.44, 0.21, 0.44, 0.65, 0.7,  0.13,
                          0.99, 0.25, 0.06, 0.32, 0.1,  0.47, 0.67, 0.36, 0.21, 0.24};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.16, 0.37, 0.98, 0.04, 0.32, 0.11, 0.82, 0.47, 0.28,
                          0.41, 0.66, 0.1,  0.98, 0.12, 0.06, 0.14, 0.84, 0.6,
                          0.3,  0.69, 0.2,  0.1,  0.74, 0.12, 0.57};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{
        0.992097589523631,  0.9872924317473087, 0.9918934368477379, 0.9880972054409752,
        0.9702574794271395, 0.9833196831058525, 0.9682546094496013, 0.9947219779640015,
        0.9841625524439991, 0.9769800957225053, 0.9854075175194142, 0.9688973564482993,
        0.982834534562489,  0.9903885877378596, 0.9769288056138571, 0.9955683488744711,
        0.9829480585071125, 0.9927102996995713, 0.9709056337665474, 0.9942082373008241};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{
        1.0000000000000000, 0.9866996419453982, 0.9928562388398912, 0.9866582014306933,
        0.9866996419453982, 1.0000000000000000, 0.9818257798127193, 0.9894510347547102,
        0.9928562388398912, 0.9818257798127193, 1.0000000000000000, 0.9813269427166608,
        0.9866582014306933, 0.9894510347547102, 0.9813269427166608, 1.0000000000000000};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{
        0.7104, 0.3329, 0.849, 0.3759, 0.6814, 0.7379, 0.7757, 1.036,  1.7774, 1.3411,
        1.951,  1.1669, 0.32,  0.2404, 0.320,  0.3882, 0.7609, 0.7343, 0.6109, 0.7938};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{
        1.2657, 0.685,  1.2385, 0.7076, 0.685,  0.7419, 0.7106, 0.513,
        1.2385, 0.7106, 1.5527, 0.7221, 0.7076, 0.513,  0.7221, 0.7891};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{
        9940.459802435902, 9972.065709945513, 9928.87449735116,  9968.461768522302,
        9942.885138199845, 9938.160324970919, 9935.000239764373, 9913.259582440562,
        9851.53223318384,  9887.822603236064, 9837.120425528115, 9902.340105836976,
        9973.147082903462, 9979.821686659576, 9973.147082903462, 9967.431053352047,
        9936.237432019652, 9938.46132477875,  9948.783063976649, 9933.48734412679};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{
        9894.104337648832, 9942.584037911316, 9896.37116197118,  9940.693953431482,
        9942.584037911316, 9937.82588875891,  9940.443077523078, 9956.977584519358,
        9896.37116197118,  9940.443077523078, 9870.209732567644, 9939.481430530815,
        9940.693953431482, 9956.977584519358, 9939.481430530815, 9933.88017887067};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{
        -0.9999999957528429, -0.9999999958196506, -0.9999999957280473,
        -0.9999999958120941, -0.9999999957580129, -0.9999999957479346,
        -0.9999999957411787, -0.9999999956943633, -0.9999999955581816,
        -0.999999995638835,  -0.9999999955256772, -0.9999999956706266,
        -0.9999999958219149, -0.9999999958358597, -0.9999999958219149,
        -0.99999999580993,   -0.9999999957438251, -0.9999999957485776,
        -0.9999999957705548, -0.99999999573794};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{
        -0.9999999956526241, -0.9999999957573714, -0.9999999956575877,
        -0.9999999957533424, -0.9999999957573714, -0.9999999957472203,
        -0.9999999957528073, -0.9999999957879098, -0.9999999956575877,
        -0.9999999957528073, -0.9999999955999037, -0.9999999957507554,
        -0.9999999957533424, -0.9999999957879098, -0.9999999957507554,
        -0.9999999957387812};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    param.epsilon_polynomial = 10 * sqrt(std::numeric_limits<T>::epsilon());

    params.push_back(param);
}

template <typename T>
void GetSquareData(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 3 by 3 and Y is 2 by 3
    KernelFunctionParamType<T> param;
    param.test_name = "Square matrix";
    param.m = 3;
    param.n = 2;
    param.p = 3;
    param.ldx = param.m;
    param.ldy = param.n;
    param.ldd = param.m;
    param.ldd_itself = param.ldd;
    param.kernel_size = param.ldd * param.n;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 0.56;
    param.degree = 3;
    param.coef0 = 0;

    std::vector<double> x{0.27, 0.58, 0.67, 0.52, 0.93, 0.13, 0.09, 0.32, 0.72};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.29, 0.02, 0.18, 0.83, 0.59, 0.0};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{0.8146799029557975, 0.6683655664301619,
                                            0.9123532758894196, 0.9108728166276732,
                                            0.7877614211538729, 0.44874342100509407};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{
        1.0000000000000000, 0.8372973623180416, 0.672307132809492,
        0.8372973623180416, 1.0000000000000000, 0.6360131752847883,
        0.672307132809492,  0.6360131752847883, 1.0000000000000000};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{0.225, 0.5244, 0.6425,
                                               0.437, 0.7835, 0.1213};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{
        0.3514, 0.669, 0.3133, 0.669, 1.3037, 0.7399, 0.3133, 0.7399, 0.9842};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{
        0.002000376000000001, 0.025325156048338952, 0.046578283192000004,
        0.014655761602048003, 0.08446583572537603,  0.0003134342731627522};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{
        0.007620252331618309, 0.05258264575334404, 0.005400640459515394,
        0.05258264575334404,  0.3891321177533335,  0.07113497168601964,
        0.005400640459515394, 0.07113497168601964, 0.1674226312504239};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{0.12533741535962414, 0.28550366048241727,
                                                0.3450378565199457,  0.239949003567583,
                                                0.41261608138674444, 0.06782371438314469};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{
        0.19428265355571225, 0.3580435890892122, 0.1736696794145327,
        0.3580435890892122,  0.623109396445363,  0.3921548857544117,
        0.1736696794145327,  0.3921548857544117, 0.5013831136645216};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    params.push_back(param);
}

template <typename T> void Get1by1Data(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 1 by 1 and Y is 2 by 1
    KernelFunctionParamType<T> param;
    param.test_name = "1 by 1 data";
    param.m = 1;
    param.n = 2;
    param.p = 1;
    param.ldx = param.m;
    param.ldy = param.n;
    param.ldd = param.m;
    param.ldd_itself = param.ldd;
    param.kernel_size = param.ldd * param.n;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 1.00;
    param.degree = 1;
    param.coef0 = 0.22;

    std::vector<double> x{1.0};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.68, 0.27};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{0.9026684120809421, 0.586900487959338};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{1.0};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{0.68, 0.27};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{1.0};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{0.9, 0.49};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{1.22};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{0.7162978701990245, 0.45421643268225903};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{0.8396541756543753};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    params.push_back(param);
}

template <typename T>
void GetSingleRowData(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 1 by 3 and Y is 1 by 3
    KernelFunctionParamType<T> param;
    param.test_name = "Single row data";
    param.m = 1;
    param.n = 1;
    param.p = 3;
    param.ldx = param.m;
    param.ldy = param.n;
    param.ldd = param.m;
    param.ldd_itself = param.ldd;
    param.kernel_size = param.ldd * param.n;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 2.00;
    param.degree = 7;
    param.coef0 = 2;

    std::vector<double> x{0.74, 0.96, 0.25};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.58, 0.59, 0.57};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{0.5887227024451752};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{1.0};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{1.1381};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{1.5317};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{26146.04860949598};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{85328.80447860713};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{0.9996139122496994};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{0.9999200167093423};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    param.epsilon_polynomial = 100 * sqrt(std::numeric_limits<T>::epsilon());

    params.push_back(param);
}

template <typename T>
void GetSingleColData(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 3 by 1 and Y is 2 by 1
    KernelFunctionParamType<T> param;
    param.test_name = "Single column data";
    param.m = 3;
    param.n = 2;
    param.p = 1;
    param.ldx = param.m;
    param.ldy = param.n;
    param.ldd = param.m;
    param.ldd_itself = param.ldd;
    param.kernel_size = param.ldd * param.n;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 0.77;
    param.degree = 1;
    param.coef0 = -1.99;

    std::vector<double> x{0.22, 0.95, 0.45};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.85, 0.7};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{0.7366716562288878, 0.9923295690574124,
                                            0.8840868275124031, 0.8374380400915132,
                                            0.9530146528001675, 0.9530146528001675};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{
        1.0, 0.6634292913946963, 0.9600854385411696, 0.6634292913946963,
        1.0, 0.8248943182036034, 0.9600854385411696, 0.8248943182036034,
        1.0};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{0.187, 0.8075, 0.3825,
                                               0.154, 0.665,  0.315};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{
        0.0484, 0.209, 0.099, 0.209, 0.9025, 0.4275, 0.099, 0.4275, 0.2025};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{-1.84601, -1.368225, -1.695475,
                                                   -1.87142, -1.47795,  -1.74745};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{
        -1.952732, -1.82907, -1.91377,  -1.82907, -1.295075,
        -1.660825, -1.91377, -1.660825, -1.834075};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{-0.9513687499991067, -0.8782870393320632,
                                                -0.9348410003298346, -0.9537226842490685,
                                                -0.9010831985696338, -0.9410846199320462};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{
        -0.9605313451047509, -0.9497350129840015, -0.9574009135709308,
        -0.9497350129840015, -0.8604499074502396, -0.9303282218906712,
        -0.9574009135709308, -0.9303282218906712, -0.9502231954123237};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    params.push_back(param);
}

template <typename T>
void GetSubarrayData(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 3 by 2 and Y is 2 by 2
    KernelFunctionParamType<T> param;
    param.test_name = "Subarray data";
    param.m = 3;
    param.n = 2;
    param.p = 2;
    param.ldx = 5;
    param.ldy = 4;
    param.ldd = 4;
    param.ldd_itself = 5;
    param.kernel_size = param.ldd * param.n;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 1.25;
    param.degree = 2;
    param.coef0 = 0.5;

    std::vector<double> x{0.3, 0.4, 0.58, 0.0, 0.0, 0.81, 0.88, 0.88, 0.0, 0.0};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.69, 0.5, 0.0, 0.0, 0.73, 0.96, 0.0, 0.0};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{
        0.8202673133150612, 0.8752462531917483, 0.9576719421517508, 0.0,
        0.9248488132162048, 0.9797086964745179, 0.9841273200552852, 0.0};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{
        1.0000000000000000, 0.9815473735014455, 0.9011126512995565, 0.0000000000000000,
        0.0000000000000000, 0.9815473735014455, 1.0000000000000000, 0.960309164511413,
        0.0000000000000000, 0.0000000000000000, 0.9011126512995565, 0.960309164511413,
        1.0000000000000000, 0.0000000000000000, 0.0000000000000000};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{0.7983, 0.9184, 1.0426, 0.0,
                                               0.9276, 1.0448, 1.1348, 0.0};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{
        0.7461, 0.8328, 0.8868, 0.0,    0.0,    0.8328, 0.9344, 1.0064,
        0.0,    0.0,    0.8868, 1.0064, 1.1108, 0.0,    0.0};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{
        2.2436295156250003, 2.715904, 3.2517105625, 0.0,
        2.75394025,         3.261636, 3.68064225,   0.0};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{
        2.0524143906250005, 2.374681, 2.58727225, 0.0, 0.0,
        2.374681,           2.782224, 3.090564,   0.0, 0.0,
        2.58727225,         3.090564, 3.56643225, 0.0, 0.0};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{
        0.9047635125899193, 0.9285826639905341, 0.947141543911347,  0.0,
        0.9301498036071515, 0.947423845868397,  0.9577935365581978, 0.0};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{
        0.8922032846500844, 0.912288252454104,  0.9229380584505494, 0.0, 0.0,
        0.912288252454104,  0.9312867684845129, 0.9422792104259986, 0.0, 0.0,
        0.9229380584505494, 0.9422792104259986, 0.9552420415583727, 0.0, 0.0};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    param.epsilon_polynomial = 10 * sqrt(std::numeric_limits<T>::epsilon());

    params.push_back(param);
}

template <typename T>
void GetSubarrayRowMajorData(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 3 by 2 and Y is 2 by 2
    KernelFunctionParamType<T> param;
    param.test_name = "Subarray row major data";
    param.m = 3;
    param.n = 2;
    param.p = 2;
    param.ldx = 4;
    param.ldy = 3;
    param.ldd = 4;
    param.ldd_itself = 4;
    param.kernel_size = param.ldd * param.m;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 1.25;
    param.degree = 2;
    param.coef0 = 0.5;
    param.order = row_major;

    std::vector<double> x{0.3, 0.81, 0.0, 0.0, 0.4, 0.88, 0.0, 0.0, 0.58, 0.88, 0.0, 0.0};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.69, 0.73, 0.0, 0.5, 0.96, 0.0};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{
        0.8202673133150612, 0.9248488132162048, 0.0, 0.0,
        0.8752462531917483, 0.9797086964745179, 0.0, 0.0,
        0.9576719421517508, 0.9841273200552852, 0.0, 0.0};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{
        1.0000000000000000, 0.9815473735014455, 0.9011126512995565, 0.0000000000000000,
        0.9815473735014455, 1.0000000000000000, 0.960309164511413,  0.0000000000000000,
        0.9011126512995565, 0.960309164511413,  1.0000000000000000, 0.0000000000000000};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{
        0.7983, 0.9276, 0.0, 0.0, 0.9184, 1.0448, 0.0, 0.0, 1.0426, 1.1348, 0.0, 0.0};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{0.7461, 0.8328, 0.8868, 0.0,
                                                           0.8328, 0.9344, 1.0064, 0.0,
                                                           0.8868, 1.0064, 1.1108, 0.0};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{
        2.2436295156250003, 2.75394025, 0.0, 0.0, 2.715904, 3.261636, 0.0, 0.0,
        3.2517105625,       3.68064225, 0.0, 0.0};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{
        2.0524143906250005, 2.374681, 2.58727225, 0.0,      2.374681,   2.782224,
        3.090564,           0.0,      2.58727225, 3.090564, 3.56643225, 0.0};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{
        0.9047635125899193, 0.9301498036071515, 0.0, 0.0,
        0.9285826639905341, 0.947423845868397,  0.0, 0.0,
        0.947141543911347,  0.9577935365581978, 0.0, 0.0};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{
        0.8922032846500844, 0.912288252454104,  0.9229380584505494, 0.0,
        0.912288252454104,  0.9312867684845129, 0.9422792104259986, 0.0,
        0.9229380584505494, 0.9422792104259986, 0.9552420415583727, 0.0};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    param.epsilon_polynomial = 10 * sqrt(std::numeric_limits<T>::epsilon());

    params.push_back(param);
}

template <typename T>
void GetRowMajorData(std::vector<KernelFunctionParamType<T>> &params) {
    // X is 6 by 5 and Y is 2 by 5 (same dataset as in tall matrix test, just row major)
    KernelFunctionParamType<T> param;
    param.test_name = "Row major data";
    param.m = 6;
    param.n = 2;
    param.p = 5;
    param.ldx = param.p;
    param.ldy = param.p;
    param.ldd = param.n;
    param.ldd_itself = param.m;
    param.kernel_size = param.ldd * param.m;
    param.kernel_itself_size = param.ldd_itself * param.m;
    param.gamma = 0.5;
    param.degree = 7;
    param.coef0 = 2;
    param.order = row_major;

    std::vector<double> x{0.55, 0.72, 0.6,  0.54, 0.42, 0.65, 0.44, 0.89, 0.96, 0.38,
                          0.79, 0.53, 0.57, 0.93, 0.07, 0.09, 0.02, 0.83, 0.78, 0.87,
                          0.98, 0.8,  0.46, 0.78, 0.12, 0.64, 0.14, 0.94, 0.52, 0.41};
    param.x = convert_vector<double, T>(x);

    std::vector<double> y{0.26, 0.77, 0.46, 0.57, 0.02, 0.62, 0.61, 0.62, 0.94, 0.68};
    param.y = convert_vector<double, T>(y);

    std::vector<double> rbf_kernel_expected{
        0.8749837186979339, 0.8847059049434833, 0.6950301868806096,  0.907964498239863,
        0.7855666299462941, 0.8146473164114145, 0.47357262453850596, 0.6925325766037563,
        0.7507369161321245, 0.7670141724615499, 0.6292982306935997,  0.75088707853109};

    param.rbf_kernel_expected = convert_vector<double, T>(rbf_kernel_expected);

    std::vector<double> rbf_kernel_with_itself_expected{
        1.0000000000000000, 0.8392471827448608, 0.8314367920628749, 0.6021191076526999,
        0.8358968990483183, 0.7942952781730198, 0.8392471827448608, 1.0000000000000000,
        0.8926595623172259, 0.6817788094932781, 0.7697419116461471, 0.8662774852301184,
        0.8314367920628749, 0.8926595623172259, 1.0000000000000000, 0.4770662065149722,
        0.9295543511462733, 0.7425983196756992, 0.6021191076526999, 0.6817788094932781,
        0.4770662065149722, 1.0000000000000000, 0.3499552464360404, 0.7377870840531275,
        0.8358968990483183, 0.7697419116461471, 0.9295543511462733, 0.3499552464360404,
        1.0000000000000000, 0.6270995368449068, 0.7942952781730198, 0.8662774852301183,
        0.7425983196756992, 0.7377870840531275, 0.6270995368449068, 1.0000000000000000};

    param.rbf_kernel_with_itself_expected =
        convert_vector<double, T>(rbf_kernel_with_itself_expected);

    std::vector<double> linear_kernel_expected{1.2896, 1.9454, 1.472,  2.384,
                                               1.4072, 2.0883, 0.8826, 1.9074,
                                               1.5294, 2.1956, 1.0112, 1.8326};

    param.linear_kernel_expected = convert_vector<double, T>(linear_kernel_expected);

    std::vector<double> linear_kernel_with_itself_expected{
        1.6489, 1.8863, 1.6897, 1.3485, 1.8626, 1.4698, 1.8863, 2.4742, 2.1734,
        1.8854, 2.1928, 1.9692, 1.6897, 2.1734, 2.0997, 1.3411, 2.1942, 1.6279,
        1.3485, 1.8854, 1.3411, 2.0627, 1.1988, 1.6029, 1.8626, 2.1928, 2.1942,
        1.1988, 2.4348, 1.6264, 1.4698, 1.9692, 1.6279, 1.6029, 1.6264, 1.7513};

    param.linear_kernel_with_itself_expected =
        convert_vector<double, T>(linear_kernel_with_itself_expected);

    std::vector<double> polynomial_kernel_expected{
        905.211064176028,   2051.4341548275343, 1147.6584306405023, 3376.293391864469,
        1055.8374555060823, 2422.491929696561,  516.8304082180927,  1961.3932709015874,
        1234.6279022558322, 2737.622918688313,  619.9864281385471,  1794.0121704874616};

    param.polynomial_kernel_expected =
        convert_vector<double, T>(polynomial_kernel_expected);

    std::vector<double> polynomial_kernel_with_itself_expected{
        1433.9606608842407, 1912.8758216821034, 1508.0495328968277, 978.1693068383559,
        1859.610013298783,  1144.4324370625866, 1912.8758216821034, 3724.712962815662,
        2669.6907306661437, 1910.8294442750512, 2728.9740848038855, 2109.613726299531,
        1508.0495328968277, 2669.6907306661437, 2454.422709199248,  968.735008989019,
        2733.2955695598503, 1397.058636636973,  978.1693068383559,  1910.8294442750512,
        968.735008989019,   2352.0826623352054, 801.8844692274317,  1354.1916431552233,
        1859.610013298783,  2728.9740848038855, 2733.2955695598503, 801.8844692274317,
        3568.9084075311252, 1394.4542208427795, 1144.4324370625866, 2109.613726299531,
        1397.058636636973,  1354.1916431552233, 1394.4542208427795, 1626.1184737921214};

    param.polynomial_kernel_with_itself_expected =
        convert_vector<double, T>(polynomial_kernel_with_itself_expected);

    std::vector<double> sigmoid_kernel_expected{
        0.989963069309474,  0.9947779605482564, 0.9916295432860046, 0.9966289794673341,
        0.9910716794832725, 0.9954717490234584, 0.984959392115766,  0.9945762530149062,
        0.9920946326248296, 0.9959315334703542, 0.9867623948326573, 0.994156229061938};

    param.sigmoid_kernel_expected = convert_vector<double, T>(sigmoid_kernel_expected);

    std::vector<double> sigmoid_kernel_with_itself_expected{
        0.9929819369796267, 0.9944609163803843, 0.9932615659984216, 0.990534454166752,
        0.9943284488286025, 0.9916111853349231, 0.9944609163803843, 0.9969192874559449,
        0.9958403934500023, 0.9944559427863934, 0.9959201490715824, 0.9949004646096297,
        0.9932615659984216, 0.9958403934500023, 0.9955229630530381, 0.990464484516954,
        0.9959258452392736, 0.9928335329623567, 0.990534454166752,  0.9944559427863934,
        0.990464484516954,  0.995354601968639,  0.989014299548725,  0.992652779451855,
        0.9943284488286025, 0.9959201490715824, 0.9959258452392736, 0.989014299548725,
        0.9967956828302027, 0.9928228138007401, 0.9916111853349231, 0.9949004646096297,
        0.9928335329623567, 0.992652779451855,  0.9928228138007401, 0.993662851810401};

    param.sigmoid_kernel_with_itself_expected =
        convert_vector<double, T>(sigmoid_kernel_with_itself_expected);

    param.expected_status = da_status_success;

    param.epsilon_polynomial = 10 * sqrt(std::numeric_limits<T>::epsilon());

    params.push_back(param);
}

template <typename T>
void GetKernelData(std::vector<KernelFunctionParamType<T>> &params) {
    GetTallData(params);
    GetFatData(params);
    GetSquareData(params);
    Get1by1Data(params);
    GetSingleRowData(params);
    GetSingleColData(params);
    GetSubarrayData(params);
    GetSubarrayRowMajorData(params);
    GetRowMajorData(params);
}

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(KernelFunctionTest, FloatTypes);

TYPED_TEST(KernelFunctionTest, KernelFunctionFunctionality) {

    std::vector<KernelFunctionParamType<TypeParam>> params;
    GetKernelData(params);
    da_int count = 0;

    for (auto &param : params) {
        count++;

        std::cout << "Functionality test " << std::to_string(count) << ": "
                  << param.test_name << std::endl;

        std::vector<TypeParam> kernel_with_y(param.kernel_size);
        std::vector<TypeParam> kernel_with_itself(param.kernel_itself_size);
        TypeParam *dummy = nullptr;
        // RBF (first between X and Y, then between X and itself)
        EXPECT_EQ(da_rbf_kernel(param.order, param.m, param.n, param.p, param.x.data(),
                                param.ldx, param.y.data(), param.ldy,
                                kernel_with_y.data(), param.ldd, param.gamma),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.kernel_size, param.rbf_kernel_expected.data(),
                        kernel_with_y.data(), param.epsilon);
        EXPECT_EQ(da_rbf_kernel(param.order, param.m, param.n, param.p, param.x.data(),
                                param.ldx, dummy, param.ldy, kernel_with_itself.data(),
                                param.ldd_itself, param.gamma),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.kernel_itself_size,
                        param.rbf_kernel_with_itself_expected.data(),
                        kernel_with_itself.data(), param.epsilon);
        // Linear (first between X and Y, then between X and itself)
        EXPECT_EQ(da_linear_kernel(param.order, param.m, param.n, param.p, param.x.data(),
                                   param.ldx, param.y.data(), param.ldy,
                                   kernel_with_y.data(), param.ldd),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.kernel_size, param.linear_kernel_expected.data(),
                        kernel_with_y.data(), param.epsilon);
        EXPECT_EQ(da_linear_kernel(param.order, param.m, param.n, param.p, param.x.data(),
                                   param.ldx, dummy, param.ldy, kernel_with_itself.data(),
                                   param.ldd_itself),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.kernel_itself_size,
                        param.linear_kernel_with_itself_expected.data(),
                        kernel_with_itself.data(), param.epsilon);
        // Polynomial (first between X and Y, then between X and itself)
        EXPECT_EQ(da_polynomial_kernel(param.order, param.m, param.n, param.p,
                                       param.x.data(), param.ldx, param.y.data(),
                                       param.ldy, kernel_with_y.data(), param.ldd,
                                       param.gamma, param.degree, param.coef0),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.kernel_size, param.polynomial_kernel_expected.data(),
                        kernel_with_y.data(), param.epsilon_polynomial);
        EXPECT_EQ(da_polynomial_kernel(param.order, param.m, param.n, param.p,
                                       param.x.data(), param.ldx, dummy, param.ldy,
                                       kernel_with_itself.data(), param.ldd_itself,
                                       param.gamma, param.degree, param.coef0),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.kernel_itself_size,
                        param.polynomial_kernel_with_itself_expected.data(),
                        kernel_with_itself.data(), param.epsilon_polynomial);
        // Sigmoid (first between X and Y, then between X and itself)
        EXPECT_EQ(da_sigmoid_kernel(param.order, param.m, param.n, param.p,
                                    param.x.data(), param.ldx, param.y.data(), param.ldy,
                                    kernel_with_y.data(), param.ldd, param.gamma,
                                    param.coef0),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.kernel_size, param.sigmoid_kernel_expected.data(),
                        kernel_with_y.data(), param.epsilon);
        EXPECT_EQ(da_sigmoid_kernel(param.order, param.m, param.n, param.p,
                                    param.x.data(), param.ldx, dummy, param.ldy,
                                    kernel_with_itself.data(), param.ldd_itself,
                                    param.gamma, param.coef0),
                  param.expected_status);
        EXPECT_ARR_NEAR(param.kernel_itself_size,
                        param.sigmoid_kernel_with_itself_expected.data(),
                        kernel_with_itself.data(), param.epsilon);
    }
}

TYPED_TEST(KernelFunctionTest, IllegalArgsKernelFunction) {

    std::vector<double> x_d{0.27, 0.58, 0.67, 0.52, 0.93, 0.13, 0.09, 0.32, 0.72};
    std::vector<TypeParam> x = convert_vector<double, TypeParam>(x_d);
    std::vector<double> y_d{0.29, 0.02, 0.18, 0.83, 0.59, 0.0};
    std::vector<TypeParam> y = convert_vector<double, TypeParam>(y_d);
    da_int m = 3, n = 2, p = 3, ldx = 3, ldy = 2, ldd = 3;
    TypeParam gamma = 0.5;
    da_int degree = 7;
    TypeParam coef0 = 2;
    std::vector<TypeParam> dummy(m * n, 0);
    TypeParam *dummy_y = nullptr;

    // Test with illegal value of gamma (strictly less than 0)
    TypeParam gamma_illegal = -0.5;
    EXPECT_EQ(da_rbf_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                            dummy.data(), ldd, gamma_illegal),
              da_status_invalid_input);
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                                   dummy.data(), ldd, gamma_illegal, degree, coef0),
              da_status_invalid_input);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                                dummy.data(), ldd, gamma_illegal, coef0),
              da_status_invalid_input);

    // Test with illegal value of degree (less than or equal to 0)
    da_int degree_illegal = 0;
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                                   dummy.data(), ldd, gamma, degree_illegal, coef0),
              da_status_invalid_input);
    degree_illegal = -1;
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                                   dummy.data(), ldd, gamma, degree_illegal, coef0),
              da_status_invalid_input);

    // Test with illegal value of ldx
    da_int ldx_illegal = 2, ldx_illegal_row = 1;
    EXPECT_EQ(da_rbf_kernel(column_major, m, n, p, x.data(), ldx_illegal, y.data(), ldy,
                            dummy.data(), ldd, gamma),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_linear_kernel(column_major, m, n, p, x.data(), ldx_illegal, dummy_y, ldy,
                               dummy.data(), ldd),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p, x.data(), ldx_illegal, y.data(),
                                   ldy, dummy.data(), ldd, gamma, degree, coef0),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m, n, p, x.data(), ldx_illegal, dummy_y,
                                ldy, dummy.data(), ldd, gamma, coef0),
              da_status_invalid_leading_dimension);
    // Row-major
    EXPECT_EQ(da_rbf_kernel(row_major, m, n, p, x.data(), ldx_illegal_row, y.data(), ldy,
                            dummy.data(), ldd, gamma),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_linear_kernel(row_major, m, n, p, x.data(), ldx_illegal_row, dummy_y,
                               ldy, dummy.data(), ldd),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_polynomial_kernel(row_major, m, n, p, x.data(), ldx_illegal_row,
                                   y.data(), ldy, dummy.data(), ldd, gamma, degree,
                                   coef0),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_sigmoid_kernel(row_major, m, n, p, x.data(), ldx_illegal_row, dummy_y,
                                ldy, dummy.data(), ldd, gamma, coef0),
              da_status_invalid_leading_dimension);

    // Test with illegal value of ldd
    da_int ldd_illegal = 2, ldd_illegal_row = 1;
    EXPECT_EQ(da_rbf_kernel(column_major, m, n, p, x.data(), ldx, dummy_y, ldy,
                            dummy.data(), ldd_illegal, gamma),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_linear_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                               dummy.data(), ldd_illegal),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p, x.data(), ldx, dummy_y, ldy,
                                   dummy.data(), ldd_illegal, gamma, degree, coef0),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                                dummy.data(), ldd_illegal, gamma, coef0),
              da_status_invalid_leading_dimension);
    // Row-major
    EXPECT_EQ(da_rbf_kernel(row_major, m, n, p, x.data(), ldx, dummy_y, ldy, dummy.data(),
                            ldd_illegal_row, gamma),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_linear_kernel(row_major, m, n, p, x.data(), ldx, y.data(), ldy,
                               dummy.data(), ldd_illegal_row),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_polynomial_kernel(row_major, m, n, p, x.data(), ldx, dummy_y, ldy,
                                   dummy.data(), ldd_illegal_row, gamma, degree, coef0),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_sigmoid_kernel(row_major, m, n, p, x.data(), ldx, y.data(), ldy,
                                dummy.data(), ldd_illegal_row, gamma, coef0),
              da_status_invalid_leading_dimension);

    // Test with illegal value of ldy
    da_int ldy_illegal = 1, ldy_illegal_row = 2;
    EXPECT_EQ(da_rbf_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy_illegal,
                            dummy.data(), ldd, gamma),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_linear_kernel(column_major, m, n, p, x.data(), ldx, y.data(),
                               ldy_illegal, dummy.data(), ldd),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p, x.data(), ldx, y.data(),
                                   ldy_illegal, dummy.data(), ldd, gamma, degree, coef0),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m, n, p, x.data(), ldx, y.data(),
                                ldy_illegal, dummy.data(), ldd, gamma, coef0),
              da_status_invalid_leading_dimension);
    // Row-major
    EXPECT_EQ(da_rbf_kernel(row_major, m, n, p, x.data(), ldx, y.data(), ldy_illegal_row,
                            dummy.data(), ldd, gamma),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_linear_kernel(row_major, m, n, p, x.data(), ldx, y.data(),
                               ldy_illegal_row, dummy.data(), ldd),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_polynomial_kernel(row_major, m, n, p, x.data(), ldx, y.data(),
                                   ldy_illegal_row, dummy.data(), ldd, gamma, degree,
                                   coef0),
              da_status_invalid_leading_dimension);
    EXPECT_EQ(da_sigmoid_kernel(row_major, m, n, p, x.data(), ldx, y.data(),
                                ldy_illegal_row, dummy.data(), ldd, gamma, coef0),
              da_status_invalid_leading_dimension);

    // Test with illegal value of m
    da_int m_illegal = 0;
    EXPECT_EQ(da_rbf_kernel(column_major, m_illegal, n, p, x.data(), ldx, y.data(), ldy,
                            dummy.data(), ldd, gamma),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_linear_kernel(column_major, m_illegal, n, p, x.data(), ldx, dummy_y, ldy,
                               dummy.data(), ldd),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_polynomial_kernel(column_major, m_illegal, n, p, x.data(), ldx, y.data(),
                                   ldy, dummy.data(), ldd, gamma, degree, coef0),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m_illegal, n, p, x.data(), ldx, dummy_y,
                                ldy, dummy.data(), ldd, gamma, coef0),
              da_status_invalid_array_dimension);

    // Test with illegal value of p
    da_int p_illegal = -1;
    EXPECT_EQ(da_rbf_kernel(column_major, m, n, p_illegal, x.data(), ldx, dummy_y, ldy,
                            dummy.data(), ldd, gamma),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_linear_kernel(column_major, m, n, p_illegal, x.data(), ldx, y.data(),
                               ldy, dummy.data(), ldd),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p_illegal, x.data(), ldx, dummy_y,
                                   ldy, dummy.data(), ldd, gamma, degree, coef0),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m, n, p_illegal, x.data(), ldx, y.data(),
                                ldy, dummy.data(), ldd, gamma, coef0),
              da_status_invalid_array_dimension);

    // Test with illegal value of n
    da_int n_illegal = -2;
    EXPECT_EQ(da_rbf_kernel(column_major, m, n_illegal, p, x.data(), ldx, y.data(), ldy,
                            dummy.data(), ldd, gamma),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_linear_kernel(column_major, m, n_illegal, p, x.data(), ldx, y.data(),
                               ldy, dummy.data(), ldd),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n_illegal, p, x.data(), ldx, y.data(),
                                   ldy, dummy.data(), ldd, gamma, degree, coef0),
              da_status_invalid_array_dimension);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m, n_illegal, p, x.data(), ldx, y.data(),
                                ldy, dummy.data(), ldd, gamma, coef0),
              da_status_invalid_array_dimension);

    // Test with invalid pointer X
    TypeParam *invalid_x = nullptr;
    EXPECT_EQ(da_rbf_kernel(column_major, m, n, p, invalid_x, ldx, dummy_y, ldy,
                            dummy.data(), ldd, gamma),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linear_kernel(column_major, m, n, p, invalid_x, ldx, y.data(), ldy,
                               dummy.data(), ldd),
              da_status_invalid_pointer);
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p, invalid_x, ldx, dummy_y, ldy,
                                   dummy.data(), ldd, gamma, degree, coef0),
              da_status_invalid_pointer);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m, n, p, invalid_x, ldx, y.data(), ldy,
                                dummy.data(), ldd, gamma, coef0),
              da_status_invalid_pointer);

    // Test with invalid pointer D
    TypeParam *invalid_d = nullptr;
    EXPECT_EQ(da_rbf_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                            invalid_d, ldd, gamma),
              da_status_invalid_pointer);
    EXPECT_EQ(da_linear_kernel(column_major, m, n, p, x.data(), ldx, dummy_y, ldy,
                               invalid_d, ldd),
              da_status_invalid_pointer);
    EXPECT_EQ(da_polynomial_kernel(column_major, m, n, p, x.data(), ldx, y.data(), ldy,
                                   invalid_d, ldd, gamma, degree, coef0),
              da_status_invalid_pointer);
    EXPECT_EQ(da_sigmoid_kernel(column_major, m, n, p, x.data(), ldx, dummy_y, ldy,
                                invalid_d, ldd, gamma, coef0),
              da_status_invalid_pointer);
}