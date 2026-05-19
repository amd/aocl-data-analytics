/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef GENERATE_TEST_DATA_HPP
#define GENERATE_TEST_DATA_HPP

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

/*
 * Custom file generation that mimics the saved types of the serialization process.
 * Changes to this file must be done only if change of types in serialization has been done.
*/

namespace test_data_generator {

// Generate test binary data to a file
inline bool generate_kernel_data(const std::string &path) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    bool ok = true;

    // Helper lambda to write any trivial type with error tracking
    auto write_val = [&out, &ok](auto val) {
        if (!ok)
            return;
        out.write(reinterpret_cast<const char *>(&val), sizeof(val));
        if (!out.good())
            ok = false;
    };

    // Helper lambda to write raw bytes with error tracking
    auto write_bytes = [&out, &ok](const char *data, size_t len) {
        if (!ok)
            return;
        out.write(data, len);
        if (!out.good())
            ok = false;
    };

    // === SCALARS ===

    // bool
    uint8_t bol = 1;
    write_val(bol);

    // da_int
    int64_t da_in = 42;
    write_val(da_in);

    // float
    float fl = 127.9f;
    write_val(fl);

    // double
    double doubl = 83.22;
    write_val(doubl);

    // std::string = "test"
    std::string str = "test";
    int64_t str_len = static_cast<int64_t>(str.size());
    write_val(str_len);
    write_bytes(str.data(), str.size());

    // === std::vector<T> ===

    // std::vector<da_int> = {88, -72, 0}
    int64_t v_da_int_len = 3;
    write_val(v_da_int_len);
    write_val(int64_t(88));
    write_val(int64_t(-72));
    write_val(int64_t(0));

    // std::vector<float> = {27.3f, -77.9f}
    int64_t v_fl_len = 2;
    write_val(v_fl_len);
    write_val(float(27.3f));
    write_val(float(-77.9f));

    // std::vector<double> = {33.3, -66.9, -250.0, -89.1}
    int64_t v_doubl_len = 4;
    write_val(v_doubl_len);
    write_val(double(33.3));
    write_val(double(-66.9));
    write_val(double(-250.0));
    write_val(double(-89.1));

    // === da_vector::da_vector<T> (same binary format as std::vector) ===

    // da_vector<da_int> = {100, -200}
    int64_t dv_int_len = 2;
    write_val(dv_int_len);
    write_val(int64_t(100));
    write_val(int64_t(-200));

    // da_vector<float> = {1.5f, 2.5f, 3.5f}
    int64_t dv_fl_len = 3;
    write_val(dv_fl_len);
    write_val(float(1.5f));
    write_val(float(2.5f));
    write_val(float(3.5f));

    // da_vector<double> = {10.1}
    int64_t dv_doubl_len = 1;
    write_val(dv_doubl_len);
    write_val(double(10.1));

    // === std::vector<da_vector::da_vector<T>> (nested) ===

    // std::vector<da_vector<da_int>> = {{1, 2}, {3}, {4, 5, 6, 7}}
    int64_t vdv_int_outer = 3;
    write_val(vdv_int_outer);
    // inner[0] = {1, 2}
    write_val(int64_t(2));
    write_val(int64_t(1));
    write_val(int64_t(2));
    // inner[1] = {3}
    write_val(int64_t(1));
    write_val(int64_t(3));
    // inner[2] = {4, 5, 6, 7}
    write_val(int64_t(4));
    write_val(int64_t(4));
    write_val(int64_t(5));
    write_val(int64_t(6));
    write_val(int64_t(7));

    // std::vector<da_vector<float>> = {{1.1f}, {2.2f, 3.3f}}
    int64_t vdv_fl_outer = 2;
    write_val(vdv_fl_outer);
    // inner[0] = {1.1f}
    write_val(int64_t(1));
    write_val(float(1.1f));
    // inner[1] = {2.2f, 3.3f}
    write_val(int64_t(2));
    write_val(float(2.2f));
    write_val(float(3.3f));

    // std::vector<da_vector<double>> = {{-1.0, -2.0, -3.0}, {99.9}}
    int64_t vdv_doubl_outer = 2;
    write_val(vdv_doubl_outer);
    // inner[0] = {-1.0, -2.0, -3.0}
    write_val(int64_t(3));
    write_val(double(-1.0));
    write_val(double(-2.0));
    write_val(double(-3.0));
    // inner[1] = {99.9}
    write_val(int64_t(1));
    write_val(double(99.9));

    // === ENUMS (stored as int64_t) ===

    // da_order = column_major = 1
    write_val(int64_t(1));
    // da_svm_model = svr = 3
    write_val(int64_t(3));
    // da_metric = da_minkowski = 2
    write_val(int64_t(2));
    // linmod_model = linmod_model_undefined = 0
    write_val(int64_t(0));
    // logistic_constraint = rsc = 1
    write_val(int64_t(1));
    // split_property = categorical_onevall = 2
    write_val(int64_t(2));
    // approx_nn_metric = inner_product = 2
    write_val(int64_t(2));
    // da_handle_type = da_handle_linmod = 1
    write_val(int64_t(1));

    // === USER DATA (serialize_user_data) ===

    // 2x3 column-major float array (m=2, n=3, ldx=2)
    // Data: {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
    // vector_size = m * n = 6
    write_val(int64_t(6));
    write_val(float(1.0f));
    write_val(float(2.0f));
    write_val(float(3.0f));
    write_val(float(4.0f));
    write_val(float(5.0f));
    write_val(float(6.0f));

    // 3x2 row-major double array (m=3, n=2, ldx=2)
    // Data: {10.0, 20.0, 30.0, 40.0, 50.0, 60.0}
    // vector_size = m * n = 6
    write_val(int64_t(6));
    write_val(double(10.0));
    write_val(double(20.0));
    write_val(double(30.0));
    write_val(double(40.0));
    write_val(double(50.0));
    write_val(double(60.0));

    // nullptr case (should be size 0)
    write_val(int64_t(0));

    out.close();
    return ok;
}

} // namespace test_data_generator

#endif // GENERATE_TEST_DATA_HPP
