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
 */

// Unit tests for the _Float16 cblas_syrk overload in fp16_helpers.hpp.
//
// The production implementation is a recursive blocked SYRK: at each level
// the n-by-n diagonal triangles are computed by two recursive _Float16
// SYRK calls and the off-diagonal block by a single _Float16 GEMM (which
// on Linux uses aocl_gemm_f16f16f16of16). Leaf triangles (n <= 32) fall
// back to a pack-cast-ssyrk-cast path.
//
// We compare against a float reference obtained by casting the _Float16
// inputs to float and calling cblas_ssyrk.
//
// The _Float16 GEMM backing the recursive step (aocl_gemm_f16f16f16of16)
// silently returns without writing C on processors that lack AVX-512-FP16.
// The recursive block decomposition is therefore only exercised on zen6
// or higher; on earlier architectures only the n <= 32 leaf path is
// well-defined. We restrict the tests to zen6+ at runtime.

// Emit our own copy of the FP16 half<->single/double compiler-rt builtins
// (__truncsfhf2 / __truncdfhf2 / __extendhfsf2). This test's own _Float16 casts
// reference them, but the copies in the aocl-da library have hidden ELF
// visibility and are not exported from the shared library, so a toolchain whose
// runtime lacks these builtins (e.g. AOCC) cannot otherwise resolve them.
#define DA_FP16_DEFINE_BUILTINS

#include "../datests_cblas.hh"
#include "../utest_utils.hpp"
#include "aoclda.h"
#include "aoclda_utils.h"
#include "fp16_helpers.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

bool host_supports_fp16_gemm() {
    da_int len = 100;
    char arch[100], ns[100];
    if (da_get_arch_info(&len, arch, ns) != da_status_success)
        return false;
    // aocl_gemm_f16f16f16of16 requires AVX-512-FP16 (zen6 or later).
    return std::strcmp(arch, "zen6") == 0;
}

struct SyrkCase {
    da_int n;
    da_int k;
    CBLAS_ORDER layout;
    CBLAS_UPLO uplo;
    CBLAS_TRANSPOSE trans;
    float alpha;
    float beta;
    unsigned seed;

    std::string describe() const {
        std::string s;
        s += "n=" + std::to_string(n);
        s += " k=" + std::to_string(k);
        s += " layout=" + std::string(layout == CblasColMajor ? "col" : "row");
        s += " uplo=" + std::string(uplo == CblasUpper ? "U" : "L");
        s += " trans=" + std::string(trans == CblasNoTrans ? "N" : "T");
        s += " alpha=" + std::to_string(alpha);
        s += " beta=" + std::to_string(beta);
        return s;
    }
};

// Convert a logical (rows, cols) matrix between layouts using user-supplied
// leading dimension `ld`. Storage is column-major when layout==CblasColMajor.
inline size_t idx(CBLAS_ORDER layout, da_int i, da_int j, da_int ld) {
    return (layout == CblasColMajor)
               ? static_cast<size_t>(i) + static_cast<size_t>(j) * ld
               : static_cast<size_t>(i) * ld + static_cast<size_t>(j);
}

// Reference SYRK: cast _Float16 -> float (no rounding loss for ssyrk's
// internal accumulation since inputs already fit in fp16) and call
// cblas_ssyrk on a tightly-packed buffer with the same leading dim.
void reference_syrk(const SyrkCase &c, const std::vector<_Float16> &A, da_int lda,
                    const std::vector<_Float16> &C_in, da_int ldc,
                    std::vector<_Float16> &C_out) {
    std::vector<float> Af(A.size());
    std::vector<float> Cf(C_in.size());
    for (size_t i = 0; i < A.size(); ++i)
        Af[i] = static_cast<float>(A[i]);
    for (size_t i = 0; i < C_in.size(); ++i)
        Cf[i] = static_cast<float>(C_in[i]);
    datest_blas::cblas_syrk(c.layout, c.uplo, c.trans, c.n, c.k, c.alpha, Af.data(), lda,
                            c.beta, Cf.data(), ldc);
    C_out.assign(C_in.begin(), C_in.end());
    for (da_int j = 0; j < c.n; ++j) {
        for (da_int i = 0; i < c.n; ++i) {
            bool in_tri = (c.uplo == CblasUpper) ? (i <= j) : (i >= j);
            if (!in_tri)
                continue;
            const size_t idxC = idx(c.layout, i, j, ldc);
            C_out[idxC] = static_cast<_Float16>(Cf[idxC]);
        }
    }
}

// Run one parameter combination and compare against the float reference.
void run_one(const SyrkCase &c) {
    SCOPED_TRACE(c.describe());
    const da_int a_rows = (c.trans == CblasNoTrans) ? c.n : c.k;
    const da_int a_cols = (c.trans == CblasNoTrans) ? c.k : c.n;
    // Pad leading dimensions to ensure the implementation respects them.
    const da_int lda_pad = 3;
    const da_int ldc_pad = 5;
    const da_int lda = ((c.layout == CblasColMajor) ? a_rows : a_cols) + lda_pad;
    const da_int ldc = c.n + ldc_pad;
    const size_t Asz = static_cast<size_t>(lda) *
                       static_cast<size_t>((c.layout == CblasColMajor) ? a_cols : a_rows);
    const size_t Csz = static_cast<size_t>(ldc) * static_cast<size_t>(c.n);

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<_Float16> A(Asz), C_initial(Csz);
    for (size_t i = 0; i < Asz; ++i)
        A[i] = static_cast<_Float16>(dist(rng));
    for (size_t i = 0; i < Csz; ++i)
        C_initial[i] = static_cast<_Float16>(dist(rng));

    // Compute reference using cblas_ssyrk on a float cast of the inputs.
    std::vector<_Float16> C_ref;
    reference_syrk(c, A, lda, C_initial, ldc, C_ref);

    // Compute via the production _Float16 overload.
    std::vector<_Float16> C_actual = C_initial;
    da_blas::cblas_syrk(c.layout, c.uplo, c.trans, c.n, c.k,
                        static_cast<_Float16>(c.alpha), A.data(), lda,
                        static_cast<_Float16>(c.beta), C_actual.data(), ldc);

    // Tolerance: half-precision epsilon ~ 9.8e-4. The reference path
    // performs identical pack-cast-ssyrk math for n <= 32, so for leaf
    // cases the result should match exactly. For larger n the recursive
    // path uses one fp16 GEMM per off-diagonal block which introduces a
    // small additional rounding error on the off-diagonal entries.
    const float tol_abs = 5e-2f * (1.0f + std::sqrt(static_cast<float>(c.k)));
    const float tol_rel = 5e-2f;
    for (da_int j = 0; j < c.n; ++j) {
        for (da_int i = 0; i < c.n; ++i) {
            bool in_tri = (c.uplo == CblasUpper) ? (i <= j) : (i >= j);
            if (!in_tri)
                continue;
            const size_t k = idx(c.layout, i, j, ldc);
            const float got = static_cast<float>(C_actual[k]);
            const float ref = static_cast<float>(C_ref[k]);
            const float diff = std::fabs(got - ref);
            const float rel = diff / (std::fabs(ref) + 1e-3f);
            EXPECT_TRUE(diff <= tol_abs || rel <= tol_rel)
                << "Mismatch at (i=" << i << ", j=" << j << "): got=" << got
                << " ref=" << ref << " abs=" << diff << " rel=" << rel
                << " (tol_abs=" << tol_abs << ", tol_rel=" << tol_rel << ")";
        }
    }
}

// Sweep of sizes that exercise:
//   - odd and even n
//   - n below, at, and above the recursion threshold (32)
//   - k smaller, equal, and larger than the leaf threshold
std::vector<SyrkCase> build_cases() {
    std::vector<SyrkCase> cases;
    const std::vector<da_int> ns = {1, 7, 8, 16, 31, 32, 33, 47, 64, 65, 100, 128};
    const std::vector<da_int> ks = {1, 5, 16, 32, 100};
    const std::vector<float> alphas = {1.0f, -0.5f};
    const std::vector<float> betas = {0.0f, 1.0f};
    const std::vector<CBLAS_ORDER> layouts = {CblasColMajor, CblasRowMajor};
    const std::vector<CBLAS_UPLO> uplos = {CblasUpper, CblasLower};
    const std::vector<CBLAS_TRANSPOSE> transs = {CblasNoTrans, CblasTrans};
    unsigned seed = 0xC0FFEEu;
    for (auto layout : layouts)
        for (auto uplo : uplos)
            for (auto trans : transs)
                for (auto n : ns)
                    for (auto k : ks)
                        for (auto alpha : alphas)
                            for (auto beta : betas)
                                cases.push_back(
                                    {n, k, layout, uplo, trans, alpha, beta, seed++});
    return cases;
}

} // namespace

class FP16SyrkTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!host_supports_fp16_gemm()) {
            GTEST_SKIP() << "Host CPU lacks AVX-512-FP16 (arch != zen6); the "
                            "production _Float16 GEMM used by the recursive SYRK "
                            "is unavailable on this machine.";
        }
    }
};

TEST_F(FP16SyrkTest, MatchesFloatReferenceAcrossSizes) {
    for (const auto &c : build_cases())
        run_one(c);
}

// Sanity check: when n <= 32 the implementation should hit only the leaf
// path (pack + cast + ssyrk + cast). This case is exercised on every host
// because it does not call the FP16 GEMM. We re-enable it independently
// of the SetUp() skip via a separate fixture.
class FP16SyrkLeafTest : public ::testing::Test {};

TEST_F(FP16SyrkLeafTest, LeafPathMatchesFloatReference) {
    const std::vector<da_int> ns = {1, 2, 8, 16, 31, 32};
    const std::vector<da_int> ks = {1, 5, 32};
    unsigned seed = 0xBEEFu;
    for (auto n : ns)
        for (auto k : ks)
            for (auto layout : {CblasColMajor, CblasRowMajor})
                for (auto uplo : {CblasUpper, CblasLower})
                    for (auto trans : {CblasNoTrans, CblasTrans})
                        run_one({n, k, layout, uplo, trans, 1.0f, 0.5f, seed++});
}
