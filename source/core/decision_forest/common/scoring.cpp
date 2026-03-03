/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "scoring.hpp"

#include <algorithm>
#include <cmath>

namespace ARCH {

namespace da_decision_forest {
template <class T>
using score_fun_t = typename std::function<T(da_int, da_int, std::vector<da_int> &)>;

template <class T>
T gini_score(da_int n_samples, da_int n_class, std::vector<da_int> &count_classes) {
    T score = 0.0;
    for (da_int c = 0; c < n_class; c++) {
        score += (T)count_classes[c] * (T)count_classes[c];
    }
    score = (T)1.0 - score / ((T)n_samples * (T)n_samples);
    return score;
}

template <class T>
T entropy_score(da_int n_samples, da_int n_class, std::vector<da_int> &count_classes) {
    T score = 0.0;
    for (da_int c = 0; c < n_class; c++) {
        T prob_c = (T)count_classes[c] / (T)n_samples;
        if (prob_c > (T)1.0e-5)
            score -= prob_c * std::log2(prob_c);
    }
    return score;
}

template <class T>
T misclassification_score(da_int n_samples, [[maybe_unused]] da_int n_class,
                          std::vector<da_int> &count_classes) {
    T score =
        (T)1.0 -
        ((T)*std::max_element(count_classes.begin(), count_classes.end())) / (T)n_samples;
    return score;
}

template double gini_score<double>(da_int n_samples, da_int n_class,
                                   std::vector<da_int> &count_classes);

template float gini_score<float>(da_int n_samples, da_int n_class,
                                 std::vector<da_int> &count_classes);

template double entropy_score<double>(da_int n_samples, da_int n_class,
                                      std::vector<da_int> &count_classes);

template float entropy_score<float>(da_int n_samples, da_int n_class,
                                    std::vector<da_int> &count_classes);

template double misclassification_score<double>(da_int n_samples,
                                                [[maybe_unused]] da_int n_class,
                                                std::vector<da_int> &count_classes);

template float misclassification_score<float>(da_int n_samples,
                                              [[maybe_unused]] da_int n_class,
                                              std::vector<da_int> &count_classes);
} // namespace da_decision_forest
} // namespace ARCH
