/*
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DA_LOGGER_HPP
#define DA_LOGGER_HPP

#include "aoclda_types.h"

#include <ios>
#include <iostream>
#include <ostream>

namespace da_logger {

class logger {
  public:
    explicit logger(std::ostream &stream = std::cout)
        : stream_(stream), state_(stream.rdbuf()) {
        state_.copyfmt(stream_);
    }

    ~logger() noexcept { reset(); }

    logger(const logger &) = delete;
    logger &operator=(const logger &) = delete;
    logger(logger &&) = delete;
    logger &operator=(logger &&) = delete;

    template <typename T> logger &operator<<(const T &value) {
        if constexpr (std::is_same_v<T, _Float16>) {
            stream_ << static_cast<float>(value);
        } else {
            stream_ << value;
        }
        return *this;
    }

    logger &operator<<(std::ostream &(*manipulator)(std::ostream &)) {
        stream_ << manipulator;
        return *this;
    }

    logger &operator<<(std::ios_base &(*manipulator)(std::ios_base &)) {
        stream_ << manipulator;
        return *this;
    }

    std::ostream &stream() { return stream_; }

    void reset() noexcept {
        try {
            stream_.copyfmt(state_);
        } catch (...) {
        }
    }

  private:
    std::ostream &stream_;
    std::ios state_;
};

} // namespace da_logger

#endif // DA_LOGGER_HPP
