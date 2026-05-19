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

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <direct.h>
#define DA_MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define DA_MKDIR(dir) mkdir(dir, 0755)
#endif

namespace model_persistence_test_utils {

// Recursively create directories (equivalent to mkdir -p)
inline bool create_directories(const std::string &path) {
    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
        current += path[i];
        if (path[i] == '/' || path[i] == '\\' || i == path.size() - 1) {
            struct stat st;
            if (stat(current.c_str(), &st) != 0) {
                if (DA_MKDIR(current.c_str()) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Thread-safe function to get and create test file directory
inline std::string get_test_file_dir() {
    static std::once_flag flag;
    static std::string test_dir;

    std::call_once(flag, []() {
#ifdef TEST_OUTPUT_DIR
        test_dir = TEST_OUTPUT_DIR;
#else
        // Fallback to current directory if TEST_OUTPUT_DIR not defined
        test_dir = ".";
#endif
        // Create directory if it doesn't exist
        struct stat st;
        if (stat(test_dir.c_str(), &st) != 0) {
            if (!create_directories(test_dir)) {
                // If directory creation fails, fall back to current directory
                test_dir = ".";
            }
        }
    });

    return test_dir;
}

} // namespace model_persistence_test_utils

#endif // TEST_UTILS_HPP
