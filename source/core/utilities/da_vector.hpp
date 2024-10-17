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

#ifndef DA_VECTOR_HPP
#define DA_VECTOR_HPP

#include "aoclda.h"
#include <stdexcept>

#define INIT_CAPACITY 64

namespace da_vector {

/*
  Lightweight implementation of a vector-like class for basic data types, to enable libmem to be
  exploited where appropriate and to run quickly without any bounds checks etc and to use malloc/free
  for a bit more speed when a lot of allocations are required and we know there are no class
  constructors/destructors to call.
 */

template <typename T> class da_vector {
  public:
    size_t size() const { return _size; }
    size_t capacity() const { return _capacity; }
    T *data() { return _data; }
    const T *data() const { return _data; }
    T &operator[](size_t i) { return _data[i]; }

    da_vector() : _data(nullptr), _size(0), _capacity(INIT_CAPACITY) {
        _data = (T *)malloc(_capacity * sizeof(T));
        if (!_data) {
            throw std::bad_alloc(); // LCOV_EXCL_LINE
        }
    }

    da_vector(size_t size) : _data(nullptr), _size(size), _capacity(INIT_CAPACITY) {
        while (_capacity < _size)
            _capacity <<= 1;
        _data = (T *)malloc(_capacity * sizeof(T));
        if (!_data) {
            throw std::bad_alloc(); // LCOV_EXCL_LINE
        }
    }

    ~da_vector() {
        if (_data)
            free(_data);
        _data = nullptr;
    }

    void push_back(const T &val) {
        if (_size == _capacity) {
            _capacity <<= 1;
            T *new_data = (T *)malloc(_capacity * sizeof(T));
            if (!new_data) {
                throw std::bad_alloc(); // LCOV_EXCL_LINE
            }
            memcpy(new_data, _data, _size * sizeof(T));
            free(_data);
            _data = new_data;
        }
        _data[_size++] = val;
    }

    // Append the values of vec onto the end of the vector
    void append(const da_vector<T> &vec) {
        if (_size + vec.size() > _capacity) {
            while (_size + vec.size() > _capacity)
                _capacity <<= 1;
            T *new_data = (T *)malloc(_capacity * sizeof(T));
            if (!new_data) {
                throw std::bad_alloc(); // LCOV_EXCL_LINE
            }
            memcpy(new_data, _data, _size * sizeof(T));
            free(_data);
            _data = new_data;
        }
        memcpy(_data + _size, vec.data(), vec.size() * sizeof(T));
        _size += vec.size();
    }

    void append(const std::vector<T> &vec) {
        if (_size + vec.size() > _capacity) {
            while (_size + vec.size() > _capacity)
                _capacity <<= 1;
            T *new_data = (T *)malloc(_capacity * sizeof(T));
            if (!new_data) {
                throw std::bad_alloc(); // LCOV_EXCL_LINE
            }
            memcpy(new_data, _data, _size * sizeof(T));
            free(_data);
            _data = new_data;
        }
        memcpy(_data + _size, vec.data(), vec.size() * sizeof(T));
        _size += vec.size();
    }

  private:
    T *_data;
    size_t _size;
    size_t _capacity;
};

} // namespace da_vector

#endif
