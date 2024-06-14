/*
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DA_ERROR_HPP
#define DA_ERROR_HPP

#include "aoclda_error.h"
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <string>
#include <vector>

/** Generating and storing error messages
 *
 * Generating errors
 * -----------------
 * To generate an error for a function that returns da_status, use MACRO da_error():
 *
 *     return da_error(e, status, msg);
 *
 * where
 *    e       is the da_error_t stored in a handle
 *    status  is a da_status error code
 *    msg     free string to be composed. The suggested format is:
 *            "Error message, reason, possible fix." This should be as explicit as
 *            possible and suggest any resolutions
 *
 * If a warning is to be returned then use
 *
 *     return da_warn(e, status, msg);
 *
 * with the same meaning as for error. There is actually NO difference between error
 * and warning except when composing the message. These two macros reset the error stack
 * before adding the new error entry. If you need to keep the existing error stack and
 * just add the new error, the next parragraph shows how to do this.
 *
 * In most cases, a function needs to handle the return status of a called function which
 * has already setup the error trace using the previously described da_error() and da_warn().
 * Under these circunstances, it is desirable to stack the next error or warning on top of the
 * existing one in such a way to construct an "error trace". This can be done by calling
 *
 *     return da_error_trace(e, status, msg);
 *  or
 *     return da_warn_trace(e, status, msg);
 *
 * with the same meaning as above. The main difference will be that these two macros
 * stack the errors and the printing method will pretty-print the error-stack and show
 * the error trace.
 *
 * Recommended usage
 * -----------------
 *
 *  => Any function that generates an error or warning should use da_error()/da_warn().
 *     This guarantees that the stack is reset before adding a new error.
 *
 *  => Any function that needs to handle errors returned by a called function should
 *     use da_error_trace()/da_warn_trace(). This will stack the new error on to the
 *     stack without resetting it.
 *
 * Methods for printing the error message
 * --------------------------------------
 *
 * 1.- print() composes the message and prints to std::cout
 * 2.- print(str) composes the message and returns it in std::string str
 * 3.- print(ss) composes the message and pipes it to std::stringstream ss
 *
 * If no error or warning was registered then a friendly banner is produced:
 * "Last operation was successful."
 *
 * The public API to print the error is always
 *     da_handle_print_error_message(handle) which simply calles e->print();
 *
 * Note: this API prints to std out and not std err.
 *
 * More details
 * ------------
 *
 * For Debug builds VERBOSE_ERROR is set (by default in CMakeLists.txt) and the
 * error is printed immediatly as it is registered in the error structure.
 * This as many benefits during development of new functionalities, mainly that
 * the trace build-up can be observed in the console.
 *
 * The trace stack size is limited to a hard-coded constant of 10 levels 0 to 9. It
 * is assumed that a stack trace deeper than 4-5 levels is an indication of a poor
 * design or miss-use of this error trace utility. As such, the stack will
 * replace the last error level with a message indicating the trace is "full."
 *
 * It is possible to "record" the error/warning directly, without using the
 * MACRO functions da_error() or da_warn(). By calling the "rec()" method.
 * This method provides full control on the telemetry and texts used to compose
 * the banner.
 * Instead of using da_error(e, status, msg), it is possible to call
 * return e.rec(da_status status, string msg, string det = "",
 *              string tel = "<no telemetry provided>", size_t ln = 0,
 *              severity_type sev = DA_ERROR);
 * Here msg is the same as for MACROS da_error() and da_warn(), while det is a string
 * that contains further details about the error and possible fixes, it is a free
 * form string that if it is not empty then it is printed under a "details:" section
 * on its own.  tel is a string to contain the telemetry data, this generally is an
 * abbreviated form of __FILE__, but need not be. ln contains the line number and is
 * generally __LINE__.
 * sev is the level of severity use as follow, DA_WARNING is a "error" but for which
 * the returned data is potentially usable, think of returning a suboptimal solution
 * to an optimization problem with poor conditioning. DA_ERROR indicates that the
 * output variables should not be relied on.
 *
 * Actions to be taken when recording
 * It is possible to specify in the constructor what action to take
 * DA_RECORD - will compare/print the error
 * DA_ABORT - compose/print and then call std::abort() this will provide a trace
 *            and within the debugger it will be possible to navigate to the line
 *            calling the rec() method
 * DA_THROW - compose/print and throw a std::runtime_error exception with what=mesg
 */

namespace da_errors {
using namespace std;

/** Depth of the stack trace
 * General usage depth should be no more than 4 or 5. Set a generous depth by default.
 */
#define DA_ERROR_STACK_SIZE 10
#define QUOTE_(X) #X
#define QUOTE(X) QUOTE_(X)

/** Labels to use when printing the error/warning messages */
string const sev_labels[3] = {"???", "WARNING", "ERROR"};

enum action_t {
    DA_RECORD = 0, // compose/print the error/warning
    DA_ABORT,      // compose/print and call std::abort()
    DA_THROW,      // compose/print and throw std::run_time_error(msg) exception
};

class da_error_t {

    // Main error message
    vector<string> mesg;
    // Some friendy message and possible fix suggestions
    vector<string> details;
    // Telemetry
    vector<string> telem;
    // This should indicate if the output of the functionality is
    // usable or not. DA_WARNING -> useable, and DA_ERROR -> not usable.
    vector<da_severity> severity; // {severity_type::DA_NOTSET};
    // Registered status, by default no error
    vector<da_status> status; // {da_status_success};
    // Action to take when rec() method is called
    action_t action{action_t::DA_RECORD};

  public:
    da_severity get_severity(void) {
        return severity.empty() ? DA_NOTSET : severity.front();
    };
    da_status get_status(void) {
        return status.empty() ? da_status_success : status.front();
    };
    string get_mesg(void) { return mesg.empty() ? "" : mesg.front(); };
    string get_details(void) { return details.empty() ? "" : details.front(); };
    string get_telem(void) { return telem.empty() ? "" : telem.front(); };

    da_status get_mesg_char(char **message) {
        size_t len = mesg.front().length();
        *message = (char *)(malloc(len + 1));
        if (*message == NULL) {
            return da_status_memory_error; // LCOV_EXCL_LINE
        }
        /* Most of the time MSVC compiler can automatically replace CRT functions with _s versions, but not this one */
#if defined(_MSC_VER)
        strncpy_s(*message, 1 + len, mesg.front().c_str(), len);
#else
        strncpy(*message, mesg.front().c_str(), len);
#endif
        // Ensure null termination
        (*message)[len] = '\0';
        return da_status_success;
    }

    da_status clear(void) {
        mesg.clear();
        details.clear();
        telem.clear();
        status.clear();
        severity.clear();
        return da_status_success;
    }

    da_error_t(enum action_t action) : action(action) {
        mesg.reserve(DA_ERROR_STACK_SIZE);
        details.reserve(DA_ERROR_STACK_SIZE);
        telem.reserve(DA_ERROR_STACK_SIZE);
        status.reserve(DA_ERROR_STACK_SIZE);
        severity.reserve(DA_ERROR_STACK_SIZE);
    };

    // da_error_t(enum action_t action) = action_t::DA_RECORD) : action(action){};
    ~da_error_t(){};
    // Build banner
    void print(stringstream &ss) {
        string tab = "";
        if (!status.empty()) {
            for (size_t trace = 0; trace < mesg.size(); trace++) {
                if (mesg.size() > 1) {
                    // There is a stack trace, add level indicator
                    if (trace == 0) {
                        tab = "   ";
                        ss << "Error stack trace:" << std::endl;
                    }
                    ss << trace << ": ";
                }
                ss << std::setw(7) << std::left << sev_labels[severity[trace]]
                   << " (Status: " << std::setw(5) << std::right << status[trace] << ") ";
                ss.unsetf(std::ios_base::adjustfield);
                if (telem[trace] != "")
                    ss << telem[trace] << ": ";
                ss << mesg[trace] << std::endl;
                if (details[trace] != "") {
                    ss << tab << "details:" << std::endl;
                    ss << details[trace] << std::endl;
                }
            }
        } else {
            ss << "Last operation was successful." << std::endl;
        }
    }
    // Return a string with the banner
    void print(string &str) {
        stringstream ss;
        print(ss);
        str = ss.str();
    }
    // Push banner to std error
    void print(void) {
        stringstream ss;
        print(ss);
        std::cerr << ss.str();
    }
    da_status rec(da_status status, string msg, string det = "",
                  string tel = "<no telemetry provided>", size_t ln = 0,
                  da_severity sev = DA_ERROR, bool stack = false) {
        if (!stack) {
            this->status.resize(0);
            this->mesg.resize(0);
            this->details.resize(0);
            this->telem.resize(0);
            this->severity.resize(0);
        }
        size_t size = this->status.size();
        if (size < DA_ERROR_STACK_SIZE - 1) {
            this->status.push_back(status);
            this->mesg.push_back(msg);
            this->details.push_back(det);
            this->telem.push_back(tel + to_string(ln));
            this->severity.push_back(sev);
        } else if (size == DA_ERROR_STACK_SIZE - 1) {
            // Stack almost full, add a generic message
            this->status.push_back(da_status_internal_error);
            this->mesg.push_back(
                "Too many errors where registered, storing the first " QUOTE(
                    DA_ERROR_STACK_SIZE));
            this->details.push_back("");
            this->telem.push_back("");
            this->severity.push_back(DA_ERROR);

        } // else stack "full ignore request to store

#ifdef VERBOSE_ERROR
        this->print();
#endif
        if (this->action == action_t::DA_ABORT)
            std::abort();
        else if (this->action == action_t::DA_THROW)
            std::runtime_error(this->get_mesg());
        return status;
    };
};

constexpr int32_t strip_path(const char *const path, const int32_t pos = 0,
                             const int32_t pos_separator = -1) {
    return path[pos]
               ? (path[pos] == '/'
                      ? strip_path(path, pos + static_cast<int32_t>(1), pos)
                      : strip_path(path, pos + static_cast<int32_t>(1), pos_separator))
               : (pos_separator + static_cast<int32_t>(1));
}

// This strips the path from the string PATH at compile time, that is
// it provides a new starting point for the PATH string where the filename
// starts. Does the same as "basename file"
constexpr const char *basename(const char *const path) { return &path[strip_path(path)]; }

#define da_error(e, status, msg)                                                         \
    (e)->rec(status, (msg), "", std::string(da_errors::basename(__FILE__)) + ":",        \
             __LINE__, DA_ERROR, false)
#define da_warn(e, status, msg)                                                          \
    (e)->rec(status, (msg), "",                                                          \
             std::string(da_errors::basename(__FILE__)) + std::string(":"), __LINE__,    \
             DA_WARNING, false)

#define da_error_trace(e, status, msg)                                                   \
    (e)->rec(status, (msg), "",                                                          \
             std::string(da_errors::basename(__FILE__)) + std::string(":"), __LINE__,    \
             DA_ERROR, true)
#define da_warn_trace(e, status, msg)                                                    \
    (e)->rec(status, (msg), "",                                                          \
             std::string(da_errors::basename(__FILE__)) + std::string(":"), __LINE__,    \
             DA_WARNING, true)

#define da_error_bypass(e, status, msg)                                                  \
    ((e) != nullptr                                                                      \
         ? (e)->rec(status, (msg), "", std::string(da_errors::basename(__FILE__)) + ":", \
                    __LINE__, DA_ERROR, false)                                           \
         : status)
#define da_warn_bypass(e, status, msg)                                                   \
    ((e) != nullptr                                                                      \
         ? (e)->rec(status, (msg), "",                                                   \
                    std::string(da_errors::basename(__FILE__)) + std::string(":"),       \
                    __LINE__, DA_WARNING, false)                                         \
         : status)
} // namespace da_errors
#endif
