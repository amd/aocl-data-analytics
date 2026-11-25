/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#ifndef AOCLDA_OPTIONS_HPP_
#define AOCLDA_OPTIONS_HPP_

#include "aoclda.h"
#include "da_logger.hpp"
#include "model_persistence.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

// Remove windows macros
#undef min
#undef max

/*
 * da_options namespace provides a Registry class that "registers"
 * options and an Option class that defines an "option" element.
 * Options can be of any four classes: Integer, Real (float or double,
 * templated), Boolean or String. The registry class
 * defines a method Called "Register" to add a new option.
 * Registered options can be set using Registry.SetOption,
 * queried using Registry.GetOption and for strings Registry.GetKey, and
 * pretty-printed using either Registry.PrintOptions or Registry.print_details.
 *
 * Options have a "name" that is used to distinguish them, so this should be
 * unique among the ALL the registered options (of any Option class).
 * Furthermore, the "name" string is sanitized before using, that is, it is
 * trimmed and blanks squeezed. Do not register the same option twice.
 * No checks on option registry are done.
 *
 * Querying options have a certain cost and should be done only once at the
 * beginning of the solver or when initializing the internal data.
 *
 *
 * Register functions returns da_status status da_status_option_*
 */

namespace da_options {
using std::endl;
using std::invalid_argument;
using std::is_floating_point;
using std::is_same;
using std::ostringstream;
using std::regex;
using std::string;
using namespace std::literals::string_literals;
using std::map;
using std::shared_ptr;

const da_int max_da_int = std::numeric_limits<da_int>::max();
const da_int prn_opt_align{40};

template <typename T> struct safe_eps {
    static constexpr T eps = 2 * std::numeric_limits<T>::epsilon();
    constexpr operator T() const noexcept { return eps; };
};

template <typename T> struct safe_tol {
  private:
    // Method to represent the tolerance in \LaTeX format
    const std::string sqrt2eps{"\\sqrt{2\\,\\varepsilon}"};
    const std::string varepsilon{"\\varepsilon"};

  public:
    T mcheps(T num = (T)1, T den = (T)1) {
        return ((da_options::safe_eps<T>() * num) / den);
    };
    T safe_eps(T num = (T)1, T den = (T)1) {
        return ((std::sqrt(da_options::safe_eps<T>()) * num) / den);
    };
    T safe_inveps(T num = (T)1, T den = (T)1) {
        return (num / (den * std::sqrt(da_options::safe_eps<T>())));
    };

    std::string mcheps_latex(T num = (T)1, T den = (T)1) {
        size_t nchar;
        std::string n, d;
        n.resize(64);
        d.resize(64);
        nchar = std::snprintf(n.data(), n.size(), "%g", num);
        n.resize(nchar);
        nchar = std::snprintf(d.data(), d.size(), "%g", den);
        d.resize(nchar);
        if (num != 1 && den != 1) {
            return n + "/" + d + this->varepsilon;
        } else if (den != 1) {
            return this->varepsilon + "/" + d;
        } else if (num != 1) {
            return n + "\\;" + this->varepsilon;
        } else {
            return this->varepsilon;
        }
    };

    std::string safe_eps_latex(T num = (T)1, T den = (T)1) {
        size_t nchar;
        std::string n, d;
        n.resize(64);
        d.resize(64);
        nchar = std::snprintf(n.data(), n.size(), "%g", num);
        n.resize(nchar);
        nchar = std::snprintf(d.data(), d.size(), "%g", den);
        d.resize(nchar);
        if (num != 1 && den != 1) {
            return n + "/" + d + this->sqrt2eps;
        } else if (den != 1) {
            return this->sqrt2eps + "/" + d;
        } else if (num != 1) {
            return n + "\\;" + this->sqrt2eps;
        } else {
            return this->sqrt2eps;
        }
    };

    std::string safe_inveps_latex(T num = (T)1, T den = (T)1) {
        size_t nchar;
        std::string n, d;
        n.resize(64);
        d.resize(64);
        nchar = std::snprintf(n.data(), n.size(), "%g", num);
        n.resize(nchar);
        nchar = std::snprintf(d.data(), d.size(), "%g", den);
        d.resize(nchar);
        if (num != 1 && den != 1) {
            return "\\frac{" + n + "}{" + d + "\\;" + this->sqrt2eps + "}";
        } else if (den != 1) {
            return "\\frac{1}{" + d + "\\;" + this->sqrt2eps + "}";
        } else if (num != 1) {
            return "\\frac{" + n + "}{" + this->sqrt2eps + "}";
        } else {
            return this->sqrt2eps;
        }
    };
};

enum lbound_t { m_inf = 0, greaterthan, greaterequal };
enum ubound_t { p_inf = 0, lessthan, lessequal };
enum setby_t { def = 0, user = 1, solver = 2 };
#ifdef __OPTIMIZE__
const string option_tl[6] = {"?", "integer", "real", "real", "string", "boolean"};
#else
const string option_tl[6] = {"?",      "integer", "real (float)", "real (double)",
                             "string", "boolean"};
#endif
// clang-format off
enum option_t { opt_undefined = 0, opt_int = 1, opt_float = 2, opt_double = 3, opt_string = 4, opt_bool = 5 };
template <typename T> struct get_type    { };
template <> struct get_type<da_int>      { constexpr operator option_t() const noexcept { return option_t::opt_int;    } };
template <> struct get_type<da_int*>     { constexpr operator option_t() const noexcept { return option_t::opt_int;    } };
template <> struct get_type<float>       { constexpr operator option_t() const noexcept { return option_t::opt_float;  } };
template <> struct get_type<float*>      { constexpr operator option_t() const noexcept { return option_t::opt_float;  } };
template <> struct get_type<double>      { constexpr operator option_t() const noexcept { return option_t::opt_double; } };
template <> struct get_type<double*>     { constexpr operator option_t() const noexcept { return option_t::opt_double; } };
template <> struct get_type<string>      { constexpr operator option_t() const noexcept { return option_t::opt_string; } };
template <> struct get_type<const char*> { constexpr operator option_t() const noexcept { return option_t::opt_string; } };
template <> struct get_type<bool>        { constexpr operator option_t() const noexcept { return option_t::opt_bool;   } };
template <> struct get_type<bool*>       { constexpr operator option_t() const noexcept { return option_t::opt_bool;   } };
// clang-format on

using option_variant_t = std::variant<bool, da_int, float, double, std::string>;

struct OptionUtils {
    static void prep_str(string &str) {
        // This string preparation function is quite expensive,
        // make sure to call it only if really required.
        // ignore regexp exceptions,
        /* coverity[fun_call_w_exception] */
        const std::regex trim("^[[:space:]]+|[[:space:]]+$");
        const std::regex squeeze("[[:space:]]+");
        str = std::regex_replace(str, trim, std::string(""));
        str = std::regex_replace(str, squeeze, std::string(" "));
        transform(str.begin(), str.end(), str.begin(), ::tolower);
    };
};

class OptionBase {
  protected:
    bool destructive{true}; // changing the value of this option
                            // indicates that the model should be retrained.

  public:
    const string setby_l[3] = {"(default)", "(user)", "(solver)"};
    const string setby_shrt[3] = {"", "*", "**"};

    bool strict{false}; // Be strict when creating an option:
    // make sure the strings passed are checked for tidyness
    // This is HIGHLY recommended to set to true while developing new options
    // but set to false once ready because it is not cheap to use.

    virtual ~OptionBase() = default;

    da_status set_name(string &str) {
        name = str;
        if (strict) {
            OptionUtils::prep_str(name); // this is expensive: done only in strict mode
            if (name.empty()) {
                errmsg = "Invalid name (string reduced to zero-length).";
                return da_status_option_invalid_value;
            }
        }
        return da_status_success;
    }
    string get_name(void) const { return name; }
    option_t get_option_t(void) const { return otype; }
    string get_errmsg(void) const { return errmsg; }
    bool is_destructive() { return destructive; }
    void set_destructive(bool value) { destructive = value; }
    template <typename T>
    da_status validate(T lower, lbound_t lbound, T upper, ubound_t ubound, T value,
                       bool checkall = true) {
        if (checkall) {
            bool has_nan = std::numeric_limits<T>::has_quiet_NaN;
            // Check all inputs
            if (has_nan) {
                if (std::isnan(static_cast<double>(upper)) ||
                    std::isnan(static_cast<double>(lower))) {
                    errmsg =
                        "Option '" + name + "': Either lower or upper are not finite.";
                    return da_status_option_invalid_bounds;
                }
            }

            if (upper < lower && ubound != p_inf) {
                errmsg = "Option '" + name + "': Invalid bounds: lower > upper.";
                return da_status_option_invalid_bounds;
            }
            // Check bounds (special case)
            // l = u and l <  value <= u OR
            //           l <  value <  u OR
            //           l <= value <  u
            if (lower == upper && lbound != m_inf && ubound != p_inf) {
                if (!(lbound == greaterequal && ubound == lessequal)) {
                    errmsg = "Option '" + name + "': Invalid bounds.";
                    return da_status_option_invalid_bounds;
                }
            }
            if (has_nan) {
                if (std::isnan(static_cast<double>(value))) {
                    errmsg = "Option '" + name + "': Invalid value.";
                    return da_status_option_invalid_value;
                }
            }
        }

        // Quick check
        da_int iflag = 0;
        // Check that it is within range (lower bound)
        if ((lbound == greaterthan) && (value <= lower))
            iflag += 1;
        else if ((lbound == greaterequal) && (value < lower))
            iflag += 2;
        // Check that it is within range (upper bound)
        if ((ubound == lessthan) && (value >= upper))
            iflag += 10;
        else if ((ubound == lessequal) && (value > upper))
            iflag += 20;
        if (iflag) {
            errmsg = "Option '" + name + "': value out-of-bounds";
            return da_status_option_invalid_value;
        }
        return da_status_success;
    }

    // Generic getter for each option type to implement
    virtual option_variant_t get(void) const = 0;

    // Prepare the option key/value pair to be printed on screen.
    // Called with option "print options = yes"
    virtual string print_option(void) const = 0;
    // Compose the option details (used for documentation)
    // Screen = true => print it in plain text pretty print
    // Screen = false => used to indicate a file format is requested
    // => doxygen = true => format is set to Doxygen
    // => doxygen = false => format is set to ReStructuredText
    virtual string print_details(bool screen = true, bool doxygen = false) const = 0;

  protected:
    // Name i.e. "Iteration Limit"
    string name;
    // Type of the option (int, real (float/double), string or bool)
    option_t otype = opt_undefined;
    string desc; // Brief description (free text)
    setby_t setby;
    string errmsg = ""; // internal error buffer
};

template <typename T> class OptionNumeric : public OptionBase {
    // Actual value of the option
    T value;
    // Default value for option
    T vdefault;
    // Descriptive string of the vdefault value (optional)
    string vddesc;
    // Lower bound value for option
    T lower;
    // Lower bound type (none (-inf), greater than..., greater or equal than...)
    lbound_t lbound;
    // Upper value for option
    T upper;
    // Upper bound type (none (+inf), less than..., less or equal than...)
    ubound_t ubound;

  public:
    OptionNumeric(string name, string desc, T lower, lbound_t lbound, T upper,
                  ubound_t ubound, T vdefault, string vddesc = "", bool be_strict = false,
                  bool destructive = true) {
        static_assert(is_same<T, da_int>::value || is_floating_point<T>::value,
                      "Constructor only valid for non boolean numeric type");
        strict = be_strict;
        OptionNumeric::destructive = destructive;
        da_status status = set_name(name);
        if (status != da_status_success)
            throw std::invalid_argument(errmsg);
        status = validate<T>(lower, lbound, upper, ubound, vdefault);
        if (status != da_status_success)
            throw std::invalid_argument(errmsg);
        OptionNumeric::desc = desc;
        OptionNumeric::vdefault = vdefault;
        OptionNumeric::value = vdefault;
        OptionNumeric::setby = setby_t::def;
        OptionNumeric::lower = lower;
        OptionNumeric::lbound = lbound;
        OptionNumeric::upper = upper;
        OptionNumeric::ubound = ubound;
        OptionNumeric::otype = get_type<T>();
        OptionNumeric::vddesc = vddesc;
    };
    OptionNumeric(string name, string desc, bool vdefault, bool be_strict = false,
                  bool destructive = true) {
        static_assert(is_same<T, bool>::value, "Constructor only valid for boolean");
        strict = be_strict;
        da_status status = set_name(name);
        if (status != da_status_success)
            throw std::invalid_argument(errmsg);
        OptionNumeric::desc = desc;
        OptionNumeric::vdefault = vdefault;
        OptionNumeric::value = vdefault;
        OptionNumeric::setby = setby_t::def;
        OptionNumeric::lower = false;
        OptionNumeric::lbound = lbound_t::greaterequal;
        OptionNumeric::upper = true;
        OptionNumeric::ubound = ubound_t::lessequal;
        OptionNumeric::otype = get_type<bool>();
        OptionNumeric::destructive = destructive;
    };

    ~OptionNumeric() override = default;

    string print_option(void) const override {
        ostringstream rec;
        rec << " " << name << " = ";
        if constexpr (std::is_same_v<T, bool>) {
            rec << std::boolalpha;
        }
        rec << value;
        if (destructive || setby != setby_t::def) {
            const size_t len = rec.str().size();
            rec << (len < prn_opt_align ? string(prn_opt_align - len, ' ') : "  ") << '(';
            rec << (destructive ? "f" : "");
            rec << setby_shrt[static_cast<int>(setby)];
            rec << ')';
        }
        rec << endl;
        return rec.str();
    }

    string print_details(bool screen = true, bool doxygen = false) const override {
        ostringstream rec;
        string tylab = option_tl[get_type<T>()];
        string t = tylab.substr(0, 1);
        if (otype == option_t::opt_bool)
            rec << std::boolalpha;
        if (!screen && doxygen) {
            if (otype == option_t::opt_bool) {
                rec << " * | **" << name << "** | " << tylab << " | \\f$ " << t
                    << " = \\f$ " << vdefault << " |" << endl;
            } else {
                if (vddesc != "") { // Pretty print vdefault value
                    rec << " * | **" << name << "** | " << tylab << " | \\f$ " << t
                        << " = " << vddesc << "\\f$ |" << endl;
                } else { // No detail, conver default value
                    rec << " * | **" << name << "** | " << tylab << " | \\f$ " << t
                        << " = " << vdefault << "\\f$ |" << endl;
                }
            }
            rec << " * | " << desc << "|||" << endl;

            if (lbound == m_inf && ubound == p_inf) {
                rec << " * | There are no constraints on \\f$" << t << "\\f$.\" |||"
                    << endl;
            } else {
                if (otype == option_t::opt_bool) {
                    rec << " * | "
                        << "Valid values: true and false.|||" << endl;
                } else {
                    rec << " * | "
                        << "Valid values: \\f$";
                    if (lbound == greaterequal) {
                        rec << lower << " \\le ";
                    } else if (lbound == greaterthan) {
                        // rec << lower << " \\lt ";
                        rec << lower << " < ";
                    }
                    rec << t;
                    if (ubound == lessequal) {
                        rec << " \\le " << upper;
                    } else if (ubound == lessthan) {
                        // rec << " \\lt " << upper;
                        rec << " < " << upper;
                    }
                    rec << "\\f$. |||" << endl;
                }
            }
            rec << " * | "
                << "Foundational: " << (destructive ? "yes" : "no") << "|||" << endl;
        } else if (!screen) { // Restructured text
            if (otype == option_t::opt_bool) {
                rec << "   \"" << name << "\", \"" << tylab << "\", \":math:`" << t
                    << "=` " << vdefault << "\", \"" << desc << "\", \"";
            } else {
                if (vddesc != "") { // Pretty print vdefault value
                    rec << "   \"" << name << "\", \"" << tylab << "\", \":math:`" << t
                        << "=" << vddesc << "`\", \"" << desc << "\", \"";
                } else { // No detail, conver default value
                    rec << "   \"" << name << "\", \"" << tylab << "\", \":math:`" << t
                        << "=" << vdefault << "`\", \"" << desc << "\", \"";
                }
            }

            if (lbound == m_inf && ubound == p_inf) {
                rec << "There are no constraints on :math:`" << t << "`.\"";
            } else {
                if (otype == option_t::opt_bool) {
                    rec << "true, or false.";
                } else {
                    rec << ":math:`";
                    if (lbound == greaterequal) {
                        rec << lower << " \\le ";
                    } else if (lbound == greaterthan) {
                        rec << lower << " < ";
                    }
                    rec << t;
                    if (ubound == lessequal) {
                        rec << " \\le " << upper;
                    } else if (ubound == lessthan) {
                        rec << " < " << upper;
                    }
                    rec << "`";
                }
                rec << "\"";
            }
            rec << ", ";
            if (destructive)
                rec << "\"Foundational.\"";
            rec << endl;
        } else { // Plain text
            rec << "Begin Option [" << tylab << "]" << endl;
            rec << "   Name: '" << name << "'" << endl;
            if (otype == option_t::opt_bool) {
                rec << "   Value: " << value << "     [default: " << vdefault << "]"
                    << endl;
                rec << "   Valid values: ";
                rec << "true and false" << endl;
            } else {
                rec << "   Value: " << value << "     [default: " << vdefault << "]"
                    << endl;
                rec << "   Range: ";
                if (lbound == m_inf && ubound == p_inf) {
                    rec << "unbounded" << endl;
                } else {
                    if (lbound == greaterequal) {
                        rec << lower << " <= ";
                    } else if (lbound == greaterthan) {
                        rec << lower << " < ";
                    }
                    rec << "value ";
                    if (ubound == lessequal) {
                        rec << " <= " << upper;
                    } else if (ubound == lessthan) {
                        rec << " < " << upper;
                    }
                    rec << endl;
                }
            }
            rec << "   Desc: " << desc << endl;
            rec << "   Set-by: " << setby_l[setby] << endl;
            rec << "   Foundational: " << (destructive ? "true" : "false") << endl;
            rec << "End Option" << endl;
        }
        return rec.str();
    }

    void get(T &value) const { value = OptionNumeric<T>::value; };
    option_variant_t get() const override { return this->value; };

    da_status set(T value, setby_t setby = setby_t::user) {
        da_status status = da_status_success;
        if (get_option_t() != da_options::option_t::opt_bool) {
            status = validate(lower, lbound, upper, ubound, value, false);
            if (status != da_status_success)
                return status; // Compose error with status+errmsg
        }
        OptionNumeric::value = value;
        OptionNumeric::setby = setby;
        return status; // Compose error with status+errmsg
    };
};

// Add OptionString class
class OptionString : public OptionBase {
    // Default label
    string vdefault;
    // Selected label
    string value;
    map<string, da_int> labels;

  public:
    OptionString(string name, string desc, map<string, da_int> labels, string vdefault,
                 bool be_strict = false, bool destructive = true) {
        strict = be_strict;
        string label, label_vdefault;
        bool defok = false;
        da_status status = set_name(name);
        if (status != da_status_success)
            throw std::invalid_argument(errmsg);
        label_vdefault = vdefault;
        if (strict) {
            OptionUtils::prep_str(
                label_vdefault); // this is expensive: done only in strict mode
            if (vdefault != label_vdefault) {
                errmsg = "Option '" + name +
                         "': Default string option changed after processing, replace '" +
                         vdefault + "' by '" + label_vdefault + "'.";
                throw std::invalid_argument(errmsg);
            }
        }

        if (labels.size() != 0 && strict) {
            // Deal with categorical data slightly differently from freeform string options

            if (label_vdefault == "") {
                errmsg = "Option '" + name +
                         "': Invalid default value (string reduced to zero-length).";
                throw std::invalid_argument(errmsg);
            }

            for (const auto &entry : labels) {
                label = entry.first;
                OptionUtils::prep_str(
                    label); // this is expensive: done only in strict mode
                if (label == "") {
                    errmsg = "Option '" + name +
                             "': Invalid option value (string reduced to zero-length).";
                    throw std::invalid_argument(errmsg);
                } else if (label != entry.first) {
                    errmsg = "Option '" + name +
                             "': Label changed after processing, replace '" +
                             entry.first + "' by '" + label + "'.";
                    throw std::invalid_argument(errmsg);
                }
                if (label == label_vdefault)
                    defok = true;
            }
            // Check that default is valid
            if (!defok) {
                errmsg = "Option '" + name + "': Default label is invalid.";
                throw std::invalid_argument(errmsg);
            }
        }

        OptionString::labels = labels;
        OptionString::vdefault = label_vdefault;
        OptionString::setby = setby_t::def;
        OptionString::value = label_vdefault;
        OptionString::desc = desc;
        OptionString::otype = get_type<string>();
        OptionString::destructive = destructive;
    };

    ~OptionString() override = default;

    string print_option(void) const override {
        ostringstream rec;
        rec << " " << name << " = ";
        rec << value;
        if (destructive || setby != setby_t::def) {
            const size_t len = rec.str().size();
            rec << (len < prn_opt_align ? string(prn_opt_align - len, ' ') : "  ") << '(';
            rec << (destructive ? "f" : "");
            rec << setby_shrt[static_cast<int>(setby)];
            rec << ')';
        }
        rec << endl;
        return rec.str();
    }
    string print_details(bool screen = true, bool doxygen = false) const override {
        ostringstream rec;
        if (!screen && doxygen) {
            rec << " * | **" << name << "** | string | \\f$ s = \\f$ `" << vdefault
                << "` |" << endl;
            rec << " * | " << desc << "|||" << endl;
            if (labels.size() > 0) {
                // Categorical options
                rec << " * | "
                    << "Valid values: \\f$s =\\f$ ";
                {
                    size_t n = labels.size();
                    for (auto const &it : labels) {
                        rec << "`" << it.first << "`";
                        switch (n) {
                        case 1:
                            rec << ".";
                            break;
                        case 2:
                            rec << ", or ";
                            break;
                        default:
                            rec << ", ";
                            break;
                        }
                        n--;
                    }
                    rec << " |||" << endl;
                }
            }
            rec << " * | "
                << "Foundational: " << (destructive ? "yes" : "no") << "|||" << endl;
        } else if (!screen) { // Restructured text
            rec << "   \"" << name << "\", \"string\", ";
            if (vdefault != "") {
                if (vdefault == "\"") {
                    rec << "\":math:`s=` `~\"`\"";
                } else if (vdefault == "~") {
                    rec << "\":math:`s=` `~~`\"";
                } else if (vdefault == "\\") {
                    rec << "\":math:`s=` `\\\\`\"";
                } else {
                    rec << "\":math:`s=` `" << vdefault << "`\"";
                }
            } else {
                rec << "\"empty\"";
            }
            rec << ", \"" << desc << "\", \"";
            if (labels.size() > 0) {
                // Categorical options
                {
                    rec << ":math:`s=` ";
                    size_t n = labels.size();
                    for (auto const &it : labels) {
                        rec << "`" << it.first << "`";
                        switch (n) {
                        case 1:
                            rec << ".";
                            break;
                        case 2:
                            rec << ", or ";
                            break;
                        default:
                            rec << ", ";
                            break;
                        }
                        n--;
                    }
                }
            }
            rec << "\", ";
            if (destructive)
                rec << "\"Foundational.\"";
            rec << endl;
        } else { // Plain text
            rec << "Begin Option [string]" << endl;
            rec << "   Name: '" << name << "'" << endl;
            rec << "   Value: '" << value << "'     [default: '" << vdefault << "']"
                << endl;
            if (labels.size() > 0) {
                // Categorical options
                rec << "   Valid values: " << endl;
                for (auto const &it : labels) {
                    rec << "      '" << it.first << "' : " << it.second << endl;
                }
            }
            rec << "   Desc: " << desc << endl;
            rec << "   Set-by: " << setby_l[setby] << endl;
            rec << "   Foundational: " << (destructive ? "true" : "false") << endl;
            rec << "End Option" << endl;
        }
        return rec.str();
    }

    void get(string &value) { value = OptionString::value; };
    void get(string &value, da_int &id) {
        value = OptionString::value;
        if (labels.size() != 0) {
            id = labels.at(OptionString::value);
        } else {
            // Internal error: free-form option does not have label id and cannot be queried with this method
            // return invalid index
            id = -1;
        }
    }
    option_variant_t get() const override { return this->value; };
    da_status query_map(string &value, da_int &id, bool find_by_value) {
        // option("option name", "value", id)
        // find_by_value true => given the value, it finds id
        //              false => given the id, finds the value
        if (find_by_value) {
            // find the id based on value
            auto pos = labels.find(value);
            if (pos != labels.end()) {
                id = pos->second;
                return da_status_success;
            }
        } else {
            if (labels.size() > 0) {
                for (auto const &it : labels) {
                    if (it.second == id) {
                        value = it.first;
                        return da_status_success;
                    }
                }
            }
        }
        return da_status_option_not_found;
    }

    da_status set(string value, setby_t setby = setby_t::user) {
        if (labels.size() != 0) {
            // Deal with categorical data slightly differently from freeform string options
            // check that value is a valid
            auto pos = labels.find(value);
            if (pos != labels.end()) {
                OptionString::value = value;
            } else {
                // not found, try again with value tidyed
                string val{value};
                OptionUtils::prep_str(val); // this is expensive: done only if find fails
                pos = labels.find(val);
                if (pos == labels.end()) {
                    errmsg = "Unrecognized value '" + val + "' for option '" +
                             OptionBase::get_name() + "'.";
                    return da_status_option_invalid_value;
                }
                OptionString::value = val;
            }
        } else {
            // free-form only process if strict mode
            if (strict) {
                string val{value};
                OptionUtils::prep_str(val); // this is expensive: done only in strict mode
                OptionString::value = val;
            } else {
                OptionString::value = value;
            }
        }

        OptionString::setby = setby;
        return da_status_success;
    };
};

// R E G I S T R Y

// Option registry
class OptionRegistry {
  private:
    bool readonly = false;

  protected:
    /* Hash table of all the options, indexed by their string name */
    std::unordered_map<string, shared_ptr<OptionBase>> registry;

  public:
    OptionRegistry() { readonly = false; };
    ~OptionRegistry() = default;
    void lock(void) { readonly = true; }
    void unlock(void) { readonly = false; }
    string errmsg = "";

    da_status register_opt(std::shared_ptr<OptionBase> o, bool overwrite = false) {

        if (readonly) {
            errmsg = "Registry is locked";
            return da_status_option_locked;
        }
        if (overwrite) {
            // Special case where we want to replace an already registered option
            auto search = registry.find(o->get_name());
            registry.erase(search);
        }
        size_t n = registry.size();
        registry.insert({o->get_name(), o});
        bool ok = (n != registry.size());
        if (!ok) {
            errmsg = "Registry could not add option. Duplicate?";
            return da_status_invalid_input;
        }
        return da_status_success;
    }

    /* Registry Setter
     * name - option name
     * value - value to set
     * setby - flag 0 (default), 1 (user), 2 (solver)
     */
    template <typename U>
    da_status set(string name, U value, setby_t setby = setby_t::user,
                  bool *destructive = nullptr) {
        if (readonly) {
            errmsg = "Registry is locked";
            return da_status_option_locked;
        }
        auto search = registry.find(name); // search assuming name is tidy
        if (search == registry.end()) {
            string oname{name};
            OptionUtils::prep_str(oname); // this is expensive: done only if search failed
            search = registry.find(oname); // search again with oname tidy
            if (search == registry.end()) {
                errmsg = "Option '" + oname + "' not found in the option registry";
                return da_status_option_not_found;
            }
        }
        option_t otype = search->second->get_option_t();
        if (otype != get_type<U>()) {
            errmsg = "Option setter for '" + name + "' of type " + option_tl[otype] +
                     ", was called with the wrong type: " + option_tl[get_type<U>()];
            return da_status_option_wrong_type;
        }

        typedef
            typename std::conditional<is_same<U, da_int>::value ||
                                          is_floating_point<U>::value ||
                                          is_same<U, bool>::value,
                                      OptionNumeric<U>, OptionString>::type OptionType;
        // Defines
        // da_int -> OptionNumeric<da_int>
        // float -> OptionNumeric<float>
        // double -> OptionNumeric<double>
        // otherwise -> OptionString
        // bool -> OptionNumeric<bool>
        da_status status =
            std::static_pointer_cast<OptionType>(search->second)->set(value, setby);
        if (status != da_status_success) {
            // Get the error message
            errmsg = std::static_pointer_cast<OptionType>(search->second)->get_errmsg();
        } else if (destructive)
            *destructive =
                std::static_pointer_cast<OptionType>(search->second)->is_destructive();
        return status;
    }

    // _Float16 overload: options are stored as float, so promote and delegate
    da_status set(string name, _Float16 value, setby_t setby = setby_t::user,
                  bool *destructive = nullptr) {
        return set(name, static_cast<float>(value), setby, destructive);
    }

    /* Registry Getter
     * name - option name
     * value - location to store option value
     */
    template <typename U> da_status get(string name, U &value) {
        auto search = registry.find(name); // search assuming name is tidy
        if (search == registry.end()) {
            string oname{name};
            OptionUtils::prep_str(oname); // this is expensive: done only if search failed
            search = registry.find(oname); // search again with oname tidy
            if (search == registry.end()) {
                errmsg = "Option '" + oname + "' not found in the option registry";
                return da_status_option_not_found;
            }
        }
        option_t otype = search->second->get_option_t();
        if (otype != get_type<U>()) {
            errmsg =
                "Option getter for'" + name + "' of type " + option_tl[otype] +
                ", was called with the wrong storage type: " + option_tl[get_type<U>()];
            return da_status_option_wrong_type;
        }

        typedef typename std::conditional<
            is_same<U, da_int>::value || is_same<U, da_int *>::value ||
                is_same<U, float>::value || is_same<U, float *>::value ||
                is_same<U, double>::value || is_same<U, double *>::value ||
                is_same<U, bool>::value || is_same<U, bool *>::value,
            OptionNumeric<U>, OptionString>::type OptionType;
        // Defines
        // da_int[*] -> OptionNumeric<da_int>
        // float[*] -> OptionNumeric<float>
        // double[*] -> OptionNumeric<double>
        // bool[*] -> OptionNumeric<bool>
        // otherwise -> OptionString
        std::static_pointer_cast<OptionType>(search->second)->get(value);
        return da_status_success;
    }

    // _Float16 overload: options are stored as float, so promote/demote
    da_status get(string name, _Float16 &value) {
        float fval;
        da_status status = get(name, fval);
        if (status == da_status_success)
            value = static_cast<_Float16>(fval);
        return status;
    }

    // Auxiliary function to get value of a string option.
    da_status get(string name, string &value) {
        auto search = registry.find(name); // search assuming name is tidy
        if (search == registry.end()) {
            string oname{name};
            OptionUtils::prep_str(oname); // this is expensive: done only if search failed
            search = registry.find(oname); // search again with oname tidy
            if (search == registry.end()) {
                errmsg = "Option '" + oname + "' not found in the option registry";
                return da_status_option_not_found;
            }
        }
        option_t otype = search->second->get_option_t();
        if (otype != option_t::opt_string) {
            errmsg = "Option getter for'" + name + "' of type " + option_tl[otype] +
                     ", was called with the wrong storage type: " +
                     option_tl[option_t::opt_string];
            return da_status_option_wrong_type;
        }
        std::static_pointer_cast<OptionString>(search->second)->get(value);
        return da_status_success;
    }

    // Auxiliary function to get value and id from a categorical/string option
    da_status get(string name, string &value, da_int &id) {
        auto search = registry.find(name); // search assuming name is tidy
        if (search == registry.end()) {
            string oname{name};
            OptionUtils::prep_str(oname); // this is expensive: done only if search failed
            search = registry.find(oname); // search again using tidy oname
            if (search == registry.end()) {
                errmsg = "Option '" + oname + "' not found in the option registry";
                return da_status_option_not_found;
            }
        }
        option_t otype = search->second->get_option_t();
        if (otype != option_t::opt_string) {
            errmsg = "Option getter for'" + name + "' of type " + option_tl[otype] +
                     ", was called with the wrong storage type: " +
                     option_tl[option_t::opt_string];
            return da_status_option_wrong_type;
        }
        std::static_pointer_cast<OptionString>(search->second)->get(value, id);
        return da_status_success;
    }

    // Auxiliary function to query the id and categorical values of a string option
    da_status query_map(string name, string &value, da_int &id,
                        bool find_by_value = false) {
        auto search = registry.find(name); // search assuming name is tidy
        if (search == registry.end()) {
            string oname{name};
            OptionUtils::prep_str(oname); // this is expensive: done only if search failed
            search = registry.find(oname); // search again using tidy oname
            if (search == registry.end()) {
                errmsg = "Option '" + oname + "' not found in the option registry";
                return da_status_option_not_found;
            }
        }
        option_t otype = search->second->get_option_t();
        if (otype != option_t::opt_string) {
            errmsg = "Option query_map for'" + name + "' of type " + option_tl[otype] +
                     ", was called with the wrong storage type: " +
                     option_tl[option_t::opt_string];
            return da_status_option_wrong_type;
        }
        return std::static_pointer_cast<OptionString>(search->second)
            ->query_map(value, id, find_by_value);
    }

    da_status set_destructive(string name, bool destructive) {
        auto search = registry.find(name); // search assuming name is tidy
        if (search == registry.end()) {
            string oname{name};
            OptionUtils::prep_str(oname); // this is expensive: done only if search failed
            search = registry.find(oname); // search again with oname tidy
            if (search == registry.end()) {
                errmsg = "Option '" + oname + "' not found in the option registry";
                return da_status_option_not_found;
            }
        }
        search->second->set_destructive(destructive);
        return da_status_success;
    }

    // Auxiliary
    void print_options(void) const {
        da_logger::logger out;
        out << "Begin Options" << std::endl;
        for (auto const &o : registry) {
            // LCOV_EXCL_START
            if (o.second->get_option_t() == option_t::opt_undefined) {
                out << "Internal ERROR: unexpected option " << o.first
                    << " with option_t::opt_undefined!" << std::endl;
                continue;
            }
            // LCOV_EXCL_STOP
            out << o.second->print_option();
        }
        out << "End Options" << std::endl;
    }

    void print_details(bool screen = true, bool doxygen = false,
                       const std::string caption = "Options table") const {
        da_logger::logger out;
        bool sep = false;
        if (!screen && doxygen) {
            out << " *" << std::endl;
            out << " * The following options are supported." << std::endl;
            out << " *" << std::endl;
            out << " * | **Option name** |  Type  | Default value|" << std::endl;
            out << " * |:----------------|:------:|-------------:|" << std::endl;
        } else if (!screen) { // Restructured text
            out << "The following options are supported." << std::endl;
            out << "\n.. csv-table:: " << caption << "\n   :escape: ~\n";
            out << "   :header: \"Option name\", \"Type\", \"Default\", "
                   "\"Description\", \"Constraints\", \"Notes\""
                << std::endl;
            out << "   " << std::endl;
        } else { // Plain text
            out << "Begin (detailed print of options)" << std::endl;
        }
        for (auto const &o : registry) {
            // LCOV_EXCL_START
            if (o.second->get_option_t() == option_t::opt_undefined) {
                out << "Internal ERROR: unexpected option " << o.first
                    << " with option_t::opt_undefined!" << std::endl;
                continue;
            }
            // LCOV_EXCL_STOP
            if (sep && doxygen)
                out << " * | |||" << std::endl;

            out << o.second->print_details(screen, doxygen);
            sep = true;
        }
        // Restructured text does not require any termination
        if (!screen && doxygen) {
            out << " *" << std::endl;
        } else if (screen) {
            out << "End" << std::endl;
        }
    };

    void export_registry(
        std::function<void(const std::string &, option_variant_t)> exporter) const {
        for (auto const &o : registry) {
            exporter(o.first, o.second->get());
        }
    };
};
}; // namespace da_options
#endif
