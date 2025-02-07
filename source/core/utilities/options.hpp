/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef AOCLDA_OPTIONS_HPP_
#define AOCLDA_OPTIONS_HPP_

#include "aoclda.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

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

struct OptionUtils {
    static void prep_str(string &str) {
        const std::regex ltrim("^[[:space:]]+");
        const std::regex rtrim("[[:space:]]+$");
        const std::regex squeeze("[[:space:]]+");
        str = std::regex_replace(str, ltrim, std::string(""));
        str = std::regex_replace(str, rtrim, std::string(""));
        str = std::regex_replace(str, squeeze, std::string(" "));
        transform(str.begin(), str.end(), str.begin(), ::tolower);
    };
};

class OptionBase {
  public:
    const string setby_l[3] = {"(default)", "(user)", "(solver)"};

    da_status set_name(string &str) {
        name = str;
        OptionUtils::prep_str(name);
        if (name == "") {
            errmsg = "Invalid name (string reduced to zero-length).";
            return da_status_option_invalid_value;
        }
        return da_status_success;
    }
    string get_name(void) const { return name; }
    option_t get_option_t(void) const { return otype; }
    string get_errmsg(void) const { return errmsg; }
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

  protected:
    // Name i.e. "Iteration Limit"
    string name;
    // Type of the option (int, real (float/double), string or bool)
    option_t otype = opt_undefined;
    string desc; // Brief description (free text)
    setby_t setby;
    string errmsg = ""; // internal error buffer
    // Prepare the option key/value pair to be printed on screen.
    // Called with option "print options = yes"
    virtual string print_option(void) = 0;
    // Compose the option details (used for documentation)
    // Screen = true => print it in plain text pretty print
    // Screen = false => used to indicate a file format is requested
    // => doxygen = true => format is set to Doxygen
    // => doxygen = false => format is set to ReStructuredText
    virtual string print_details(bool screen = true, bool doxygen = false) = 0;
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
                  ubound_t ubound, T vdefault, string vddesc = "") {
        static_assert(is_same<T, da_int>::value || is_floating_point<T>::value,
                      "Constructor only valid for non boolean numeric type");
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
    OptionNumeric(string name, string desc, bool vdefault) {
        static_assert(is_same<T, bool>::value, "Constructor only valid for boolean");
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
    };

    virtual ~OptionNumeric(){};

    string print_option(void) {
        ostringstream rec;
        rec << " " << name << " = ";
        if constexpr (std::is_same_v<T, bool>) {
            rec << std::boolalpha;
        }
        rec << value << endl;
        return rec.str();
    }

    string print_details(bool screen = true, bool doxygen = false) {
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
                rec << "There are no constraints on :math:`" << t << "`.\"" << endl;
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
                rec << "\"" << endl;
            }
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
            rec << "End Option" << endl;
        }
        return rec.str();
    }

    void get(T &value) const { value = OptionNumeric<T>::value; };

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
    OptionString(string name, string desc, map<string, da_int> labels, string vdefault) {
        string label, label_vdefault;
        bool defok = false;
        da_status status = set_name(name);
        if (status != da_status_success)
            throw std::invalid_argument(errmsg);

        label_vdefault = vdefault;
        OptionUtils::prep_str(label_vdefault);
        if (vdefault != label_vdefault) {
            errmsg = "Option '" + name +
                     "': Default string option changed after processing, replace '" +
                     vdefault + "' by '" + label_vdefault + "'.";
            throw std::invalid_argument(errmsg);
        }

        if (labels.size() != 0) {
            // Deal with categorical data slightly differently from freeform string options

            if (label_vdefault == "") {
                errmsg = "Option '" + name +
                         "': Invalid default value (string reduced to zero-length).";
                throw std::invalid_argument(errmsg);
            }

            for (const auto &entry : labels) {
                label = entry.first;
                OptionUtils::prep_str(label);
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
    };

    virtual ~OptionString(){};

    string print_option(void) {
        ostringstream rec;
        rec << " " << name << " = " << value << endl;
        return rec.str();
    }
    string print_details(bool screen = true, bool doxygen = false) {
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
            rec << "\"" << endl;
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
            throw std::runtime_error("free-form option does not have label id and cannot "
                                     "be queried with this method");
        }
    }
    da_status set(string value, setby_t setby = setby_t::user) {
        string val(value);
        OptionUtils::prep_str(val);

        if (labels.size() != 0) {
            // Deal with categorical data slightly differently from freeform string options
            // check that value is a valid
            auto pos = labels.find(val);
            if (pos == labels.end()) {
                errmsg = "Unrecognized value '" + val + "' for option '" +
                         OptionBase::get_name() + "'.";
                return da_status_option_invalid_value;
            }
        }

        OptionString::value = val;
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
    ~OptionRegistry(){};
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
    da_status set(string name, U value, setby_t setby = setby_t::user) {
        if (readonly) {
            errmsg = "Registry is locked";
            return da_status_option_locked;
        }
        string oname = name;
        OptionUtils::prep_str(oname);
        auto search = registry.find(oname);
        if (search == registry.end()) {
            errmsg = "Option '" + oname + "' not found in the option registry";
            return da_status_option_not_found;
        }
        option_t otype = search->second->get_option_t();
        if (otype != get_type<U>()) {
            errmsg = "Option setter for '" + oname + "' of type " + option_tl[otype] +
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
        }
        return status;
    }

    /* Registry Getter
     * name - option name
     * value - location to store option value
     */
    template <typename U> da_status get(string name, U &value) {
        string oname = name;
        OptionUtils::prep_str(oname);
        auto search = registry.find(oname);
        if (search == registry.end()) {
            errmsg = "Option '" + oname + "' not found in the option registry";
            return da_status_option_not_found;
        }
        option_t otype = search->second->get_option_t();
        if (otype != get_type<U>()) {
            errmsg =
                "Option getter for'" + oname + "' of type " + option_tl[otype] +
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

    // Auxiliary function to get value of a string option.
    da_status get(string name, string &value) {
        string oname = name;
        OptionUtils::prep_str(oname);
        auto search = registry.find(oname);
        if (search == registry.end()) {
            errmsg = "Option '" + oname + "' not found in the option registry";
            return da_status_option_not_found;
        }
        option_t otype = search->second->get_option_t();
        if (otype != option_t::opt_string) {
            errmsg = "Option getter for'" + oname + "' of type " + option_tl[otype] +
                     ", was called with the wrong storage type: " +
                     option_tl[option_t::opt_string];
            return da_status_option_wrong_type;
        }
        std::static_pointer_cast<OptionString>(search->second)->get(value);
        return da_status_success;
    }

    // Auxiliary function to get value and id from a categorical/string option
    da_status get(string name, string &value, da_int &id) {
        string oname = name;
        OptionUtils::prep_str(oname);
        auto search = registry.find(oname);
        if (search == registry.end()) {
            errmsg = "Option '" + oname + "' not found in the option registry";
            return da_status_option_not_found;
        }
        option_t otype = search->second->get_option_t();
        if (otype != option_t::opt_string) {
            errmsg = "Option getter for'" + oname + "' of type " + option_tl[otype] +
                     ", was called with the wrong storage type: " +
                     option_tl[option_t::opt_string];
            return da_status_option_wrong_type;
        }
        std::static_pointer_cast<OptionString>(search->second)->get(value, id);
        return da_status_success;
    }

    // Auxiliary
    void print_options(void) {
        std::cout << "Begin Options" << std::endl;
        for (auto const &o : registry) {
            option_t otype = (o.second)->get_option_t();
            switch (otype) {
            case option_t::opt_int:
                std::cout << std::static_pointer_cast<OptionNumeric<da_int>>(o.second)
                                 ->print_option();
                break;
            case option_t::opt_float:
                std::cout << std::static_pointer_cast<OptionNumeric<float>>(o.second)
                                 ->print_option();
                break;
            case option_t::opt_double:
                std::cout << std::static_pointer_cast<OptionNumeric<double>>(o.second)
                                 ->print_option();
                break;
            case option_t::opt_string:
                std::cout
                    << std::static_pointer_cast<OptionString>(o.second)->print_option();
                break;
            case option_t::opt_bool:
                std::cout << std::static_pointer_cast<OptionNumeric<bool>>(o.second)
                                 ->print_option();
                break;
            default: // LCOV_EXCL_LINE
                // LCOV_EXCL_START
                std::cout << "Internal ERROR: unexpected option with option_t::undefined"
                          << std::endl;
                // LCOV_EXCL_STOP
            }
        }
        std::cout << "End Options" << std::endl;
    }

    void print_details(bool screen = true, bool doxygen = false,
                       const std::string caption = "Options table") {
        bool sep = false;
        if (!screen && doxygen) {
            std::cout << " *" << std::endl;
            std::cout << " * The following options are supported." << std::endl;
            std::cout << " *" << std::endl;
            std::cout << " * | **Option name** |  Type  | Default value|" << std::endl;
            std::cout << " * |:----------------|:------:|-------------:|" << std::endl;
        } else if (!screen) { // Restructured text
            std::cout << "The following options are supported." << std::endl;
            std::cout << "\n.. csv-table:: " << caption << "\n   :escape: ~\n";
            std::cout << "   :header: \"Option name\", \"Type\", \"Default\", "
                         "\"Description\", \"Constraints\""
                      << std::endl;
            std::cout << "   " << std::endl;
        } else { // Plain text
            std::cout << "Begin (detailed print of options)" << std::endl;
        }
        for (auto const &o : registry) {
            if (sep && doxygen)
                std::cout << " * | |||" << std::endl;

            option_t otype = (o.second)->get_option_t();
            switch (otype) {
            case option_t::opt_int:
                std::cout << std::static_pointer_cast<OptionNumeric<da_int>>(o.second)
                                 ->print_details(screen, doxygen);
                break;
            case option_t::opt_float:
                std::cout << std::static_pointer_cast<OptionNumeric<float>>(o.second)
                                 ->print_details(screen, doxygen);
                break;
            case option_t::opt_double:
                std::cout << std::static_pointer_cast<OptionNumeric<double>>(o.second)
                                 ->print_details(screen, doxygen);
                break;
            case option_t::opt_string:
                std::cout << std::static_pointer_cast<OptionString>(o.second)
                                 ->print_details(screen, doxygen);
                break;
            case option_t::opt_bool:
                std::cout << std::static_pointer_cast<OptionNumeric<bool>>(o.second)
                                 ->print_details(screen, doxygen);
                break;
            default: // LCOV_EXCL_LINE
                // LCOV_EXCL_START
                std::cout << "Internal ERROR: unexpected option with option_t::undefined"
                          << std::endl;
                // LCOV_EXCL_STOP
            }
            sep = true;
        }
        // Restructured text does not require any termination
        if (!screen && doxygen) {
            std::cout << " *" << std::endl;
        } else if (screen) {
            std::cout << "End" << std::endl;
        }
    };
};
}; // namespace da_options
#endif
