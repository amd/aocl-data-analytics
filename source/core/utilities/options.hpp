/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

/*
 * Current status TODO
 * ===================
 * [ ] Rename CamelCase to snake_case
 * [ ] Update of the comments bellow...
 *
 * Options Registry TODO REWRITE THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 * ========================================================================
 *
 * da_options namespace provides a Registry class that "registers"
 * options and an Option class that defines an "option" element.
 * Options can be of any four classes: Integer, Real (float or double,
 * templated), Boolean or String. The registry class
 * defines a method Called "Register" to add a new option.
 * Registered options can be set using Registry.SetOption,
 * queried using Registry.GetOption and for strings Registry.GetKey, and
 * prety-printed using either Registry.PrintOptions or Registry.print_details.
 *
 * Options have a "name" that is used to distiguish them, so this should be
 * unique among the ALL the registered options (of any Option class).
 * Furthermore, the "name" string is sanitized before using, that is, it is
 * trimmed and blanks squeezed. Do not register the same option twice.
 * No checks on option registry are done.
 *
 * Quering options have a certain cost and should be done only once at the
 * beggining of the solver or when initializing the internal data.
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

enum lbound_t { m_inf = 0, greaterthan, greaterequal };
enum ubound_t { p_inf = 0, lessthan, lessequal };
enum setby_t { def = 0, user = 1, solver = 2 };
const string option_tl[6] = {"?", "integer", "real", "real", "string", "boolean"};
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
    void prep_str(string &str) {
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
        OptionUtils util;
        util.prep_str(name);
        if (name == "") {
            errmsg = "Invalid name (string reduced to zero-length).";
            return da_status_option_invalid_value;
        }
        return da_status_success;
    }
    string get_name(void) const { return name; }
    option_t get_option_t(void) const { return otype; }
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
        // check that it is within range (lower bound)
        if ((lbound == greaterthan) && (value <= lower))
            iflag += 1;
        else if ((lbound == greaterequal) && (value < lower))
            iflag += 2;
        // check that it is within range (upper bound)
        if ((ubound == lessthan) && (value >= upper))
            iflag += 10;
        else if ((ubound == lessequal) && (value > upper))
            iflag += 20;
        if (iflag) {
            // FIXME: use iflag for pretty printing error
            errmsg = "Option '" + name + "': value out-of-bounds";
            return da_status_option_invalid_value;
        }
        return da_status_success;
    }

  protected:
    // name i.e. "Iteration Limit"
    string name;
    // type of the option (int, real (float/double), string or bool)
    option_t otype = opt_undefined;
    string desc; // brief description (free text)
    setby_t setby;
    string errmsg = ""; // internal error buffer
    virtual string print_option(void) = 0;
    virtual string print_details(bool doxygen) = 0;
};

template <typename T> class OptionNumeric : public OptionBase {
    // actual value of the option
    T value;
    // default value for option
    T vdefault;
    // lower bound value for option
    T lower;
    // lower bound type (none (-inf), greater than..., greater or equal than...)
    lbound_t lbound;
    // upper value for option
    T upper;
    // upper bound type (none (+inf), less than..., less or equal than...)
    ubound_t ubound;

  public:
    OptionNumeric(string name, string desc, T lower, lbound_t lbound, T upper,
                  ubound_t ubound, T vdefault) {
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
        rec << " " << name << " = " << value << endl;
        return rec.str();
    }

    string print_details(bool doxygen) {
        ostringstream rec;
        string tylab = option_tl[get_type<T>()];
        string t = tylab.substr(0, 1);
        if (doxygen) {
            if (otype == option_t::opt_bool) {
                rec << " * | **" << name << "** | " << tylab << " | \\f$ " << t
                    << " = \\f$ " << (vdefault ? "True"s : "False"s) << " |" << endl;
            } else {
                rec << " * | **" << name << "** | " << tylab << " | \\f$ " << t << " = "
                    << vdefault << "\\f$ |" << endl;
            }
            rec << " * | " << desc << "|||" << endl;

            if (lbound == m_inf && ubound == p_inf) {
                rec << " * | There are no constraints on \\f$i\\f$. |||" << endl;
            } else {
                if (otype == option_t::opt_bool) {
                    rec << " * | "
                        << "Valid values: 1 (True) and 0 (False).|||" << endl;
                } else {
                    rec << " * | "
                        << "Valid values: \\f$";
                    if (lbound == greaterequal) {
                        rec << lower << " \\le ";
                    } else if (lbound == greaterthan) {
                        rec << lower << " \\lt ";
                    }
                    rec << t;
                    if (ubound == lessequal) {
                        rec << " \\le " << upper;
                    } else if (ubound == lessthan) {
                        rec << " \\lt " << upper;
                    }
                    rec << "\\f$. |||" << endl;
                }
            }
        } else {
            rec << "Begin Option [" << tylab << "]" << endl;
            rec << "   Name: '" << name << "'" << endl;
            if (otype == option_t::opt_bool) {
                rec << "   Value: " << (value ? "True"s : "False"s)
                    << "     [default: " << (vdefault ? "True"s : "False"s) << "]"
                    << endl;
                rec << "   Valid values: ";
                rec << "1 (True) and 0 (False)" << endl;
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
                return status;
        }
        OptionNumeric::value = value;
        OptionNumeric::setby = setby;
        return status;
    };
};

// add OptionString class
class OptionString : public OptionBase {
    // default label
    string vdefault;
    // selected label
    string value;
    map<string, da_int> labels;

  public:
    OptionString(string name, string desc, map<string, da_int> labels, string vdefault) {
        string label, label_vdefault;
        bool defok = false;
        da_status status = set_name(name);
        OptionUtils util;
        if (status != da_status_success)
            throw std::invalid_argument(errmsg);

        label_vdefault = vdefault;
        util.prep_str(label_vdefault);
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
                util.prep_str(label);
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
            // check that default is valid
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
    string print_details(bool doxygen) {
        ostringstream rec;
        if (doxygen) {
            rec << " * | **" << name << "** | string | \\f$ s = \\f$ `" << vdefault
                << "` |" << endl;
            rec << " * | " << desc << "|||" << endl;
            if (labels.size() > 0) {
                // categorical options
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
        } else {
            rec << "Begin Option [String]" << endl;
            rec << "   Name: '" << name << "'" << endl;
            rec << "   Value: '" << value << "'     [default: '" << vdefault << "']"
                << endl;
            if (labels.size() > 0) {
                //categorical options
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
        OptionUtils util;
        util.prep_str(val);

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
        //TODO add unit tests for overwrite (ANDREW!!!!!)
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
        OptionUtils util;
        util.prep_str(oname);
        auto search = registry.find(oname);
        if (search == registry.end()) {
            errmsg = "Option '" + oname + "' not found in the option registry";
            return da_status_option_not_found;
        }
        option_t otype = search->second->get_option_t();
        if (otype != get_type<U>()) {
            errmsg = "Option setter for'" + oname + "' of type " + option_tl[otype] +
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
        return std::static_pointer_cast<OptionType>(search->second)->set(value, setby);
    }

    /* Registry Getter
     * name - option name
     * value - location to store option value
     */
    template <typename U> da_status get(string name, U &value) {
        string oname = name;
        OptionUtils util;
        util.prep_str(oname);
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
        OptionUtils util;
        util.prep_str(oname);
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
        OptionUtils util;
        util.prep_str(oname);
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

    void print_details(bool doxygen) {
        bool sep = false;
        if (doxygen) {
            std::cout << " *" << std::endl;
            std::cout << " * \\section anchor_itsol_options Options" << std::endl;
            std::cout << " * The iterative solver framework has the following options."
                      << std::endl;
            std::cout << " *" << std::endl;
            std::cout << " * | **Option name** |  Type  | Default value|" << std::endl;
            std::cout << " * |:----------------|:------:|-------------:|" << std::endl;
        } else {
            std::cout << "Begin (detailed print of registered options)" << std::endl;
        }
        for (auto const &o : registry) {
            if (sep && doxygen)
                std::cout << " * | |||" << std::endl;

            option_t otype = (o.second)->get_option_t();
            switch (otype) {
            case option_t::opt_int:
                std::cout << std::static_pointer_cast<OptionNumeric<da_int>>(o.second)
                                 ->print_details(doxygen);
                break;
            case option_t::opt_float:
                std::cout << std::static_pointer_cast<OptionNumeric<float>>(o.second)
                                 ->print_details(doxygen);
                break;
            case option_t::opt_double:
                std::cout << std::static_pointer_cast<OptionNumeric<double>>(o.second)
                                 ->print_details(doxygen);
                break;
            case option_t::opt_string:
                std::cout << std::static_pointer_cast<OptionString>(o.second)
                                 ->print_details(doxygen);
                break;
            case option_t::opt_bool:
                std::cout << std::static_pointer_cast<OptionNumeric<bool>>(o.second)
                                 ->print_details(doxygen);
                break;
            default: // LCOV_EXCL_LINE
                // LCOV_EXCL_START
                std::cout << "Internal ERROR: unexpected option with option_t::undefined"
                          << std::endl;
                // LCOV_EXCL_STOP
            }
            sep = true;
        }
        if (doxygen) {
            std::cout << " *" << std::endl;
        } else {
            std::cout << "End" << std::endl;
        }
    };
};
}; // namespace da_options
#endif
