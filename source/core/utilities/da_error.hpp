#ifndef DA_ERROR_HPP
#define DA_ERROR_HPP

#include "aoclda_error.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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
 * and warning except when composing the message.
 * 
 * Methods for printing the error message
 * --------------------------------------
 * 
 * 1.- print() composes the message and prints to std::cout
 * 2.- print(str) composes the message and returns it in std::string str
 * 3.- print(ss) composes the message and pipes it to std::stringstream ss
 *  
 * If not error or warning was registered then a friendly banner is produced:
 * "Last operation was successful."
 * 
 * More details
 * ------------
 * 
 * For debug builds where NDEBUG and VERBOSE are set, the error is
 * printed immediatly. This as many benefits during development of 
 * new functionalities where not all the calling stack as access to 
 * the error trace.
 * 
 * It is possible to "record" the error/warning directly, without using the
 * MACRO functions da_error() or da_warn(). By the "rec()" method.
 * This method provides full control on the telemetry and texts used to compose 
 * the banner.
 * In stead of using da_error(e, status, msg), it is possible to call
 * return e.rec(da_status status, string msg, string det = "",
 *              string tel = "<no telemetry provided>", size_t ln = 0,
 *              severity_type sev = DA_ERROR);
 * Here msg is the same as for MACROS da_error() and da_warn(), while det is a string
 * that contains further details about the error and possible fixes, it is a free
 * form string that if not empty is printed under a "details:" section on its own.
 * tel is a string to contain the telemetry data, this generally is the __FILE__,
 * but need not be. ln contains the line number and is generally __LINE__.
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

/** Labels to use when printing the error/warning messages */
string const sev_labels[3] = {"???", "WARNING", "ERROR"};
enum severity_type {
    DA_NOTSET = 0,  // Initial state
    DA_WARNING = 1, // something happened. Returned data is potentially safe to use
    DA_ERROR = 2,   // error occurred. Returned data is unsafe
};

enum action_t {
    DA_RECORD = 0, // compose/print the error/warning
    DA_ABORT,      // compose/print and call std::abort()
    DA_THROW,      // compose/print and throw std::run_time_error(msg) exception
};

class da_error_t {

    // Main error message
    string mesg{""};
    // Some friendy message and possible fix suggestions
    string details{""};
    // Telemetry
    string telem{""};
    // This should indicate if the output of the functionality is
    // usable or not. DA_WARNING -> useable, and DA_ERROR -> not usable.
    severity_type severity{severity_type::DA_NOTSET};
    // Registered status, by default no error
    da_status status{da_status_success};
    // Action to take when rec() method is called
    action_t action{action_t::DA_RECORD};

  public:
    severity_type get_severity(void) { return severity; };
    da_status get_status(void) { return status; };
    string get_mesg(void) { return mesg; };
    string get_details(void) { return details; };
    string get_telem(void) { return telem; };

    da_status clear(void) {
        mesg = "";
        details = "";
        telem = "";
        status = da_status_success;
        severity = severity_type::DA_NOTSET;
        return da_status_success;
    }

    da_error_t(enum action_t action) : action(action){};
    // da_error_t(enum action_t action) = action_t::DA_RECORD) : action(action){};
    ~da_error_t(){};
    // Build banner
    void print(stringstream &ss) {
        if (status != da_status_success) {
            ss << sev_labels[severity] << ":" << telem << ": " << mesg << std::endl;
            ss << "status: " << status << std::endl;
            if (details != "") {
                ss << "details:" << std::endl;
                ss << details << std::endl;
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
                  severity_type sev = DA_ERROR) {
        this->status = status;
        this->mesg = msg;
        this->details = det;
        this->telem = tel + to_string(ln);
        this->severity = sev;
#ifndef NDEBUG
        this->print();
#endif
        if (this->action == action_t::DA_ABORT)
            std::abort();
        else if (this->action == action_t::DA_THROW)
            std::runtime_error(this->mesg);
        return status;
    };
};

#define da_error(e, status, msg)                                                         \
    (e)->rec(status, (msg), "", __FILE__ ":", __LINE__,                                  \
             da_errors::severity_type::DA_ERROR)
#define da_warn(e, status, msg)                                                          \
    (e)->rec(status, (msg), "", __FILE__ ":", __LINE__,                                  \
             da_errors::severity_type::DA_WARNING)

} // namespace da_errors
#endif