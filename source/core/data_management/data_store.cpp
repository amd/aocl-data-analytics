#include "data_store.hpp"

using namespace da_data;

bool da_data::validate_interval(interval p, da_int max_val) {
    if (p.first > p.second)
        return false;
    if (p.first < 0 || p.second >= max_val)
        return false;
    return true;
}