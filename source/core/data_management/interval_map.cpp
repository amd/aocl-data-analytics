#include "interval_map.hpp"
#include "aoclda.h"

using namespace da_interval_map;

namespace da_interval_map {
interval intersection(interval i1, interval i2) {
    da_int lb = std::max(i1.first, i2.first);
    da_int ub = std::min(i1.second, i2.second);
    return {lb, ub};
}
}; // namespace da_interval_map
