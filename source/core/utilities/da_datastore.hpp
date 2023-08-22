#ifndef DA_DATASTORE_HPP
#define DA_DATASTORE_HPP

#include "csv_reader.hpp"
#include "da_error.hpp"
#include "data_store.hpp"
#include "options.hpp"

/**
 * @brief Datastore structure used to store and manipulate data
 *
 */
struct _da_datastore {
  public:
    da_data::data_store *store = nullptr;
    da_csv::csv_reader *csv_parser = nullptr;
    da_errors::da_error_t *err = nullptr;
    da_options::OptionRegistry *opts = nullptr;
};

#endif