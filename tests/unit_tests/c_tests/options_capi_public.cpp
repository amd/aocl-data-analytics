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

#include "aoclda.h"
#include "gtest/gtest.h"

/*
 * Tests for options APIs (aoclda_options.h) - handle variants
 */
TEST(OptionsCAPI, SetGetInt) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);

    EXPECT_EQ(da_options_set_int(handle, "n_clusters", 5), da_status_success);

    da_int value = 0;
    EXPECT_EQ(da_options_get_int(handle, "n_clusters", &value), da_status_success);
    EXPECT_EQ(value, 5);

    da_handle_destroy(&handle);
}

TEST(OptionsCAPI, SetGetRealDouble) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_dbscan), da_status_success);

    EXPECT_EQ(da_options_set_real_d(handle, "eps", 2.5), da_status_success);

    double value = 0.0;
    EXPECT_EQ(da_options_get_real_d(handle, "eps", &value), da_status_success);
    EXPECT_EQ(value, 2.5);

    da_handle_destroy(&handle);
}

TEST(OptionsCAPI, SetGetRealFloat) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_s(&handle, da_handle_dbscan), da_status_success);

    EXPECT_EQ(da_options_set_real_s(handle, "eps", 2.5f), da_status_success);

    float value = 0.0f;
    EXPECT_EQ(da_options_get_real_s(handle, "eps", &value), da_status_success);
    EXPECT_EQ(value, 2.5f);

    da_handle_destroy(&handle);
}

TEST(OptionsCAPI, SetGetString) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);

    EXPECT_EQ(da_options_set_string(handle, "algorithm", "lloyd"), da_status_success);

    char value[100];
    da_int lvalue = 100;
    EXPECT_EQ(da_options_get_string(handle, "algorithm", value, &lvalue),
              da_status_success);

    da_handle_destroy(&handle);
}

TEST(OptionsCAPI, GetStringKey) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_options_set_string(handle, "algorithm", "lloyd"), da_status_success);

    char value[100];
    da_int lvalue = 100;
    da_int key = -1;
    EXPECT_EQ(da_options_get_string_key(handle, "algorithm", value, &lvalue, &key),
              da_status_success);

    da_handle_destroy(&handle);
}

TEST(OptionsCAPI, Print) {
    da_handle handle = nullptr;
    EXPECT_EQ(da_handle_init_d(&handle, da_handle_kmeans), da_status_success);
    EXPECT_EQ(da_options_set_int(handle, "n_clusters", 3), da_status_success);

    EXPECT_EQ(da_options_print(handle), da_status_success);

    da_handle_destroy(&handle);
}

/*
 * Tests for datastore options (aoclda_options.h) - datastore variants
 */
TEST(OptionsCAPI, DatastoreSetGetInt) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    EXPECT_EQ(da_datastore_options_set_int(store, "use header row", 1),
              da_status_success);

    da_int val = 0;
    EXPECT_EQ(da_datastore_options_get_int(store, "use header row", &val),
              da_status_success);
    EXPECT_EQ(val, 1);

    da_datastore_destroy(&store);
}

TEST(OptionsCAPI, DatastoreSetGetString) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    EXPECT_EQ(da_datastore_options_set_string(store, "datastore precision", "double"),
              da_status_success);

    char value[100];
    EXPECT_EQ(da_datastore_options_get_string(store, "datastore precision", value, 100),
              da_status_success);

    da_datastore_destroy(&store);
}

TEST(OptionsCAPI, DatastoreSetGetRealDouble) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    // Datastore has no real-valued options; verify the API returns an error
    da_status status = da_datastore_options_set_real_d(store, "nonexistent_option", 1.5);
    EXPECT_NE(status, da_status_success);

    double val = 0.0;
    status = da_datastore_options_get_real_d(store, "nonexistent_option", &val);
    EXPECT_NE(status, da_status_success);

    da_datastore_destroy(&store);
}

TEST(OptionsCAPI, DatastoreSetGetRealFloat) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    da_status status = da_datastore_options_set_real_s(store, "nonexistent_option", 1.5f);
    EXPECT_NE(status, da_status_success);

    float val = 0.0f;
    status = da_datastore_options_get_real_s(store, "nonexistent_option", &val);
    EXPECT_NE(status, da_status_success);

    da_datastore_destroy(&store);
}

TEST(OptionsCAPI, DatastorePrint) {
    da_datastore store = nullptr;
    EXPECT_EQ(da_datastore_init(&store), da_status_success);

    EXPECT_EQ(da_datastore_options_print(store), da_status_success);

    da_datastore_destroy(&store);
}
