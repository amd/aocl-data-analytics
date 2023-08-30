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

#ifndef PCA_HPP
#define PCA_HPP

#include "../basic_statistics/moment_statistics.hpp"
#include "../basic_statistics/statistical_utilities.hpp"
#include "aoclda.h"
#include "aoclda_pca.h"
#include "basic_handle.hpp"
#include "callbacks.hpp"
#include "da_error.hpp"
#include "lapack_templates.hpp"
#include "options.hpp"
#include "pca_options.hpp"
#include <iostream>
#include <string.h>

/*
    data struct required to compute pca using 
    svd method 
*/
template <typename T> struct pca_usr_data_svd {
    T *colmean = nullptr; //Store Column Mean of X
    T *u = nullptr;
    T *sigma = nullptr;
    T *vt = nullptr;
    T *work = nullptr;
    // A needs to be copied as lapack's dgesvd modifies the matrix
    T *A = nullptr;
    da_int *iwork = nullptr;
};

/* 
    TODO: Yet to define internal data struct required to
    compute pca using correlation method
*/
template <typename T> struct pca_usr_data_corr {};

/* PCA Results data struct  */
template <typename T> struct pca_results {
    T *scores = nullptr;     /*U*Sigma*/
    T *components = nullptr; /*Vt*/
    T *variance = nullptr;   /*S**2*/
    T total_variance = 0;    /*Sum((MeanCentered X [][])**2)*/
};

/* PCA Internal Class */
template <typename T> class da_pca : public basic_handle<T> {
  private:
    /*n_samples x n_features = (nxp) */
    da_int n = 0;
    da_int p = 0;
    da_int lda = 0;

    /*Set true when init done*/
    bool initdone = false;

    /*Set true when compute done successfully*/
    bool iscomputed = false;

    /*Define default pca compute method to svd*/
    pca_comp_method method = pca_method_svd;

    /*Define default pca compute method to svd*/
    bool svd_flip_u_based = true;

    /*Defult number of output Components */
    da_int npc = 5;

    /*Data required to compute pca using svd method*/
    pca_usr_data_svd<T> *svd_data = nullptr;

    /*Data required to compute pca using corr method*/
    pca_usr_data_corr<T> *corr_data = nullptr;

    /*pca results*/
    pca_results<T> *results = nullptr;

    /* pointer to error trace */
    da_errors::da_error_t *err = nullptr;

  public:
    da_options::OptionRegistry opts;
    da_pca(da_errors::da_error_t &err) {
        this->err = &err;
        svd_data = new pca_usr_data_svd<T>;
        corr_data = new pca_usr_data_corr<T>;
        results = new pca_results<T>;
        register_pca_options<T>(opts);
    };
    ~da_pca();

    da_status init(da_int n, da_int p, T *dataX, da_int lda);

    void set_pca_compute_method(pca_comp_method method) {
        if (method != this->method) {
            this->method = method;
            iscomputed = false;
        }
    };

    void set_pca_components(da_int npc) {
        if (npc != this->npc) {
            this->npc = npc;
            iscomputed = false;
        }
    };

    da_status compute();

    da_status init_results();

    /* get_result (required to be defined by basic_handle) */
    da_status get_result(da_result query, da_int *dim, T *result) {
        da_status status = da_status_success;
        // Don't return anything if PCA compute is not done!
        if (!iscomputed) {
            return da_warn(this->err, da_status_no_data,
                           "pca_compute is not done!. "
                           "Call compute before calling get_results");
        }
        if (this->method == pca_method_corr) {
            return da_error(this->err, da_status::da_status_not_implemented,
                            "PCA using corr method is not yet implemented!");
        }

        //FIXME: Currently assuming da_rinfo query can take all results computed
        da_int rinfo_size = (2 * (this->npc * this->npc) + this->npc + 2);
        if (result == nullptr || *dim <= 0 ||
            ((query == da_result::da_pca_scores ||
              query == da_result::da_pca_components) &&
             *dim < (npc * npc)) ||
            (query == da_result::da_pca_variance && *dim < npc) ||
            (query == da_result::da_rinfo && *dim < rinfo_size)) {
            return da_warn(this->err, da_status::da_status_invalid_array_dimension,
                           "Size of the array is too small, provide an array of at "
                           "least size: " +
                               std::to_string(*dim) + ".");
        }

        switch (query) {
        case da_result::da_rinfo:
            //FIXME: Currently giving all results info if the buffer has space
            for (da_int i = 0; i < rinfo_size; i++)
                result[i] = *(results->scores + i);
            break;
        case da_result::da_pca_scores:
            for (da_int i = 0; i < (npc * npc); i++)
                result[i] = *(results->scores + i);
            break;
        case da_result::da_pca_components:
            for (da_int i = 0; i < (npc * npc); i++)
                result[i] = *(results->components + i);
            break;
        case da_result::da_pca_variance:
            for (da_int i = 0; i < npc; i++)
                result[i] = *(results->variance + i);
            break;
        case da_result::da_pca_total_variance:
            result[0] = results->total_variance;
            break;
        default:
            return da_warn(this->err, da_status_unknown_query,
                           "The requested result could not be queried by this handle.");
        }
        return status;
    };

    da_status get_result(da_result query, da_int *dim, da_int *result) {
        return da_warn(this->err, da_status_unknown_query,
                       "There is no pca implementation done for integer datatye!");
    };
};

/*Deallocate major strucures*/
template <typename T> da_pca<T>::~da_pca() {
    if (svd_data) {
        if (svd_data->colmean != nullptr)
            delete[] svd_data->colmean;
        if (svd_data->u)
            delete[] svd_data->u;
        if (svd_data->sigma)
            delete[] svd_data->sigma;
        if (svd_data->vt)
            delete[] svd_data->vt;
        if (svd_data->A)
            delete[] svd_data->A;
        if (svd_data->iwork)
            delete[] svd_data->iwork;
        delete svd_data;
    }

    if (corr_data)
        delete corr_data;

    if (results) {
        if (results->scores)
            delete[] results->scores;
        if (results->components)
            delete[] results->components;
        if (results->variance)
            delete[] results->variance;
        delete results;
    }
}

/*
    Initialize all the memory required to compute PCA based on inputs
*/
template <typename T>
da_status da_pca<T>::init(da_int n, da_int p, T *dataX, da_int lda) {
    /*Init with success */
    da_status status = da_status_success;
    da_int ldu, ldvt, npc;
    std::string method;

    /*Return error any dimension is less than 2*/
    if (n <= 1 || p <= 1)
        return da_status_invalid_input;

    if (this->opts.get("npc", npc) != da_status_success) {
        return da_error(this->err, da_status_internal_error,
                        "Unexpectedly <number of principal components> option not "
                        "found in the pca option registry.");
    }

    /*Save the A dims, assuming the data is in ColMajor order*/
    this->n = n;
    this->p = p;
    this->lda = lda;
    npc = std::min(npc, std::min(n, p));
    this->npc = npc;

    if (this->opts.set("npc", npc) != da_status_success) {
        return da_error(this->err, da_status_internal_error,
                        "Unexpectedly pca provided an invalid value to the number of "
                        "principal components option");
    }

    if (this->opts.get("pca method", method) != da_status_success) {
        return da_error(
            this->err, da_status_internal_error,
            "Unexpectedly pca provided an invalid value to the pca method option");
    }

    if (method == "svd")
        this->method = pca_method_svd;
    else
        this->method = pca_method_corr;

    /*Initialize with A values*/
    ldu = this->n;
    ldvt = this->npc;

    switch (this->method) {
    case pca_method_svd:
        svd_data->colmean = new T[std::max(n, p)];
        svd_data->u = new T[ldu * std::max(n, p)];
        svd_data->sigma = new T[std::max(n, p)];
        svd_data->vt = new T[ldvt * std::max(n, p)];
        svd_data->iwork = new da_int[12 * std::min(n, p)];
        svd_data->A = new T[std::max(n, p) * std::max(n, p)];

        /*Copy the given input dataX matrix into internal A matrix buffer */
        if (this->lda > n)
            memcpy(svd_data->A, dataX, sizeof(T) * n * p);
        else
            for (da_int i = 0; i < p; i++)
                memcpy(svd_data->A + i * n, dataX + i * lda, sizeof(T) * n);
        break;
    case pca_method_corr:
        //corr_data = new pca_usr_data_corr<T>;
        /*TODO: Yet to implement this method*/
        status = da_status_not_implemented;
        break;
    default:
        status = da_status_invalid_input;
        break;
    }

    status = da_pca<T>::init_results();

    /*Reset init done*/
    initdone = true;

    return status;
}

/*
    Initialize memory for results
*/
template <typename T> da_status da_pca<T>::init_results() {
    results->components = new T[npc * npc];
    results->scores = new T[npc * npc];
    results->variance = new T[npc];
    return da_status_success;
}

/*
    Compute PCA of matrix X (n x p) using user choosen method,
    defult is svd_method and save results 
*/
template <typename T> da_status da_pca<T>::compute() {
    char JOBU, JOBVT, RANGE;
    da_int lwork, ldu, ldvt, INFO, lm, ln, lda;
    da_int il, iu, ns = 1;
    T vu, vl;
    T estworkspace[1];
    T tv = 0;

    if (initdone == false)
        return da_error(this->err, da_status_invalid_pointer, "pca is not initialized!");

    switch (method) {
    case pca_method_svd:
        //Input data matrix X (n x p) is A matrix
        //step 1:  Find column Mean of X
        da_basic_statistics::mean(da_axis_col, n, p, svd_data->A, n, svd_data->colmean);

        //step 2:  Substract column mean from A matrix
        da_basic_statistics::standardize(da_axis_col, n, p, svd_data->A, n,
                                         svd_data->colmean, (T *)nullptr);

        /*Save the total variance of mean centered input A*/
        results->total_variance = tv;

        //step 3:  Find Singular values using SVD
        //Construct SVD args based on inputs
        JOBU = 'V';
        JOBVT = 'V';
        RANGE = 'I';
        ldu = std::max(n, p);
        lda = std::max(n, p);
        INFO = 0;
        lm = n;
        ln = p;
        vl = 0.0;
        vu = 0.0;
        iu = std::min(n, p);
        il = iu - npc + 1;
        if (RANGE == 'A') {
            ns = std::min(n, p);
        } else if (RANGE == 'I') {
            ns = iu - il + 1;
        }
        ldvt = ns;

        //Query gesvdx for optimal work space required
        lwork = -1;

        da::gesvdx(&JOBU, &JOBVT, &RANGE, &lm, &ln, svd_data->A, &lda, &vl, &vu, &il, &iu,
                   &ns, svd_data->sigma, svd_data->u, &ldu, svd_data->vt, &ldvt,
                   estworkspace, &lwork, svd_data->iwork, &INFO);

        //Handle SVD Error
        if (INFO != 0) {
            if (INFO < 0)
                return da_error(this->err, da_status_invalid_input,
                                std::to_string(INFO) +
                                    "th argument had an illegal value. Please verify "
                                    "the input arguments");
            if (INFO > 0)
                return da_error(this->err, da_status_invalid_input,
                                "ith eigen value is not converged or something went "
                                "wrong in svd computation!");
        }

        /*Read the space required*/
        lwork = (da_int)estworkspace[0];

        /*Allocate optimal memory required to compute SVD*/
        svd_data->work = new T[lwork];

        /*Initialize the INFO with failure */
        INFO = -1;

        /*Call gesvdx*/
        da::gesvdx(&JOBU, &JOBVT, &RANGE, &lm, &ln, svd_data->A, &lda, &vl, &vu, &il, &iu,
                   &ns, svd_data->sigma, svd_data->u, &ldu, svd_data->vt, &ldvt,
                   svd_data->work, &lwork, svd_data->iwork, &INFO);

        //Handle SVD Error
        if (INFO != 0) {
            if (INFO < 0)
                return da_error(this->err, da_status_invalid_input,
                                std::to_string(INFO) +
                                    "th argument had an illegal value. Please verify "
                                    "the input arguments");

            if (INFO > 0) {
                if (INFO == (2 * this->n + 1))
                    return da_error(this->err, da_status_internal_error,
                                    "An internal error occured in gesvdx!");
                else
                    return da_error(
                        this->err, da_status_invalid_input,
                        std::to_string(INFO) +
                            "th eigen value is not converged or something went "
                            "wrong in svd computation!");
            }
        }

        //Step 4: Compute and save the results
        //Find the sign of column/row max in U/VT, flip the signs of U & Vt if negative
        //TODO: Simplify for various shapes of A
        for (da_int j = 0; j < n; j++) {
            T *uptr = &svd_data->u[j * n];
            T *vtptr = &svd_data->vt[j];
            T *ptr = svd_flip_u_based ? uptr : vtptr; //load the first column value
            da_int uv_size = svd_flip_u_based ? n : p;
            T colmax = std::abs(ptr[0]);
            da_int maxidx = 0;
            for (da_int i = 1; i < uv_size; i++) {
                T u_t = std::abs(ptr[i]);
                if (u_t > colmax) {
                    colmax = u_t;
                    maxidx = i;
                }
            }
            //If the max value is negative flip the sign of all values
            bool iscolmax_neg = std::signbit(ptr[maxidx]);
            if (iscolmax_neg) {
                for (da_int i = 0; i < n; i++) {
                    uptr[i] = -uptr[i];
                }
                for (da_int i = 0; i < p; i++) {
                    vtptr[i * p] = -vtptr[i * p];
                }
            }
        }

        //Compute Scores (n x n) = U (nxn) * Sigma (n x n) and save
        for (da_int i = 0; i < npc; i++) {
            for (da_int j = 0; j < npc; j++) {
                *(results->scores + i * npc + j) =
                    (*(svd_data->sigma + i) * (*(svd_data->u + i * npc + j)));
            }
        }

        //Save npc
        for (da_int i = 0; i < npc; i++) {
            for (da_int j = 0; j < npc; j++) {
                *(results->components + i * npc + j) = *(svd_data->vt + i * npc + j);
            }
        }

        //compute variance
        //variance = (S**2) / (nsamples-1)
        for (da_int j = 0; j < npc; j++) {
            T sigma = *(svd_data->sigma + j);
            *(results->variance + j) = (sigma * sigma) / (this->n - 1);
        }

        //update flag to true!
        iscomputed = true;

        //Free the temporary memory created for gesvdx workspace
        delete[] svd_data->work;

        break;

    case pca_method_corr:
        //TODO: Yet to implement
        return da_error(this->err, da_status_not_implemented,
                        "PCA using corr method is not yet implemented!");
        break;
    default:
        return da_error(this->err, da_status_invalid_input, "Invalid pca method !");
    }

    return da_status_success;
}

#endif //PCA_HPP
