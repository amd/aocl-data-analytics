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

#include "aoclda.h"
#include "../basic_statistics/moment_statistics.hpp"
#include "lapack_templates.hpp"
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
template <typename T> class da_pca {
  private:
    /*n_samples x n_features = (nxp) */
    da_int n = 0;
    da_int p = 0;

    /*Set true when init done*/
    bool initdone = false;

    /*Set true when compute done successfully*/
    bool iscomputed = false;

    /*Define default pca compute method to svd*/
    pca_comp_method method = pca_method_svd;

    /*Defult number of output Components */
    //TODO: Redundent with "r".
    //                              Need to remove one of them.
    da_int ncomponents = 5;

    /*Data required to compute pca using svd method*/
    pca_usr_data_svd<T> *svd_data = nullptr;

    /*Data required to compute pca using corr method*/
    pca_usr_data_corr<T> *corr_data = nullptr;

    /*pca results*/
    pca_results<T> *results = nullptr;

    /* pointer to error trace */
    da_errors::da_error_t *err = nullptr;

  public:
    da_pca() {
        svd_data = new pca_usr_data_svd<T>;
        corr_data = new pca_usr_data_corr<T>;
        results = new pca_results<T>;
    };

    ~da_pca();

    da_status init(da_int n, da_int p, T *dataX);

    void set_pca_compute_method(pca_comp_method method) { method = method; };

    void set_pca_components(da_int ncomponents) {
        ncomponents = std::min(ncomponents, std::min(n, p));
    };

    da_status compute();
    da_status init_results();
    da_status get_results(T *output, pca_results_flags flags);
};

/*Deallocate major strucures*/
template <typename T> da_pca<T>::~da_pca() {
    if (svd_data) {
        if (svd_data->colmean != nullptr)
            delete svd_data->colmean;
        if (svd_data->u)
            delete svd_data->u;
        if (svd_data->sigma)
            delete svd_data->sigma;
        if (svd_data->vt)
            delete svd_data->vt;
        if (svd_data->A)
            delete svd_data->A;
        if (svd_data->iwork)
            delete svd_data->iwork;
        delete svd_data;
    }

    if (corr_data)
        delete corr_data;

    if (results) {
        if (results->scores)
            delete results->scores;
        if (results->components)
            delete results->components;
        if (results->variance)
            delete results->variance;
        delete results;
    }
}

/*
    Initialize all the memory required to compute PCA based on inputs
*/
template <typename T> da_status da_pca<T>::init(da_int n, da_int p, T *dataX) {
    /*Init with success */
    da_status status = da_status_success;
    da_int ldu, ldvt;

    /*Save the A dims, assuming the data is in ColMajor order*/
    this->n = n;
    this->p = p;
    this->ncomponents = std::min(this->ncomponents, std::min(n, p));

    /*Initialize with A values*/
    ldu = this->n;
    ldvt = this->ncomponents;

    switch (method) {
    case pca_method_svd:
        svd_data->colmean = new T[std::max(n, p)];
        svd_data->u = new T[ldu * std::max(n, p)];
        svd_data->sigma = new T[std::max(n, p)];
        svd_data->vt = new T[ldvt * std::max(n, p)];
        svd_data->iwork = new da_int[12 * std::min(n, p)];
        svd_data->A = new T[std::max(n, p) * std::max(n, p)];

        /*Copy A buffer address*/
        memcpy(svd_data->A, dataX, sizeof(T) * n * p);

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
    results->components = new T[ncomponents * ncomponents];
    results->scores = new T[ncomponents * ncomponents];
    results->variance = new T[ncomponents];
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
    T estworkspace;
    T tv = 0;

    /*Initilize error with success */
    da_status status = da_status_success;

    if (initdone == false)
        return da_error(this->err, da_status_invalid_pointer, "pca is not initialized!");

    switch (method) {
    case pca_method_svd:
        //Input data matrix X (n x p) is A matrix

        //step 1:  Find column Mean of X
        da_colmean(n, p, svd_data->A, p, svd_data->colmean);

        //step 2:  Substract column mean from A matrix
        // A[][] = A[][] - colmean[]
        //TODO: Write an utility function
        for (da_int i = 0; i < n; i++) {
            for (da_int j = 0; j < p; j++) {
                T mean_minus_a = (*(svd_data->A + i * p + j) - *(svd_data->colmean + j));
                *(svd_data->A + i * p + j) = mean_minus_a;
                tv += (mean_minus_a * mean_minus_a);
            }
        }
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
        lm = p;
        ln = n;
        vl = 0.0;
        vu = 0.0;
        iu = std::min(n, p);
        il = iu - ncomponents + 1;
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
                   &estworkspace, &lwork, svd_data->iwork, &INFO);

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
        lwork = (da_int)estworkspace;

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

        //Step 4: Save the results
        //TODO: May needs to do the sign flip
        //Scores (n x n) = U (nxn) * Sigma (n x n)
        for (da_int i = 0; i < ncomponents; i++) {
            for (da_int j = 0; j < ncomponents; j++) {
                *(results->scores + i * ncomponents + j) =
                    (*(svd_data->sigma + i) * (*(svd_data->vt + i * ncomponents + j)));
            }
        }

        //Save ncomponents
        for (da_int i = 0; i < ncomponents; i++) {
            for (da_int j = 0; j < ncomponents; j++) {
                *(results->components + i * ncomponents + j) =
                    *(svd_data->u + i * ncomponents + j);
            }
        }

        //compute variance
        //variance = (S**2) / (nsamples-1)
        for (da_int j = 0; j < ncomponents; j++) {
            T sigma = *(svd_data->sigma + j);
            *(results->variance + j) = (sigma * sigma) / (this->n - 1);
        }

        //update flag to true!
        iscomputed = true;

        //Free the temporary memory created for gesvdx workspace
        if (svd_data->work)
            delete svd_data->work;

        break;

    case pca_method_corr:
        /*TODO: Yet to implement*/
        return da_error(this->err, da_status_not_implemented,
                        "PCA using corr method is not yet implemented!");
        break;

    default:
        return da_error(this->err, da_status_invalid_input, "Invalid pca method !");

        break;
    }

    return status;
}

/*
    Copy the requested results to user buffer if compute is already done
    else return with error
*/
template <typename T>
da_status da_pca<T>::get_results(T *output, pca_results_flags flags) {
    /*Initilize error with success */
    da_status status = da_status_success;

    if (initdone == false)
        return da_error(this->err, da_status_invalid_pointer, "PCA is not initialized!");

    switch (method) {
    case pca_method_svd:
        if (iscomputed == false) {
            return da_error(this->err, da_status_out_of_date, "PCA is not computed!");
        }
        if (output == nullptr) {
            return da_error(this->err, da_status_invalid_pointer,
                            "Given output pointer is invalid");
        }

        /*Default provide ncomponents if flags are not properly*/
        if (flags < pca_components || flags > 0xf)
            flags = pca_components;

        /*Copy the requested output*/
        if (flags & pca_components)
            for (da_int i = 0; i < (ncomponents * ncomponents); i++)
                *output++ = *(results->components + i);
        if (flags & pca_scores)
            for (da_int i = 0; i < (ncomponents * ncomponents); i++)
                *output++ = *(results->scores + i);
        if (flags & pca_variance)
            for (da_int i = 0; i < ncomponents; i++)
                *output++ = *(results->variance + i);
        if (flags & pca_total_variance)
            *output++ = results->total_variance;
        break;

    case pca_method_corr:
        return da_error(this->err, da_status_not_implemented,
                        "PCA using corr method is not yet implemented!");
        break;

    default:
        return da_error(this->err, da_status_invalid_input, "Invalid pca method !");
        break;
    }

    return status;
}

#endif //PCA_HPP
