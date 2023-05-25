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
#include "../basic_statistics/da_mean.hpp"
#include "lapack_templates.hpp"
#include <iostream>

//#define DEBUG_PRINT

/* data struct required to compute pca using svd method */
template <typename T> struct pca_usr_data_svd {
  T *Colmean = nullptr; //Store Column Mean of X
  T *U = nullptr;
  T *Sigma = nullptr;
  T *VT = nullptr;
  T *work = nullptr;
  da_int *iwork = nullptr;
};

/* TODO: yet to define internal data struct required to compute
   compute pca using correlation method */
template <typename T> struct pca_usr_data_corr {

};

/* PCA Results data struct  */
template <typename T> struct pca_results {
  T *eigenvalues = nullptr;
  T *eigenvectors = nullptr;
  T *mean = nullptr;
  T *variance = nullptr;
};

/* PCA Internal Class */
template <typename T> class  da_pca {
  private:

  /*n_samples x n_features = (nxp) */
  da_int n = 0;
  da_int p = 0;

  /*Initialize with default value of 5 but
    Principal components should be r <= min(n,p)
  */
  da_int r = 5;

  /*Set true when init done*/
  bool initdone = false;

  /*Set true when compute done successfully*/
  bool computedone = false;

  /*Define default pca compute method to svd*/
  pca_comp_method method = pca_method_svd;

  /*Defult number of output Components */
  //TODO: Redundent with "r".
  //      Need to remove one of them.
  da_int pca_components = 5;

  /*Data required to compute pca using svd method*/
  pca_usr_data_svd<T> *svd_data = nullptr;

  /*Data required to compute pca using svd method*/
  pca_usr_data_corr<T> *corr_data = nullptr;

  /*pca results*/
  pca_results<T> *results = nullptr;

  /*Copy of the user input pointers*/
  T* InData;

  public:
  da_pca() {
        svd_data = new pca_usr_data_svd<T>;
        corr_data = new pca_usr_data_corr<T>;
        results = new pca_results<T>;
  };

  ~da_pca();

  da_status init(da_int vectors, da_int features,  T* dataX);

  void set_pca_compute_method(pca_comp_method method) {
      method = method;
  };

  void set_pca_components(da_int components) {
     pca_components = r = std::min(components,std::min(n,p));
  };

  da_status free();
  da_status compute();
  da_status init_results();
  da_status free_results();
  da_status get_results();

};


/*Deallocate major strucures*/
template <typename T> da_pca<T>::~da_pca()
{
    if (svd_data){
        da_pca<T>::free();
        delete svd_data;
    }

    if (corr_data) {
        da_pca<T>::free();
        delete corr_data;
    }

    if (results){
        da_pca<T>::free_results();
        delete results;
    }
}
/*
    Initialize all the memory required to compute PCA based on inputs
*/
template <typename T> da_status da_pca<T>::init(da_int vectors,
                                                     da_int features,
                                                     T* dataX)
{
    /*Init with success */
    da_status err_status = da_status_success;
    da_int ldu, ldvt;

    /*Assuming the data is in row major where */
    n = vectors;
    p = features;
    r = std::min(pca_components,std::min(n,p));

    /*Initialize with input values*/
    ldu = std::max(n,p);
    ldvt = std::max(r,p);

    /*Save input buffer address*/
    InData = dataX;

    switch(method) {
        case pca_method_svd:
            /*TODO::Use new rather than std::malloc/aligned_alloc
               u[LDU*M], sigma[N], vt[LDVT*N], a[LDA*N]
            */
            svd_data->Colmean = (T*)std::malloc(n*sizeof(T));
            svd_data->U = (T*)std::malloc(ldu * n * sizeof(T));
            svd_data->Sigma = (T*)std::malloc(std::max(n,p) * sizeof(T));
            svd_data->VT = (T*)std::malloc(ldvt * p * sizeof(T));
            svd_data->iwork = (da_int*)std::malloc(std::min(n,p) * sizeof(da_int));

            if((svd_data->U==NULL) ||
               (svd_data->Sigma==NULL) ||
               (svd_data->VT==NULL) ||
               (svd_data->iwork==NULL) ||
               (svd_data->Colmean==NULL)
               )
            {
                err_status = da_status_memory_error;
                printf(" Unable to create pca_method_svd \n");
                return err_status;
            }
            break;
        case pca_method_corr:
            //corr_data = new pca_usr_data_corr<T>;
            /*TODO: Yet to implement this method*/
             err_status = da_status_not_implemented;
            break;
        default:
            err_status =  da_status_memory_error;
            break;
    }

    err_status = da_pca<T>::init_results();

    /*Reset init done*/
    initdone = true;

    return err_status;
}

/*
    Initialize memory for results
*/
template <typename T> da_status da_pca<T>::init_results()
{
    //TODO: Update the sizes with proper values and may 
    //use new rather than alloc
    results->eigenvalues = (T*)std::malloc(sizeof(T) * pca_components );
    results->eigenvectors = (T*)std::malloc(sizeof(T)* pca_components * pca_components );
    results->mean = (T*)std::malloc(sizeof(T) );
    results->variance = (T*)std::malloc(sizeof(T));

    if((results->eigenvalues == nullptr) ||
       (results->eigenvectors == nullptr) ||
       (results->mean == nullptr) ||
       (results->variance == nullptr))
    {
        return da_status_memory_error;
    }

    return da_status_success;
}

/*
    Free all the memory allocated
*/
template <typename T> da_status da_pca<T>::free()
{
    /*Initilize error with success */
    da_status err_status = da_status_success;

    switch(method) {
        case pca_method_svd:
            if(nullptr!=svd_data->Colmean){
                 std::free(svd_data->Colmean);
                 svd_data->Colmean = nullptr;
            }

            if(nullptr!=svd_data->U){
                 std::free(svd_data->U);
                 svd_data->U = nullptr;
            }

            if(nullptr!=svd_data->Sigma) {
                std::free(svd_data->Sigma);
                svd_data->Sigma = nullptr;
            }

            if(nullptr!=svd_data->VT){
                std::free(svd_data->VT);
                svd_data->VT = nullptr;
            }

            if(nullptr!=svd_data->iwork) {
                std::free(svd_data->iwork);
                svd_data->iwork = nullptr;
            }

        break;

        case pca_method_corr:
        break;

        default:
        break;
    }

    /*Reset the initdone flag*/
    initdone = false;

    return err_status;
}

/*
    Free memory for results
*/
template <typename T> da_status da_pca<T>::free_results()
{
    if(nullptr!=results->mean) std::free(results->mean);
    if(nullptr!=results->variance) std::free(results->variance);
    if(nullptr!=results->eigenvalues) std::free(results->eigenvalues);
    if(nullptr!=results->eigenvectors) std::free(results->eigenvectors);

    return da_status_success;
}

/*
    Compute PAC of matrix X (n x p) using user choosen method,
    defult is svd_method and save results 
*/
template <typename T> da_status da_pca<T>::compute()
{
    char JOBU, JOBVT, RANGE;
    da_int lwork, ldu, ldvt, INFO, lm , ln, lda;
    da_int il, iu, ns=1;
    T vu,vl;
    T estworkspace;

    /*Initilize error with success */
    da_status err_status = da_status_success;

    if(initdone == false) return da_status_invalid_pointer;

    switch(method)
    {
        case pca_method_svd:
            //Input data matrix X (m x n) is input matrix

            //step 1:  Find column Mean of X
            da_colmean(n, p, InData, p, svd_data->Colmean);

#ifdef DEBUG_PRINT
            printf("\nColMean: \n");
            for(da_int i=0;i<p;i++){
                printf("%.4f, ",svd_data->Colmean[i]);
            }
            printf("\n\n");
#endif
            //step 2:  Substract column mean from X
            // X = X - X.mean(axis=0)
            //ToDo: Yet to optimize this substration
            for(da_int i=0;i<n;i++) {
                for(da_int j=0;j<p;j++) {
                    *(InData + i*p + j) -= svd_data->Colmean[j];
                }
            }

            //step 3:  Find Singular values using SVD
            //Construct SVD args based on inputs
            JOBU = 'V';
            JOBVT = 'V';
            RANGE = 'I';
            ldu = std::max(n,p);
            lda = std::max(n,p);
            INFO = 0;
            lm = n;
            ln = p;
            vl = 0.0;
            vu = 0.0;
            iu = std::min(n,p);
            il = iu - pca_components + 1;
            if(RANGE == 'A') ns = std::min(n,p);
            else if(RANGE == 'I') ns = iu-il+1;
            ldvt = ns;

            //Query gesvdx for optimal work space required
            lwork = -1;

#ifdef DEBUG_PRINT
            printf("Estimate gesvdx : \nJOBU: %c JOBVT: %c RANGE:%c lm:%d ln:%d \n\
                    lda:%d vl:%.1f vu:%.1f il:%d iu:%d ns:%d ldu:%d ldvt:%d lwork:%d INFO:%d\n",
                    JOBU,JOBVT,RANGE,lm,ln,lda,vl,vu,il,iu,ns,ldu,ldvt,lwork,INFO);
#endif

            da::gesvdx(&JOBU,
                      &JOBVT,
                      &RANGE,
                      &lm,
                      &ln,
                      InData,
                      &lda,
                      &vl,
                      &vu,
                      &il,
                      &iu,
                      &ns,
                      svd_data->Sigma,
                      svd_data->U,
                      &ldu,
                      svd_data->VT,
                      &ldvt,
                      &estworkspace,
                      &lwork,
                      svd_data->iwork,
                      &INFO );

            //Handle SVD Error
            if(INFO != 0) {
                printf("SVD workspace estimation Failed with error %x \n",INFO);
                if(INFO < 0) err_status = da_status_invalid_input;
                if(INFO > 0) err_status = da_status_internal_error;
            }

            /*Read the space required*/
            lwork = (da_int)estworkspace;

            /*Allocate optimal memory required to compute SVD*/
            svd_data->work = (T*)std::malloc(lwork*sizeof(T));

#ifdef DEBUG_PRINT
            //print the input
            printf("InputData \n");
            for(da_int i=0;i<n;i++) {
                for(da_int j=0;j<p;j++) {
                    printf("%.2f ",*(InData+(i*p+j)));
                }
                printf("\n");
            }
#endif
            /*Call gesvdx*/
            da::gesvdx(&JOBU,
                      &JOBVT,
                      &RANGE,
                      &lm,
                      &ln,
                      InData,
                      &lda,
                      &vl,
                      &vu,
                      &il,
                      &iu,
                      &ns,
                      svd_data->Sigma,
                      svd_data->U,
                      &ldu,
                      svd_data->VT,
                      &ldvt,
                      svd_data->work,
                      &lwork,
                      svd_data->iwork,
                      &INFO );

            //Handle SVD Error
            if(INFO != 0) {
                printf("SVD Failed \n");
                if(INFO < 0) err_status = da_status_invalid_input;
                if(INFO > 0) err_status = da_status_internal_error;
            }

#ifdef DEBUG_PRINT
            printf("\n X (n:%d x p:%d) \n",n,p);
            for(da_int i=0;i<n;i++) {
                for(da_int j=0;j<p;j++) {
                    printf("%.4f ",*( InData + (i*p+j)));
                }
                printf("\n");
            }

            printf("\n Printing U \n");
            for(da_int i=0;i<n;i++) {
                for(da_int j=0;j<n;j++) {
                    printf("%.4f ",*( svd_data->U + (i*n+j)));
                }
                printf("\n");
            }

            printf("\n Printing sigma: ");
            for(da_int i=0;i<p;i++) {
                if(i%8==0) printf("\n");
                printf("%.4f ",svd_data->Sigma[i]);
            }

            printf("\n\n Printing VT: \n");
            for(da_int i=0;i<p;i++) {
                for(da_int j=0;j<p;j++) {
                    printf("%.4f ",*( svd_data->VT + (i*p+j)));
                }
                printf("\n");
            }

            /*Sigma is a diagonal matrix stored in an array*/
            /*U = m x r */
            /*T = U * Sigma , scale U with Sigma */
            /*We can use U with Sigma*/
            printf("\n pca components \n");
            for(da_int i=0;i<n;i++) {
                for(da_int j=0;j<r;j++) {
                    printf("%.4f  ",*(svd_data->U + i*r + j));
                }
                printf("\n");
            }
            printf("\n singular values \n");
            for(da_int j=0;j<r;j++) {
                printf("%.4f  ",*(svd_data->Sigma +j));
            }
            printf("\n");

#endif //DEBUG_PRINT

            //ToDO: May be we can write results directly into 
            //user given output buffers if compute() function
            // can have output buffers from user

            //Step 4: Save the results
            //Save eigen vectors aka components
            for(da_int i=0;i<n;i++) {
                for(da_int j=0;j<p;j++) {
                    *(results->eigenvectors+j+i*p) = *(svd_data->U+i*p+j);
                }
            }

            //Save eigen values aka singular values
            for(da_int j=0;j<p;j++) {
                *(results->eigenvalues + j)= *(svd_data->Sigma + j);
            }

            //compute mean of eigen vectors and save
            //results->mean

            //compute variance and save
            //results->variance


            //Set to true
            computedone = true;

            //Free the memory created for gesvdx workspace
            if(svd_data->work!=NULL){
                std::free(svd_data->work);
                svd_data->work = nullptr;
            }

            //Make the pointer null before closing
            InData = NULL;

            break;

        case pca_method_corr:
            /*TODO: Yet to implement*/
            err_status = da_status_not_implemented;
            break;

        default:
            err_status = da_status_invalid_input;
            break;
    }

    return err_status;
}

/*
    Copy the results to output buffer based on user requirements
*/
template <typename T> da_status da_pca<T>::get_results()
{
    /*Initilize error with success */
    da_status err_status = da_status_success;

    if(initdone==false)
        return da_status_invalid_pointer;

    switch(method)
    {
        case pca_method_svd:
            if(computedone==false){
                return da_status_internal_error;
            }

            break;
        case pca_method_corr:
            err_status = da_status_not_implemented;
            break;
        default:
            err_status = da_status_invalid_input;
            break;
    }

    return err_status;
}

#endif //PCA_HPP
