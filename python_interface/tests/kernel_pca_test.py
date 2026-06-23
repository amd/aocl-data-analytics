# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

"""
Kernel principal component analysis Python test script
"""

import numpy as np
import pytest
from aoclda.factorization import KernelPCA

A = np.array([[1.0, 2.0, -1.0],
              [3.0, -1.0, 0.5],
              [0.5, 1.5, 2.0],
              [-2.0, 0.0, 1.0],
              [1.5, -0.5, -1.5],
              [-1.0, 1.0, 0.0]])

X = np.array([[0.5, -1.0, 1.5],
              [-0.5, 2.0, -1.0],
              [1.0, 0.0, 0.5]])

# Precomputed kernel matrix
# Lifted from kernel_pca_test_data.hpp add_precomputed_square.
K_PRE = np.array([
    [1.00000000000000000e+00, 1.19432968266719619e-01, 6.39278612067075702e-02,
     4.72366552741014689e-01, 1.70361979580257398e-03],
    [1.19432968266719619e-01, 1.00000000000000000e+00, 1.70361979580257398e-03,
     6.87289278790972236e-01, 6.39278612067075702e-02],
    [6.39278612067075702e-02, 1.70361979580257398e-03, 1.00000000000000000e+00,
     1.11089965382423061e-02, 8.04733010124613246e-04],
    [4.72366552741014689e-01, 6.87289278790972236e-01, 1.11089965382423061e-02,
     1.00000000000000000e+00, 2.66490973363554852e-02],
    [1.70361979580257398e-03, 6.39278612067075702e-02, 8.04733010124613246e-04,
     2.66490973363554852e-02, 1.00000000000000000e+00],
])

# Cross-kernel matrix K_cross_pre (4x5), stored column-major in kernel_pca_test_data.hpp.
# Flat column-major: [col0(4), col1(4), col2(4), col3(4), col4(4)] -> reshape (4,5,'F').
K_CROSS_PRE = np.array([
    6.87289278790972236e-01, 1.61634945881658741e-02, 3.24652467358349739e-01,
    5.35261428518990279e-01,
    4.72366552741014689e-01, 1.73773943450445140e-01, 1.83156388887341787e-02,
    1.73773943450445140e-01,
    2.66490973363554852e-02, 5.94621735647209420e-03, 6.87289278790972236e-01,
    5.94621735647209420e-03,
    8.82496902584595344e-01, 1.19432968266719619e-01, 9.30144892106634924e-02,
    5.35261428518990279e-01,
    1.11089965382423061e-02, 7.78800783071404878e-01, 3.18278079650966689e-03,
    3.18278079650966689e-03,
]).reshape((4, 5), order='F')

# ---------------------------------------------------------------------------
# Per-kernel reference data (from kernel_pca_test_data.hpp).
# All 2-D arrays are in row-major form, converted from column-major C++ storage
# via np.array(flat).reshape((rows, cols), order='F').
# ---------------------------------------------------------------------------

# linear_tall: kernel=linear, n=6, p=3, nc=3, fit_inverse_transform=True
_lin_eigenvecs = np.array([
    8.62948669540869462e-02, 6.06456873889861248e-01, -2.05409257584859228e-01,
    -5.52530436625913945e-01, 4.03508842196679263e-01, -3.38320888829854005e-01,
    5.26237765855806838e-01, -4.89792827467531167e-01, -5.16718413116041830e-01,
    -1.33917212983749284e-01, 3.76485404837801518e-01, 2.37705282873712398e-01,
    5.45314575607418472e-01, -1.48924748101957716e-01, 5.35040989459531158e-01,
    -5.07508208089862967e-01, -3.64874355212676194e-01, -5.90482536624547596e-02,
]).reshape((6, 3), order='F')

_lin_scores = np.array([
    3.69212681586973235e-01, 2.59472639079286527e+00, -8.78843730717443372e-01,
    -2.36400207064011569e+00, 1.72641301771466504e+00, -1.44750628873694298e+00,
    1.42191644626689495e+00, -1.32344069891494143e+00, -1.39619475714320584e+00,
    -3.61849908795781716e-01, 1.01727930538722422e+00, 6.42289613199805709e-01,
    1.30497000988612277e+00, -3.56385724306790885e-01, 1.28038471102070939e+00,
    -1.21449713789624814e+00, -8.73165897681538516e-01, -1.41305961022259535e-01,
]).reshape((6, 3), order='F')

_lin_transform = np.array([
    2.82450111998761681e-02, -9.81122101044141104e-01, 4.94629198892678312e-01,
    -1.68857555355862798e+00, 1.75868547965963362e+00, -5.72179687098797740e-01,
    -1.08429359356415156e+00, 7.45342818834525689e-01, -1.97644813707366113e-01,
]).reshape((3, 3), order='F')

_lin_inv_transform = np.array([
    4.48863636363635132e-01, 2.36297928262213919e+00, -6.77952999381563198e-02,
    -2.33219310451453232e+00, 9.95439084724799250e-01, -1.40729359925788500e+00,
    1.27840909090908750e+00, -1.36359771181199863e+00, 8.64950525664811343e-01,
    -3.60138373531232348e-01, -8.92779839208410642e-01, 4.73156307977736668e-01,
    -1.04166666666666607e+00, 2.38945578231292755e-01, 1.62414965986394622e+00,
    7.91241496598640293e-01, -1.49829931972789132e+00, -1.14370748299320424e-01,
]).reshape((6, 3), order='F')

# poly_tall: kernel=poly, n=6, p=3, nc=3, gamma=0.5, degree=2, coef0=1.0,
#            fit_inverse_transform=True
_poly_eigenvecs = np.array([
    -2.73301497257292580e-01, 8.56978266303177860e-01, -3.20438971211782275e-01,
    -1.74561970802561417e-01, 1.19689257418373773e-01, -2.08365084449915722e-01,
    6.71666545903379375e-01, -1.18807267377822101e-01, -5.91475930190690224e-01,
    -2.54605289008694780e-01, 3.42941943313711306e-01, -4.97200026398837841e-02,
    -2.99718444324180933e-01, -1.26502171652715850e-01, -5.88779043511001299e-01,
    6.95073126721668322e-01, 7.87162727563856662e-02, 2.41210260009843552e-01,
]).reshape((6, 3), order='F')

_poly_scores = np.array([
    -1.53116192030838172e+00, 4.80119026446463604e+00, -1.79524794202049609e+00,
    -9.77977234333404777e-01, 6.70554808766644417e-01, -1.16735797656899964e+00,
    2.63488515091744402e+00, -4.66069817745446924e-01, -2.32030485229604055e+00,
    -9.98792778121419422e-01, 1.34532922560364687e+00, -1.95046928358184823e-01,
    -1.10739464613718219e+00, -4.67398087324324973e-01, -2.17541086606143974e+00,
    2.56814444950496013e+00, 2.90839555139325967e-01, 8.91219594878658872e-01,
]).reshape((6, 3), order='F')

_poly_transform = np.array([
    -1.33944437804207808e-02, -1.48199198470137050e+00, 2.33091152752886478e-02,
    -6.76746000481215804e-01, 1.53054680629574791e+00, -1.45303469742128516e-01,
    -6.01868651453384210e-02, -2.04510062494455080e-01, -3.29439161029442096e-01,
]).reshape((3, 3), order='F')

_poly_inv_transform = np.array([
    1.00071347448220460e+00, 2.98491683024966381e+00, 4.97273810833001095e-01,
    -1.97774350930994403e+00, 1.23899294144788152e+00, -8.18122823954347544e-01,
    1.93908154440707436e+00, -9.88637735487921998e-01, 1.47730233574145520e+00,
    8.37917667384892317e-02, -3.49269957374555629e-01, 6.95529495989266500e-01,
    -1.00545617085360872e+00, 4.88794377823636272e-01, 1.95442465916293262e+00,
    9.40184197148184797e-01, -1.21074056415688158e+00, 9.16004108402117267e-02,
]).reshape((6, 3), order='F')

# rbf_tall: kernel=rbf, n=6, p=3, nc=5, gamma=0.5, fit_inverse_transform=True
_rbf_eigenvecs = np.array([
    -1.95028014286023982e-01, -3.94798609311985960e-01, -1.06080900507886405e-01,
    5.77080518678823151e-01, -4.17431923304374508e-01, 5.36258928731447648e-01,
    2.03277015442453690e-01, -4.26037216817467201e-01, 8.01002270085189094e-01,
    -2.16518839920699063e-01, -2.88713023829341664e-01, -7.30102049601350500e-02,
    7.39328180274708813e-01, -5.00222506788661869e-01, -4.00884385569803159e-01,
    -7.07999594271337546e-02, 1.88414024608524189e-01, 4.41646469023658159e-02,
    -4.40908483370571203e-01, -4.93806029526193124e-01, 1.33445035924157157e-01,
    7.60466819753996270e-02, 7.33549793901154268e-01, -8.32699890394634351e-03,
    -1.13883331147902001e-01, 4.35055193133711132e-02, -4.42681713273022498e-02,
    -6.65308400788504395e-01, 4.61905597439004820e-02, 7.33763824206437154e-01,
]).reshape((6, 5), order='F')

_rbf_scores = np.array([
    -2.08131744945377534e-01, -4.21324719727715846e-01, -1.13208366546269992e-01,
    6.15853962141346578e-01, -4.45478742688867613e-01, 5.72289611766884421e-01,
    2.03883472070575555e-01, -4.27308256208729254e-01, 8.03391980179940735e-01,
    -2.17164802206682828e-01, -2.89574370236584555e-01, -7.32280235985198896e-02,
    7.38582917974258235e-01, -4.99718269311865082e-01, -4.00480283538559800e-01,
    -7.07285912011655915e-02, 1.88224098303585108e-01, 4.41201277737471587e-02,
    -4.29525923643774010e-01, -4.81057858791155646e-01, 1.29999998804345873e-01,
    7.40834493948139133e-02, 7.14612362083490660e-01, -8.11202784772041552e-03,
    -1.00063985381965842e-01, 3.82262760030636708e-02, -3.88963828501703995e-02,
    -5.84575542530803571e-01, 4.05854961249403692e-02, 6.44724138634935939e-01,
]).reshape((6, 5), order='F')

_rbf_transform = np.array([
    -3.23178684524370874e-02, 6.82903208331758355e-02, -8.32522890797626203e-02,
    1.99334687074597637e-02, 5.21438300158077322e-02, 1.86186052944305354e-02,
    -3.49148012957224835e-02, 2.46382377694858545e-01, -2.99402012077579008e-02,
    -5.42760827410980126e-03, -1.47711471467210448e-01, 2.75309134599570210e-02,
    -1.95356460243256425e-02, 2.11250004018720017e-01, 3.90323844419649887e-02,
]).reshape((3, 5), order='F')

_rbf_inv_transform = np.array([
    5.70410869347726335e-01, 1.35383625951921815e+00, 3.71077723430620565e-01,
    -6.29830732520272996e-01, 7.83702250371070219e-01, -2.72153610068759488e-01,
    9.49595944846236550e-01, -2.18934105710737875e-01, 7.58774381441299051e-01,
    1.98327530606759034e-01, -1.94650942724729735e-02, 5.66730786744574444e-01,
    -3.36029209124823147e-01, 2.42677256950412423e-01, 8.31193105855172809e-01,
    4.42036204914346753e-01, -5.26291152015300079e-01, 9.08113776061106187e-02,
]).reshape((6, 3), order='F')

# sigmoid_tall: kernel=sigmoid, n=6, p=3, nc=2, gamma=1e-4, coef0=0.0,
#               fit_inverse_transform=True
_sig_eigenvecs = np.array([
    8.62949115563199365e-02, 6.06456808608454967e-01, -2.05409264444078393e-01,
    -5.52530455661531694e-01, 4.03508895540004076e-01, -3.38320895599168892e-01,
    5.26237723939191993e-01, -4.89792838636811112e-01, -5.16718462240296605e-01,
    -1.33917138640366307e-01, 3.76485365465003496e-01, 2.37705350113278202e-01,
]).reshape((6, 2), order='F')

_sig_scores = np.array([
    3.69212849071820720e-03, 2.59472594741807852e-02, -8.78843704494080245e-03,
    -2.36400200260481162e-02, 1.72641313678057497e-02, -1.44750622617158247e-02,
    1.42191620394502907e-02, -1.32344060897160304e-02, -1.39619476315220453e-02,
    -3.61849675073262506e-03, 1.01727910666608422e-02, 6.42289736585955912e-03,
]).reshape((6, 2), order='F')

_sig_transform = np.array([
    2.82449857691371856e-04, -9.81121952271260013e-03, 4.94629266296353880e-03,
    -1.68857568707467577e-02, 1.75868552039861945e-02, -5.72179857928879071e-03,
]).reshape((3, 2), order='F')

_sig_inv_transform = np.array([
    3.75352495440569067e-08, 4.49280876017674677e-07, -1.21939425553838631e-07,
    -3.83634805749934504e-07, 2.67822293164069096e-07, -2.49064187422028001e-07,
    1.11240925536065784e-08, -1.71531370428425792e-07, 1.73981518425376572e-08,
    1.21687385507875942e-07, -7.22509579655287595e-08, 9.35726984899345140e-08,
    -1.17527643296565869e-07, -6.05423046184767283e-08, 1.45260655306254886e-07,
    1.61188735524285965e-07, -1.68647860202797168e-07, 4.02684172872988958e-08,
]).reshape((6, 3), order='F')

# precomputed_square: kernel=precomputed, n=5, p=5, nc=3, no inverse transform
_pre_eigenvecs = np.array([
    -1.61015148844806999e-01, -4.15181464190497618e-01, 5.68390845682762991e-01,
    -4.85280034656519976e-01, 4.93085802009061269e-01,
    -3.38511806819172589e-01, 2.04741665273422246e-01, -5.74637504707371405e-01,
    -7.98511147375696555e-03, 7.16392757726878937e-01,
    7.45094714157353066e-01, -5.11782671380051957e-01, -3.76798798008218638e-01,
    -5.20322370388089719e-02, 1.95518992269726188e-01,
]).reshape((5, 3), order='F')

_pre_scores = np.array([
    -1.84244900102900427e-01, -4.75079940882352947e-01, 6.50392930935710933e-01,
    -5.55291673787787321e-01, 5.64223583837329401e-01,
    -3.42764313485112992e-01, 2.07313703467753435e-01, -5.81856307035801934e-01,
    -8.08542330656865892e-03, 7.25392340359730436e-01,
    6.91426482869123227e-01, -4.74919611885669057e-01, -3.49658456443050270e-01,
    -4.82844207158856717e-02, 1.81436006175481418e-01,
]).reshape((5, 3), order='F')

_pre_transform = np.array([
    -4.56697623521052032e-01, 3.90219027706195820e-01, 4.18634774306407165e-01,
    -1.93403507535594599e-01,
    -1.42281585252488241e-01, 5.82620229941195000e-01, -4.87158223117837463e-01,
    -1.42953192358950376e-01,
    2.39818883308549030e-01, 7.85696196551257942e-02, -2.65934589069756443e-02,
    3.08633415849249360e-01,
]).reshape((4, 3), order='F')

# ---------------------------------------------------------------------------
# Kernel config dictionary
# ---------------------------------------------------------------------------

KERNEL_CONFIGS = {
    "linear": {
        "constructor_kwargs": {"kernel": "linear", "fit_inverse_transform": True, 'n_components': 3, "eigensolver": "syevd"},
        "A": A,
        "X": X,
        "n_samples": 6, "n_features": 3, "nc": 3,
        "expected_eigenvalues": np.array([
            1.83055715040013816e+01, 7.30102995247190556e+00, 5.72673187686005924e+00]),
        "expected_eigenvectors": _lin_eigenvecs,
        "expected_scores": _lin_scores,
        "expected_transform": _lin_transform,
        "expected_inverse_transform": _lin_inv_transform,
        "expected_gamma": 1.0 / 3.0,
        "tol_multiplier": 1000,
    },
    "poly": {
        "constructor_kwargs": {
            "n_components": 3,
            "kernel": "poly",
            "gamma": 0.5,
            "degree": 2,
            "coef0": 1.0,
            "fit_inverse_transform": True,
        },
        "A": A,
        "X": X,
        "n_samples": 6, "n_features": 3, "nc": 3,
        "expected_eigenvalues": np.array([
            3.13876078230150242e+01, 1.53891964842798430e+01, 1.36514222371583678e+01]),
        "expected_eigenvectors": _poly_eigenvecs,
        "expected_scores": _poly_scores,
        "expected_transform": _poly_transform,
        "expected_inverse_transform": _poly_inv_transform,
        "expected_gamma": 0.5,
        "tol_multiplier": 1000,
    },
    "rbf": {
        "constructor_kwargs": {"kernel": "rbf", "gamma": 0.5, "fit_inverse_transform": True, "n_components": 5, "eigensolver": "syevd"},
        "A": A,
        "X": X,
        "n_samples": 6, "n_features": 3, "nc": 5,
        "expected_eigenvalues": np.array([
            1.13889228957286082e+00, 1.00597570048067242e+00, 9.97984963380812107e-01,
            9.49034172788480568e-01, 7.72031940305863085e-01]),
        "expected_eigenvectors": _rbf_eigenvecs,
        "expected_scores": _rbf_scores,
        "expected_transform": _rbf_transform,
        "expected_inverse_transform": _rbf_inv_transform,
        "expected_gamma": 0.5,
        "tol_multiplier": 1000,
    },
    "sigmoid": {
        "constructor_kwargs": {
            "n_components": 2,
            "kernel": "sigmoid",
            "gamma": 1e-4,
            "coef0": 0.0,
            "fit_inverse_transform": True,
            "eigensolver": "syevd"
        },
        "A": A,
        "X": X,
        "n_samples": 6, "n_features": 3, "nc": 2,
        "expected_eigenvalues": np.array([
            1.83055691890268539e-03, 7.30102862710383868e-04]),
        "expected_eigenvectors": _sig_eigenvecs,
        "expected_scores": _sig_scores,
        "expected_transform": _sig_transform,
        "expected_inverse_transform": _sig_inv_transform,
        "expected_gamma": 1e-4,
        "tol_multiplier": 5000,
    },
    "precomputed": {
        "constructor_kwargs": {"n_components": 3, "kernel": "precomputed", "eigensolver": "auto"},
        "A": K_PRE,
        "X": K_CROSS_PRE,
        "n_samples": 5, "n_features": 5, "nc": 3,
        "expected_eigenvalues": np.array([
            1.30935519359013486e+00, 1.02528252980439105e+00, 8.61130664749314767e-01]),
        "expected_eigenvectors": _pre_eigenvecs,
        "expected_scores": _pre_scores,
        "expected_transform": _pre_transform,
        "expected_inverse_transform": None,
        "expected_gamma": 1.0 / 5.0,
        "tol_multiplier": 1000,
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "numpy_precision",
    [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, 'object'])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
def test_kernel_pca_all_dtypes(numpy_precision, numpy_order):
    """
    Test it runs when supported/unsupported C-interface type is provided.
    """
    A_local = np.array([[1.3, 2.53, 3.86], [2.4, 5.5, 4.5], [3.33, 6.21, 1.76]],
                       dtype=numpy_precision, order=numpy_order)
    X_local = np.array([[1.2, 1.1, 4.3], [3.333, 2.6, 3.4], [0.3, 2.2, 3.8],
                        [1.8, 0.7, 1.3]], dtype=numpy_precision, order=numpy_order)

    kpca = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True)
    kpca.fit(A_local)
    kpca.inverse_transform(kpca.transform(X_local))


@pytest.mark.parametrize("numpy_precision", [np.float32])
@pytest.mark.parametrize("numpy_orders", [("C", "F"), ("F", "C")])
def test_kernel_pca_multiple_orders(numpy_precision, numpy_orders):
    """
    Test it runs when arrays of multiple orders are provided.
    """
    A_local = np.array([[1.3, 2.53, 3.86], [2.4, 5.5, 4.5], [3.33, 6.21, 1.76]],
                       dtype=numpy_precision, order=numpy_orders[0])
    X_local = np.array([[1.2, 1.1, 4.3], [3.333, 2.6, 3.4], [0.3, 2.2, 3.8],
                        [1.8, 0.7, 1.3]], dtype=numpy_precision, order=numpy_orders[1])

    kpca = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True)
    kpca.fit(A_local)
    with pytest.warns(UserWarning):
        kpca.inverse_transform(kpca.transform(X_local))
    A_local = np.array(A_local, order=numpy_orders[1])
    with pytest.warns(UserWarning):
        kpca.fit(A_local)


@pytest.mark.parametrize(
    "numpy_precisions", [('float32', 'float64'), ('float64', 'float32')])
@pytest.mark.parametrize("numpy_order", ["C"])
def test_kernel_pca_multiple_dtypes(numpy_precisions, numpy_order):
    """
    Test it runs when arrays of multiple dtypes are provided.
    """
    A_local = np.array([[1.3, 2.53, 3.86], [2.4, 5.5, 4.5], [3.33, 6.21, 1.76]],
                       dtype=numpy_precisions[0], order=numpy_order)
    X_local = np.array([[1.2, 1.1, 4.3], [3.333, 2.6, 3.4], [0.3, 2.2, 3.8],
                        [1.8, 0.7, 1.3]], dtype=numpy_precisions[1], order=numpy_order)

    kpca = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True)
    kpca.fit(A_local)
    kpca.inverse_transform(kpca.transform(X_local))
    A_local = np.array(A_local, dtype=numpy_precisions[1])
    kpca.fit(A_local)


@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("kernel_name",
                         ["linear", "poly", "rbf", "sigmoid", "precomputed"])
def test_kernel_pca_functionality(numpy_precision, numpy_order, kernel_name):
    """
    Test the functionality of the Kernel PCA Python wrapper
    """
    cfg = KERNEL_CONFIGS[kernel_name]
    tol = np.finfo(numpy_precision).eps * cfg["tol_multiplier"]

    A_local = np.array(cfg["A"], dtype=numpy_precision, order=numpy_order)
    X_local = np.array(cfg["X"], dtype=numpy_precision, order=numpy_order)

    kpca = KernelPCA(**cfg["constructor_kwargs"])
    kpca.fit(A_local)

    # Check eigenvalues (always positive; direct comparison)
    assert np.all(np.abs(kpca.eigenvalues - cfg["expected_eigenvalues"]) < tol)

    # Check eigenvectors shape and values (abs for sign ambiguity)
    assert kpca.eigenvectors.shape == (cfg["n_samples"], cfg["nc"])
    assert np.all(
        np.abs(np.abs(kpca.eigenvectors) - np.abs(cfg["expected_eigenvectors"])) < tol)

    # Check scores shape and values (abs for sign ambiguity)
    assert kpca.scores.shape == (cfg["n_samples"], cfg["nc"])
    assert np.all(np.abs(np.abs(kpca.scores) - np.abs(cfg["expected_scores"])) < tol)

    # Check integer dimension properties
    assert kpca.n_samples == cfg["n_samples"]
    assert kpca.n_features == cfg["n_features"]
    assert kpca.n_components == cfg["nc"]

    # Check gamma_ (scalar)
    assert abs(kpca.gamma_ - cfg["expected_gamma"]) < tol

    # Check transform shape and values (abs for sign ambiguity)
    X_transform = kpca.transform(X_local)
    assert X_transform.shape == (cfg["X"].shape[0], cfg["nc"])
    assert np.all(
        np.abs(np.abs(X_transform) - np.abs(cfg["expected_transform"])) < tol)

    # Check output array order matches input
    assert A_local.flags.f_contiguous == kpca.transform(X_local).flags.f_contiguous

    # Check inverse_transform and dual_coef when available
    if cfg["expected_inverse_transform"] is not None:
        Y_inv = kpca.inverse_transform(kpca.scores)
        assert np.all(np.abs(Y_inv - cfg["expected_inverse_transform"]) < tol)

        assert kpca.dual_coef.shape == (cfg["n_samples"], cfg["n_features"])


@pytest.mark.parametrize("da_precision, numpy_precision", [
    ("double", np.float64), ("single", np.float32),
])
def test_kernel_pca_error_exits(da_precision, numpy_precision):
    """
    Test error exits in the Kernel PCA Python wrapper
    """
    A_local = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=numpy_precision)

    # Invalid kernel string
    with pytest.raises(RuntimeError):
        KernelPCA(kernel='invalid_kernel')

    # Wrong-dimension transform input (2 features instead of 3)
    kpca = KernelPCA(n_components=2, kernel='linear')
    kpca.fit(A_local)
    B = np.array([[1, 1], [2, 2]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        kpca.transform(B)

    # inverse_transform when fit_inverse_transform=False
    kpca = KernelPCA(n_components=2, kernel='linear')
    kpca.fit(A_local)
    with pytest.raises(RuntimeError):
        kpca.inverse_transform(kpca.scores)

    # fit_inverse_transform=True with precomputed kernel is not supported
    K_local = np.array([[1.0, 0.5, 0.2],
                        [0.5, 1.0, 0.3],
                        [0.2, 0.3, 1.0]], dtype=numpy_precision)
    kpca = KernelPCA(n_components=2, kernel='precomputed', fit_inverse_transform=True)
    with pytest.raises(RuntimeError):
        kpca.fit(K_local)


@pytest.mark.parametrize("copy_X", [True, False])
@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_copy_X_and_X_fit(copy_X, numpy_precision):
    """X_fit_ is accessible after fit() for both copy_X=True and copy_X=False."""
    A_local = np.array(A, dtype=numpy_precision)
    kpca = KernelPCA(n_components=2, kernel='rbf', copy_X=copy_X)
    kpca.fit(A_local)

    X_fit = kpca.X_fit_
    assert X_fit is not None
    assert X_fit.shape == A_local.shape
    # X_fit_ is a straight memcpy — values must be bit-for-bit identical
    assert np.array_equal(X_fit, A_local)


@pytest.mark.parametrize("copy_X", [True, False])
def test_copy_X_transform_after_fit(copy_X):
    """transform() works correctly after fit() regardless of copy_X."""
    kpca = KernelPCA(n_components=2, kernel='rbf', copy_X=copy_X)
    # Fit with a local array that goes out of scope after the call
    kpca.fit(np.array(A, dtype=np.float64))
    # If copy_X=False, py::keep_alive must keep the buffer alive here
    result = kpca.transform(np.array(X, dtype=np.float64))
    assert result.shape == (X.shape[0], 2)


@pytest.mark.parametrize("kernel", ['linear', 'rbf', 'poly', 'sigmoid'])
@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_kernel_pca_randomized_solver(numpy_precision, kernel):
    """
    Test that KernelPCA with randomized eigensolver works
    """
    # not concerned with accuracy here, just check API works
    rng = np.random.default_rng(42)
    a = rng.standard_normal((50, 10)).astype(numpy_precision)
    kpca_rand = KernelPCA(n_components=3, kernel=kernel, eigensolver='randomized',
                          n_oversamples=5, power_iterations=4, fit_inverse_transform=True)
    kpca_rand.fit(a)
    b = kpca_rand.transform(a)
    kpca_rand.inverse_transform(b)
