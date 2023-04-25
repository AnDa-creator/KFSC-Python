"""
    Large-Scale Subspace Clustering via k-Factorization. KDD 2021.
"""
from typing import Optional
import numpy as np
import numpy.matlib
from sklearn import preprocessing  # to normalise existing X
from sklearn.cluster import KMeans
from scipy import sparse
from soyclustering import SphericalKMeans
from numba import jit, prange

def KFSC(X, k, d, lamda, options: Optional[dict] = None, use_numba=False):
    """
    Large-Scale Subspace Clustering via k-Factorization. KDD 2021.
    """
    maxiter = 500 if "maxiter" not in options.keys() else options["maxiter"]
    iter_D = 5 if "iter_D" not in options.keys() else options["iter_D"]
    tol = 1e-4 if "tol" not in options.keys() else options["tol"]
    solver = 2 if "solver" not in options.keys() else options["solver"]
    init_type = 'k-means' if "init_type" not in options.keys(
    ) else options["init_type"]
    nrep_kmeans = 100 if "nrep_kmeans" not in options.keys(
    ) else options["nrep_kmeans"]
    obj_all = 0 if "obj_all" not in options.keys() else options["obj_all"]
    classifier = 'abs' if "classifier" not in options.keys(
    ) else options["classifier"]

    if solver == 0:
        print('Solve k-FSC by Jacobi optimization ...')
    elif solver == 1:
        print('Solve k-FSC by Gauss-Seidel optimization ...')
    elif solver == 2:
        print('Solve k-FSC by accelerated Gauss-Seidel optimization ...')

    m, n = np.shape(X)
    # X = np.divide(X, np.matlib.repmat(np.sqrt(np.sum(X**2, axis=0)), m, 1))
    X = preprocessing.normalize(X, norm='l2', axis=0)
    D, C = initial_DC(X, m, d, k, init_type, nrep_kmeans, use_numba=use_numba)

    dfC = np.zeros(np.shape(C))
    Q = np.zeros((maxiter, k))
    i = 0
    loss = np.zeros(maxiter)

    while i < maxiter:
        i += 1
        if len(np.where(np.sum(np.abs(C)) == 0)[0]) >= n*0.1:
            lamda = lamda/2
            print('Too large lambda! Restart with lambda=', lamda)
            D, C = initial_DC(X, m, d, k, init_type, nrep_kmeans, use_numba=use_numba)
            i = 1
            dfC = np.zeros(np.shape(C))
            Q = np.zeros((maxiter, k))

        # update C by Jacobi or Gauss-Seidel
        if solver == 0:
            C_new = update_C_Jacobi(X, D, C, d, k, lamda)
        elif solver == 1:
            C_new = update_C_GaussSeidel(X, D, C, d, k, lamda)
        elif solver == 2:
            C_new, Q = update_C_GaussSeidel_extrop(
                X, D, C, d, k, lamda, dfC, Q, i)
            dfC = C-C_new

        # update D by GD
        D_new = update_D(X, D, C_new, iter_D)

        # check convergence
        dC = np.linalg.norm(C_new-C, 'fro')/np.linalg.norm(C, 'fro')
        dD = np.linalg.norm(D_new-D, 'fro')/np.linalg.norm(D, 'fro')
        isstop = (max(dC, dD) < tol)

        loss[i-1] = 0.5*np.linalg.norm(X-D_new@C_new, 'fro')**2 + lamda * np.sum(
                np.float_power(np.sum(np.reshape(C_new, (d, k*n), order='F')**2, axis=0), 0.5))
        
        # print loss and other information
        if i <= 3 or np.mod(i, 50) == 0 or isstop or obj_all:
            print("iter: %d, loss: %f, dC: %f, dD: %f" % (i, loss[i-1], dC, dD))
        if isstop:
            print('Converged!')
            break
        if i == maxiter:
            print('Max iter reached!')
            break
        C = C_new
        D = D_new

    if classifier == 'abs':
        print('Use absolute value of C as the classifier!')
        Y = np.zeros((k, n))
        for i in range(1, k+1):
            Y[i-1, :] = np.sum(np.abs(C[(i-1)*d:i*d, :])**2, axis=0)

        L = np.max(Y, axis=0)
    elif classifier == 're':
        print('Assign clusters by reconstruction error')
        E = np.zeros((k, n))
        for i in range(1, k+1):
            Di = D[:, (i-1)*d:i*d]
            C_t = np.linalg.inv(Di.T@Di + 1e-5 * np.eye(d))@Di.T@X
            E[i-1, :] = np.sum((X-Di@C_t)**2, axis=0)

        L = np.argmin(E, axis=0)

    output = {'D': D, 'C': C, 'loss': loss}
    return L, output

@jit(parallel=True, forceobj=True)
def initial_DC(X, m, d, k, init_type, nrep_kmeans, use_numba=False):
    """
    Initialize D and C by k-means or random initialization.
    input:  X: data matrix, m x n
            m: number of rows of D
            d: number of rows of each column of C
            k: number of columns of C
            init_type: 'random' or 'k-means'
            nrep_kmeans: number of replicates for k-means
    output: D: initialized dictionary, m x d*k
            C: initialized coefficient matrix, d*k x n
    """
    X0 = X
    my_generator = np.random.default_rng()
    idxs = my_generator.choice(X.shape[1],
                               min(X.shape[1], 50000), replace=False)
    X = X[:, idxs]
    D = np.zeros((m, d*k))

    if init_type == "random":
        D = np.random.randn(m, d*k)
    elif init_type == "k-means" or init_type == "k-means-cos":
        print("Initializing D by k-means algorithm ({} replicates)...".format(nrep_kmeans))
        if init_type == "k-means":
            X_Norm = preprocessing.normalize(X.T, norm='l2')
            kmeans_x = KMeans(n_clusters=k, n_init=nrep_kmeans).fit(X_Norm)
            dist = kmeans_x.transform(X_Norm)**2
        elif init_type == "k-means-cos":
            X_Norm = preprocessing.normalize(X.T, norm='l2')
            sX_norm = sparse.csr_matrix(X_Norm)
            clusterer_array = []
            inertia_array = []
            if use_numba:
                for _ in prange(nrep_kmeans):
                    kclusterer_now = SphericalKMeans(n_clusters=k,verbose=0,
                                                    init='k-means++').fit(sX_norm)
                    clusterer_array.append(kclusterer_now)
                    inertia_array.append(kclusterer_now.inertia_)
            else:
                for _ in range(nrep_kmeans):
                    kclusterer_now = SphericalKMeans(n_clusters=k,verbose=0,
                                                    init='k-means++').fit(sX_norm)
                    clusterer_array.append(kclusterer_now)
                    inertia_array.append(kclusterer_now.inertia_)
            kclusterer = clusterer_array[np.argmin(inertia_array)]
            dist = kclusterer.transform(sX_norm)**2            
            # skm = SphericalKMeans(n_clusters=k, n_init=nrep_kmeans).fit(X_Norm)
            # dist = skm.transform(X_Norm)**2


        idx = np.argsort(dist, axis=0)
        for i in range(1, k+1):
            temp = X[:, idx[0:d, i - 1]]
            if m < d:
                D[:, (i-1)*d:i*d] = temp
            else:
                u, _, _ = np.linalg.svd(temp)
                D[:, (i-1)*d:i*d] = u[:, 0:d]

    ld = np.float_power(np.sum(D**2, axis=0), 0.5)
    idx = np.where(ld > 1)[0]
    D[:, idx] = np.divide(D[:, idx], np.matlib.repmat(ld[idx], m, 1))
    # D[:, idx] = preprocessing.normalize(D[:, idx], norm='l2')
    C = np.linalg.inv(D.T@D + 1e-5 * np.eye(d*k))@D.T@X0
    return D, C


def update_C_Jacobi(X, D, C, d, k, lamda):
    """
    Update C by Jacobi iteration.
    input:  X: data matrix, m x n
            D: dictionary, m x d*k
            C: coefficient matrix, d*k x n
            d: number of rows of each column of C
            k: number of columns of C
            lamda: regularization parameter
    output: C_new: updated coefficient matrix, d*k x n
    """
    gC = (-D.T) @ (X - D @ C)
    C_new = C.copy()
    tau = 1.0*np.linalg.norm(D, 2)**2
    for j in range(1, k+1):
        temp = C[(j-1)*d:j*d, :] - gC[(j-1)*d:j*d, :]/tau
        C_new[(j-1)*d:j*d, :] = solve_L21(temp, lamda/tau)
    return C_new


def update_C_GaussSeidel(X, D, C, d, k, lamda):
    C_new = C.copy()
    Xh = D@C_new
    for j in range(1, k+1):
        gC = (-D[:, (j-1)*d:j*d]).T @ (X - Xh)
        tau = 1.0*np.linalg.norm(D[:, (j-1)*d:j*d], 2)**2
        temp = C[(j-1)*d:j*d, :] - gC/tau
        C_new[(j-1)*d:j*d, :] = solve_L21(temp, lamda/tau)
        Xh = Xh+D[:, (j-1)*d:j*d]@(C_new[(j-1)*d:j*d, :]-C[(j-1)*d:j*d, :])

    return C_new


def update_C_GaussSeidel_extrop(X, D, C, d, k, lamda, dfC, Q, i):
    C_h = np.zeros(np.shape(C))
    for j in range(1, k+1):
        if i > 2:
            eta = np.sqrt(Q[i-3, j-1]/Q[i-2, j-1]) * 0.95
        else:
            eta = 0
        C_h[(j-1)*d:j*d, :] = C[(j-1)*d:j*d, :] - eta * dfC[(j-1)*d:j*d, :]

    C_new = C_h.copy()
    Xh = D@C_new

    for j in range(1, k+1):
        gC = (-D[:, (j-1)*d:j*d]).T @ (X - Xh)
        tau = 1.0*np.linalg.norm(D[:, (j-1)*d:j*d], 2)**2
        Q[i-1, j-1] = tau
        temp = C_h[(j-1)*d:j*d, :] - gC/tau
        C_new[(j-1)*d:j*d, :] = solve_L21(temp, lamda/tau)
        Xh = Xh+D[:, (j-1)*d:j*d]@(C_new[(j-1)*d:j*d, :]-C_h[(j-1)*d:j*d, :])

    return C_new, Q


def update_D(X, D, C_new, iter_D):
    """
    Update D by  gradient descent.
    input:  X: data matrix, m x n
            D: dictionary, m x d*k
            C_new: coefficient matrix, d*k x n
            iter_D: number of iterations for updating D
    """
    m = D.shape[0]
    D_t = D.copy()
    XC = X@C_new.T
    CC = C_new@C_new.T
    tau = 1.0*np.linalg.norm(CC, 2)
    for _ in range(1, iter_D+1):
        gD = -XC + D_t@CC
        D_t = D_t - gD/tau
        ld = np.float_power(np.sum(D_t**2, axis=0), 0.5)
        idx = np.where(ld > 1)[0]
        D_t[:, idx] = np.divide(D_t[:, idx], np.matlib.repmat(ld[idx], m, 1))
        # D_t[:, idx] = preprocessing.normalize(D_t[:, idx], norm='l2')
        if np.linalg.norm(gD/tau, 'fro')/np.linalg.norm(D_t, 'fro') < 1e-3:
            break
    D_new = D_t
    return D_new


def solve_L21(X, threshold):
    L = np.float_power(np.sum(X**2, axis=0), 0.5)
    Lc = np.divide(np.maximum(L - threshold, 0), L)
    X = np.multiply(X, np.matlib.repmat(Lc, X.shape[0], 1))
    X[:, np.where(L == 0)] = 0
    return X
