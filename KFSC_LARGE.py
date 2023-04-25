"""
    Perform kFSC on a large dataset by selecting landmark data points randomly or by k-means
"""
import numpy as np
from sklearn import preprocessing  # to normalise existing X
from sklearn.cluster import KMeans
from KFSC import KFSC
from scipy import sparse
from soyclustering import SphericalKMeans

def KFSC_LARGE(X, k, d, lamda, opt, n_sel, sel_type, use_numba=False):
    """
    Perform kFSC on a large dataset by selecting landmark data points randomly or by k-means
        input:
            X: data matrix, each column is a data point 
            d: subspace dimension
            k: number of clusters
            lamda: regularization parameter
            opt: options
            n_sel: number of landmark data points selected
            sel_type: 'random' or 'k-means'
        output:
            L: labels of all data points
            output: output of KFSC
    """
    n = X.shape[1]
    if sel_type == 'random':
        print("Select landmark data points randomly...")
        my_generator = np.random.default_rng()
        ids = np.sort(my_generator.choice(n, n_sel*k, replace=False), axis=0)
        Xs=X.copy()
        X_train=Xs[:,ids]
    elif sel_type == 'k-means' or sel_type == 'k-means-cos':
        print("Select landmark data points by k-means...")
        my_generator = np.random.default_rng()
        ids = np.sort(my_generator.choice(
            n, min(n, n_sel*k*5), replace=False), axis=0)
        Xs = X[:, ids]
        X_Norm = preprocessing.normalize(Xs.T)
        if sel_type == 'k-means':
            kmeans_x = KMeans(n_clusters=n_sel*k, n_init=1).fit(X_Norm)
            C = kmeans_x.cluster_centers_
            dist = kmeans_x.transform(X_Norm)**2
        elif sel_type == 'k-means-cos':
            sX_norm = sparse.csr_matrix(X_Norm)
            kmeans_x = SphericalKMeans(n_clusters=n_sel*k, verbose=0).fit(sX_norm)
            C = kmeans_x.cluster_centers_
            dist = kmeans_x.transform(sX_norm)**2

        idx = np.argsort(dist, axis=0)
        ids = idx[1, :]
        X_train = C.T

        del Xs, C, dist
    print('Perform kFSC on the selected landmark data points...')
    L, output = KFSC(X_train, k, d, lamda, opt, use_numba=use_numba)
    print('Predict the labels of all data points...')
    D = output['D']
    E = np.zeros((k, n))
    for i in range(1, k+1):
        Di = D[:, (i-1)*d:i*d]
        C = np.linalg.inv(Di.T@Di + 1e-5 * np.eye(d))@Di.T@X
        E[i-1, :] = np.sum((X-Di@C)**2, axis=0)

    Label = np.argmin(E, axis=0)
    return Label, output
