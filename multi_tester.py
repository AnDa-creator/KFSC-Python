"""
Test KFSC on real-world data Batch multiprocess
"""
import os
import time
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from BestMap import BestMap
from KFSC import KFSC
from KFSC_LARGE import KFSC_LARGE

def test_dataset(X, k, label, function):
    tic = time.time()
    print("Starting {} at process id {}".format(function.__name__, os.getpid()))
    opt = {
        'solver': 2,
        'maxiter': 300,
        'tol': 1e-4,
        'init_type': 'k-means-cos',
        'nrep_kmeans': 20,
        'classifier': 're'
    }
    lamda = 0.5
    d = 30
    func_result, OUT = function(X, k, d, lamda, opt, 500, 'k-means-cos')
    func_result = BestMap(label[:], func_result[:])
    func_acc = accuracy_score(label, func_result)
    func_nmi = normalized_mutual_info_score(label, func_result)
    print('kFSC_LARGE: acc = %.4f, nmi = %.4f' % (func_acc, func_nmi))
    toc = time.time()
    dt = toc - tic
    print("Time elapsed: ", dt, " seconds")
    run_info = pd.DataFrame({'Dataset': [dataset] ,'acc': [func_acc], 
                            'nmi': [func_nmi], 'time': [dt], 
                            'Function': function.__name__})
    run_info.to_csv('run_info.csv', mode='a', index=False, header=False)

if __name__ == '__main__':
    # to get the current working directory
    path = os.getcwd()
    os.chdir(path)
    # dataset = 'mnist_sc_f150.mat'
    dataset = 'fmnist_fea_150.mat'
    # dataset = 'Epileptic.mat'
    f = sio.loadmat(dataset)
    X = f['X']
    label = np.concatenate(f['Label'])
    k = len(np.unique(label))
    cpu_count = multiprocessing.cpu_count()
    use_cpu = cpu_count - 2
    print("CPU count: ", cpu_count)
    print("Using CPU: ", use_cpu)
    iterations = use_cpu

    funct_to_run = KFSC_LARGE
    with Pool(processes=use_cpu) as p:
        p.starmap(test_dataset, [(X, k, label, funct_to_run) for i in range(iterations)])

    