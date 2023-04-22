"""
Test KFSC on real-world data Batch multiprocess
"""
import os
import sys
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
from KFSC_minibatch import KFSC_MB

def test_dataset(X, k, label, function, dataset, testparam={}):
    """
    Test KFSC on a dataset
    input:
        X: data matrix, each column is a data point
        k: number of clusters
        label: ground truth labels
        function: KFSC, KFSC_LARGE, KFSC_MB
        dataset: name of the dataset
        testparam: parameters for KFSC, KFSC_LARGE, KFSC_MB
    """
    tic = time.time()
    print("Starting {} at process id {}".format(function.__name__, os.getpid()))
    sys.stdout.flush()
    if "opt" in testparam.keys():
        opt = testparam["opt"]
    else:
        opt = {
            'solver': 2,
            'maxiter': 300,
            'tol': 1e-4,
            'init_type': 'k-means-cos',
            'nrep_kmeans': 1,
            'classifier': 're'
        }
    if "lamda" in testparam.keys():
        lamda = testparam["lamda"]
    else:   
        lamda = 0.5
    if "d" in testparam.keys():
        d = testparam["d"]
    else:
        d = 30
    if function.__name__ == 'KFSC':
        func_result, OUT = KFSC(X, k, d, lamda, opt)
    elif function.__name__ == 'KFSC_LARGE':
        if "n_sel" in testparam.keys():
            n_sel = testparam["n_sel"]
        else:
            n_sel = 500
        if "sel_type" in testparam.keys():
            sel_type = testparam["sel_type"]
        else:
            sel_type = 'k-means-cos'
        func_result, OUT = KFSC_LARGE(X, k, d, lamda, opt, n_sel, sel_type)
    elif function.__name__ == 'KFSC_MB':
        func_result, OUT = KFSC_MB(X, k, d, lamda, opt)
    else:
        print("Function not found")
        return
    
    func_result = BestMap(label[:], func_result[:])
    func_acc = accuracy_score(label, func_result)
    func_nmi = normalized_mutual_info_score(label, func_result)
    print('{}: acc = {}, nmi = {}, params: {}'.format(function.__name__,func_acc, func_nmi, 
                                                      testparam))
    toc = time.time()
    dt = toc - tic
    print("Time elapsed: ", dt, " seconds")
    run_info = pd.DataFrame({'Dataset': [dataset] ,'acc': [func_acc], 
                            'nmi': [func_nmi], 'time': [dt], 'd': [d], 'lamda': [lamda],
                            'opt': [opt],'Function': function.__name__, 
                            'parameters': [testparam]})
    return run_info
    # run_info.to_csv('Scores//fmnist_run_info.csv', mode='a', index=False, header=False)

if __name__ == '__main__':
    # to get the current working directory
    path = os.getcwd()
    os.chdir(path)
    dataset = 'mnist_sc_f150.mat'
    # dataset = 'fmnist_fea_150.mat'
    # dataset = 'Epileptic.mat'
    f = sio.loadmat(dataset)
    X = f['X']
    label = np.concatenate(f['Label'])
    k = len(np.unique(label))
    cpu_count = multiprocessing.cpu_count()
    use_cpu = 50
    print("CPU count: ", cpu_count)
    print("Using CPU: ", use_cpu)
    iterations = use_cpu

    funct_to_run = KFSC_LARGE
    with Pool(processes=use_cpu) as p:
        df_list = p.starmap(test_dataset, [(X, k, label, funct_to_run, dataset) 
        for i in range(iterations)])
    combined_df = pd.concat(df_list, ignore_index=True)
    path = os.getcwd()
    os.chdir(path+'//Scores')
    combined_df.to_csv(dataset.split('_')[0] + 'run_info.csv', mode='a', index=False, header=False)
    