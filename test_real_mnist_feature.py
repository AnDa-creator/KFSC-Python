"""
Test KFSC on real-world data
"""
import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from BestMap import BestMap
from KFSC import KFSC
from KFSC_LARGE import KFSC_LARGE

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
# X = np.divide(X, np.matlib.repmat(np.sqrt(np.sum(X**2, axis=0)), X.shape[0], 1))
# Perform KFSC
tic = time.time()
opt = {
    'solver': 2,
    'maxiter': 300,
    'tol': 1e-4,
    'init_type': 'k-means-cos',
    'nrep_kmeans': 10,
    'classifier': 're'
}
lamda = 0.5
d = 30
L_kFSC, OUT = KFSC(X, k, d, lamda, opt)
L_kFSC = BestMap(label[:], L_kFSC[:])
acc_kFSC = accuracy_score(label, L_kFSC)
nmi_kFSC = normalized_mutual_info_score(label, L_kFSC)
print('kFSC: acc = %.4f, nmi = %.4f' % (acc_kFSC, nmi_kFSC))
toc = time.time()
dt = toc - tic
print("Time elapsed: ", dt, " seconds")
run_info = pd.DataFrame({'Dataset': [dataset] ,'acc': [acc_kFSC],
                            'nmi': [nmi_kFSC], 'time': [dt], 'Function': 'KFSC'})
run_info.to_csv('run_info.csv', mode='a', index=False, header=False)
# Perform KFSC Large
tic = time.time()
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
L_kFSC_LARGE, OUT = KFSC_LARGE(X, k, d, lamda, opt, 500, 'k-means-cos')
L_kFSC_LARGE = BestMap(label[:], L_kFSC_LARGE[:])
acc_kFSC_LARGE = accuracy_score(label, L_kFSC_LARGE)
nmi_kFSC_LARGE = normalized_mutual_info_score(label, L_kFSC_LARGE)
print('kFSC_LARGE: acc = %.4f, nmi = %.4f' % (acc_kFSC_LARGE, nmi_kFSC_LARGE))
toc = time.time()
dt = toc - tic
print("Time elapsed: ", dt, " seconds")
run_info = pd.DataFrame({'Dataset': [dataset] ,'acc': [acc_kFSC_LARGE], 
                         'nmi': [nmi_kFSC_LARGE], 'time': [dt], 'Function': 'KFSC_LARGE'})
run_info.to_csv('run_info.csv', mode='a', index=False, header=False)
