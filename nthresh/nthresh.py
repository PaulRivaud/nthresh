import itertools
import numpy as np
from scipy import stats
from multiprocessing import Pool

def return_idx(X, T, n):
    I = []
    I.append(np.where(X<=T[0])[0])
    if len(T)>=2:
        for i in range(len(T)-1):
            I.append(np.where((X>T[i]) & (X<=T[i+1]))[0])
    I.append(np.where(X>T[-1])[0])
    return I

def intraclassvar(X, T, n, N):
    I = return_idx(X, T, n)
    V = []
    W = []
    for i in range(n+1):
        idx = I[i]
        if len(idx) != 0:
            var_ = np.var(X[idx])
            W.append(len(idx)/N)
            V.append(var_)
    intraclassvar = np.sum([v*w for v,w in zip(V,W)])
    return intraclassvar

def nthresh(X, n_classes=2, bins=10, n_jobs=None):
    '''
    X : ndarray
        1-dimensional Numpy array
    n_classes : int
        number of expected classes. n_classes-1 threshold values will be returned in a list
    bins : int
        Number of bins to use when binning the space of X
    n_jobs : int
        Number of cores to use. If None, all possible cores will be used
    '''
    n = n_classes-1
    V = np.arange(X.min(), X.max(), (X.max()-X.min())/bins)[:-1]
    Ts = list(itertools.combinations(V, n))
    N = len(X)
    
    with Pool(n_jobs) as p:
        q = p.starmap(intraclassvar, [(X,T,n,N) for T in Ts])
    return Ts[np.argmin(q)]