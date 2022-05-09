# nthresh
Multilevel thresholding of a 1-dimensional array using Otsu's method. Implemented in Python3.

## Install
```
pip install git+https://github.com/PaulRivaud/nthresh.git
```

## Use
Generate a vector following a bimodal distribution:
```python
import numpy as np

X = np.concatenate((np.random.normal(0, 1, int(0.3 * 1000)),
                    np.random.normal(5, 1, int(0.7 * 1000))))
```

Get the optimum threshold.
```python
import nthresh

threshold = nthresh.nthresh(X, n_classes=2, bins=10, n_jobs=1)
```

Description of the parameters used:
```
X : ndarray
    A 1-dimensional Numpy array
n_classes : int
    Number of expected classes. n_classes-1 threshold values will be returned in a list
bins : int
    Number of bins to use when binning the space of X
n_jobs : int
    Number of cores to use. If None, all possible cores will be used
```
