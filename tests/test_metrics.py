import numpy as np
from abtc.metrics import fbeta, gmean, mcc, balanced_acc

def test_fbeta_edges():
    y = np.array([0,1,0,1]); yhat = np.array([0,0,0,0])
    assert fbeta(y,yhat,2.0) >= 0.0

def test_gmean_bounds():
    y = np.array([0,1,0,1]); yhat = np.array([0,1,0,1])
    g = gmean(y,yhat); assert 0.0 <= g <= 1.0
