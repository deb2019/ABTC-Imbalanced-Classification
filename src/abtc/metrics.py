import numpy as np
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, average_precision_score

def fbeta(y_true, y_pred, beta=1.0):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true==1)&(y_pred==1))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    if tp==0 and (fp>0 or fn>0): return 0.0
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    if precision==0 and recall==0: return 0.0
    b2 = beta*beta
    return (1+b2)*(precision*recall)/(b2*precision+recall) if (b2*precision+recall)>0 else 0.0

def gmean(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true==1)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
    tnr = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return float(np.sqrt(tpr*tnr))

def mcc(y_true, y_pred): return float(matthews_corrcoef(y_true, y_pred))
def balanced_acc(y_true, y_pred): return float(balanced_accuracy_score(y_true, y_pred))
def pr_auc(y_true, y_score): return float(average_precision_score(y_true, y_score))
