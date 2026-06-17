import numpy as np, matplotlib.pyplot as plt
def reliability_diagram(y_true, y_prob, n_bins=10, show=True, ax=None):
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0,1.0,n_bins+1); idx = np.digitize(y_prob, bins)-1
    bin_acc, bin_conf = [], []
    for b in range(n_bins):
        m = idx==b
        if m.sum()==0: continue
        bin_acc.append(y_true[m].mean()); bin_conf.append(y_prob[m].mean())
    if ax is None: fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], linestyle="--"); ax.scatter(bin_conf, bin_acc)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); ax.set_title("Reliability diagram")
    if show: plt.show()
    return ax
