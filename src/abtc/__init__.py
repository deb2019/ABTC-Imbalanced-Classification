"""
ABTC - Adaptive Bayesian Thresholding Classifier
=================================================
Core library for metric- or cost-optimal threshold selection with 
isotonic calibration and Bayesian posterior adjustment.

Version: 1.2.0

Search Configuration:
- Path A (Metric-Driven): 27α × 20τ = 540 evaluations
- Path B (Cost-Sensitive): 27α = 27 evaluations (τ is analytical)

Modules
-------
calibration : Probability calibration (isotonic regression)
prior_adjust : Posterior adjustment with α parameter
thresholding : Single-layer grid search for (α*, τ*)
metrics : Evaluation metrics (F1, F2, G-Mean, MCC, etc.)

Quick Start
-----------
>>> from abtc import alpha_adjust_posterior, joint_alpha_tau_search, fbeta
>>> 
>>> # Path A: Metric-driven optimization
>>> alpha, tau, score, n_eval = joint_alpha_tau_search(
...     y_val, p_calibrated, pi=0.1,
...     adjust_fn=alpha_adjust_posterior,
...     metric_fn=lambda yt, yp: fbeta(yt, yp, beta=2.0)
... )
"""

__version__ = "1.2.0"
__author__ = "Debashis Roy, Anandarup Roy, Bibhuprasad Sahu, Utpal Roy"

# Calibration
from .calibration import isotonic_calibrator

# Prior Adjustment (UPDATED for single-layer)
from .prior_adjust import (
    odds_correction,
    prior_adjust_posteriors,
    alpha_adjust_posterior,      # NEW: Core ABTC adjustment
    compute_adjustment_strength,  # Diagnostic utility
    effective_threshold           # (alpha, tau) -> threshold on the calibrated probability
)

# Thresholding (UPDATED for single-layer)
from .thresholding import (
    tau_cost,
    sweep_metric_threshold,
    apply_threshold,
    operating_point,
    joint_alpha_tau_search,       # NEW: Path A search
    alpha_search_analytical_tau,  # NEW: Path B search
    get_search_config,           # Configuration utility
    joint_alpha_tau_search_fast,  # Vectorised Path A (same candidates)
    metric_from_counts
)

# Metrics
from .metrics import (
    fbeta,
    gmean,
    mcc,
    balanced_acc,
    pr_auc
)

__all__ = [
    # Version
    "__version__",
    
    # Calibration
    "isotonic_calibrator",
    
    # Prior Adjustment
    "odds_correction",
    "prior_adjust_posteriors",
    "alpha_adjust_posterior",
    "compute_adjustment_strength",
    "effective_threshold",
    
    # Thresholding
    "tau_cost",
    "sweep_metric_threshold",
    "apply_threshold",
    "operating_point",
    "joint_alpha_tau_search",
    "alpha_search_analytical_tau",
    "get_search_config",
    "joint_alpha_tau_search_fast",
    "metric_from_counts",
    
    # Metrics
    "fbeta",
    "gmean",
    "mcc",
    "balanced_acc",
    "pr_auc"
]
