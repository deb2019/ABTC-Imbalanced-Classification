"""
ABTC Thresholding Module
========================
Single-layer grid search for optimal (α, τ) parameter selection.

Search Configuration:
- Path A (Metric-Driven): Joint (α, τ) grid search, 27×20 = 540 evaluations
- Path B (Cost-Sensitive): α search with analytical τ*, 27 evaluations

Every (α, τ) pair is equivalent to one threshold on the calibrated probability
(see prior_adjust.effective_threshold), so Path A is a search over 540 candidate
thresholds. ``joint_alpha_tau_search_fast`` evaluates the same 540 candidates with
sorted scores and binary search instead of one confusion matrix per candidate.
"""

from typing import Callable, Tuple, Optional
import numpy as np


# =============================================================================
# CORE THRESHOLD FUNCTIONS
# =============================================================================

def tau_cost(cfn: float, cfp: float, pi: float = None) -> float:
    """
    Compute analytical optimal threshold for cost-sensitive classification.
    
    For Path B (Cost-Sensitive), τ* is computed analytically:
    
        τ* = C_FP / (C_FP + C_FN)
    
    Parameters
    ----------
    cfn : float
        Cost of false negative (C_FN)
    cfp : float
        Cost of false positive (C_FP)
    pi : float, optional
        Prior probability (not used in simplified formula, kept for API compatibility)
        
    Returns
    -------
    tau : float
        Optimal cost-based threshold
        
    Examples
    --------
    >>> tau_cost(cfn=10.0, cfp=1.0)  # FN costs 10x more than FP
    0.0909...
    >>> tau_cost(cfn=1.0, cfp=1.0)   # Equal costs
    0.5
    """
    assert cfn > 0 and cfp > 0, "Costs must be positive"
    return cfp / (cfp + cfn)


def apply_threshold(p: np.ndarray, tau: float) -> np.ndarray:
    """
    Apply decision threshold to probabilities.
    
    Parameters
    ----------
    p : array-like
        Predicted probabilities
    tau : float
        Decision threshold
        
    Returns
    -------
    y_pred : ndarray
        Binary predictions (1 if p >= tau, else 0)
    """
    return (np.asarray(p) >= tau).astype(int)


def operating_point(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute confusion matrix operating point statistics.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
        
    Returns
    -------
    stats : dict
        Dictionary with tp, fp, tn, fn, tpr, tnr
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, tpr=tpr, tnr=tnr)


# =============================================================================
# SINGLE-LAYER SEARCH FUNCTIONS
# =============================================================================

def sweep_metric_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable,
    grid: int | None = 512
) -> Tuple[float, float]:
    """
    Sweep over τ values to find optimal threshold for a given metric.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Predicted probabilities
    metric : callable
        Function(y_true, y_pred) -> float (higher is better)
    grid : int or None
        Number of threshold values to try (None uses unique scores)
        
    Returns
    -------
    best_tau : float
        Optimal threshold
    best_val : float
        Best metric value achieved
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    
    candidates = np.unique(y_score) if grid is None else np.linspace(0.0, 1.0, int(grid))
    best_tau, best_val = 0.5, -np.inf
    
    for tau in candidates:
        y_pred = (y_score >= tau).astype(int)
        val = float(metric(y_true, y_pred))
        if val > best_val:
            best_val, best_tau = val, float(tau)
    
    return best_tau, best_val


def joint_alpha_tau_search(
    y_val: np.ndarray,
    p_raw: np.ndarray,
    pi: float,
    adjust_fn: Callable,
    metric_fn: Callable,
    n_alpha: int = 27,
    n_tau: int = 20,
    alpha_range: Tuple[float, float] = (0.1, 3.0),
    tau_range: Tuple[float, float] = (0.01, 0.99)
) -> Tuple[float, float, float, int]:
    """
    PATH A: Single-layer joint (α, τ) grid search.
    
    This is the core ABTC metric-driven optimization:
    
        (α*, τ*) = argmax_{α,τ} Metric(y_val, I[p_adj(α) ≥ τ])
    
    Parameters
    ----------
    y_val : array-like
        Validation labels
    p_raw : array-like
        Raw calibrated probabilities
    pi : float
        Minority class prior from training
    adjust_fn : callable
        Posterior adjustment function: adjust_fn(p, alpha, pi) -> p_adj
    metric_fn : callable
        Optimization metric: metric_fn(y_true, y_pred) -> float (higher is better)
    n_alpha : int, default=27
        Number of α values in search grid
    n_tau : int, default=20
        Number of τ values in search grid
    alpha_range : tuple, default=(0.1, 3.0)
        Range for α parameter
    tau_range : tuple, default=(0.01, 0.99)
        Range for τ parameter
        
    Returns
    -------
    best_alpha : float
        Optimal α*
    best_tau : float
        Optimal τ*
    best_score : float
        Best metric value achieved
    n_evaluations : int
        Total number of evaluations (should be n_alpha × n_tau)
        
    Notes
    -----
    Grid Configuration (default):
    - α: 27 values in [0.1, 3.0], Δα ≈ 0.11
    - τ: 20 values in [0.01, 0.99], Δτ ≈ 0.05
    - Total: 27 × 20 = 540 evaluations
    
    Examples
    --------
    >>> from abtc.prior_adjust import alpha_adjust_posterior
    >>> from abtc.metrics import fbeta
    >>> alpha, tau, score, n_eval = joint_alpha_tau_search(
    ...     y_val, p_raw, pi=0.1,
    ...     adjust_fn=alpha_adjust_posterior,
    ...     metric_fn=lambda yt, yp: fbeta(yt, yp, beta=2.0)
    ... )
    >>> print(f"α*={alpha:.3f}, τ*={tau:.3f}, F2={score:.3f}, evals={n_eval}")
    """
    y_val = np.asarray(y_val).astype(int)
    p_raw = np.asarray(p_raw, dtype=float)
    
    alpha_list = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    tau_list = np.linspace(tau_range[0], tau_range[1], n_tau)
    
    best_alpha, best_tau, best_score = None, None, -np.inf
    n_evaluations = 0
    
    for alpha in alpha_list:
        # Apply posterior adjustment with current α
        p_adj = adjust_fn(p_raw, alpha, pi)
        
        for tau in tau_list:
            y_pred = (p_adj >= tau).astype(int)
            score = float(metric_fn(y_val, y_pred))
            n_evaluations += 1
            
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
                best_tau = float(tau)
    
    return best_alpha, best_tau, best_score, n_evaluations


def alpha_search_analytical_tau(
    y_val: np.ndarray,
    p_raw: np.ndarray,
    pi: float,
    adjust_fn: Callable,
    cost_fn: float,
    cost_fp: float,
    metric_fn: Callable,
    n_alpha: int = 27,
    alpha_range: Tuple[float, float] = (0.1, 3.0)
) -> Tuple[float, float, float, int]:
    """
    PATH B: α search with analytical τ* from cost ratio.
    
    For cost-sensitive optimization:
    1. τ* is calculated analytically: τ* = C_FP / (C_FP + C_FN)
    2. Only α is optimized via grid search
    
        α* = argmin_α Cost(y_val, I[p_adj(α) ≥ τ*])
    
    Parameters
    ----------
    y_val : array-like
        Validation labels
    p_raw : array-like
        Raw calibrated probabilities
    pi : float
        Minority class prior from training
    adjust_fn : callable
        Posterior adjustment function: adjust_fn(p, alpha, pi) -> p_adj
    cost_fn : float
        Cost of false negative (C_FN)
    cost_fp : float
        Cost of false positive (C_FP)
    metric_fn : callable
        Cost metric: metric_fn(y_true, y_pred) -> float
        Note: Should return NEGATIVE cost so higher is better
    n_alpha : int, default=27
        Number of α values in search grid
    alpha_range : tuple, default=(0.1, 3.0)
        Range for α parameter
        
    Returns
    -------
    best_alpha : float
        Optimal α*
    tau_star : float
        Analytical τ* (from cost ratio)
    best_score : float
        Best (negative) cost achieved
    n_evaluations : int
        Total number of evaluations (should be n_alpha)
        
    Notes
    -----
    Grid Configuration (default):
    - α: 27 values in [0.1, 3.0], Δα ≈ 0.11
    - τ: Analytical (single value)
    - Total: 27 evaluations
    
    Examples
    --------
    >>> alpha, tau, score, n_eval = alpha_search_analytical_tau(
    ...     y_val, p_raw, pi=0.1,
    ...     adjust_fn=alpha_adjust_posterior,
    ...     cost_fn=10.0, cost_fp=1.0,
    ...     metric_fn=lambda yt, yp: -cost_metric(yt, yp, 1.0, 10.0)
    ... )
    >>> print(f"α*={alpha:.3f}, τ*={tau:.3f}, cost={-score:.1f}, evals={n_eval}")
    """
    y_val = np.asarray(y_val).astype(int)
    p_raw = np.asarray(p_raw, dtype=float)
    
    # Calculate analytical τ* from cost ratio
    tau_star = tau_cost(cost_fn, cost_fp)
    
    alpha_list = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    best_alpha, best_score = None, -np.inf
    n_evaluations = 0
    
    for alpha in alpha_list:
        # Apply posterior adjustment with current α
        p_adj = adjust_fn(p_raw, alpha, pi)
        y_pred = (p_adj >= tau_star).astype(int)
        score = float(metric_fn(y_val, y_pred))
        n_evaluations += 1
        
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    
    return best_alpha, tau_star, best_score, n_evaluations


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_search_config(
    path: str = 'A',
    n_alpha: int = 27,
    n_tau: int = 20,
    alpha_range: Tuple[float, float] = (0.1, 3.0),
    tau_range: Tuple[float, float] = (0.01, 0.99)
) -> dict:
    """
    Get search configuration details.
    
    Parameters
    ----------
    path : str
        'A' for metric-driven, 'B' for cost-sensitive
    n_alpha : int
        Number of α values
    n_tau : int
        Number of τ values (only for Path A)
    alpha_range : tuple
        α search range
    tau_range : tuple
        τ search range (only for Path A)
        
    Returns
    -------
    config : dict
        Search configuration details
    """
    delta_alpha = (alpha_range[1] - alpha_range[0]) / (n_alpha - 1) if n_alpha > 1 else 0
    
    if path.upper() == 'A':
        delta_tau = (tau_range[1] - tau_range[0]) / (n_tau - 1) if n_tau > 1 else 0
        return {
            'path': 'A (Metric-Driven)',
            'search_type': 'Joint (α, τ) grid search',
            'n_alpha': n_alpha,
            'n_tau': n_tau,
            'alpha_range': alpha_range,
            'tau_range': tau_range,
            'delta_alpha': delta_alpha,
            'delta_tau': delta_tau,
            'total_evaluations': n_alpha * n_tau
        }
    else:
        return {
            'path': 'B (Cost-Sensitive)',
            'search_type': 'α search with analytical τ*',
            'n_alpha': n_alpha,
            'alpha_range': alpha_range,
            'tau_formula': 'τ* = C_FP / (C_FP + C_FN)',
            'delta_alpha': delta_alpha,
            'total_evaluations': n_alpha
        }


# =============================================================================
# FAST (VECTORISED) PATH A SEARCH
# =============================================================================

def _counts_at(p: np.ndarray, y: np.ndarray, thr: np.ndarray):
    """TP, FP, FN, TN for predictions (p >= t), for every t in thr."""
    pos = np.sort(p[y == 1]); neg = np.sort(p[y == 0]); thr = np.atleast_1d(thr)
    TP = len(pos) - np.searchsorted(pos, thr, side="left")
    FP = len(neg) - np.searchsorted(neg, thr, side="left")
    return (TP.astype(float), FP.astype(float),
            (len(pos) - TP).astype(float), (len(neg) - FP).astype(float))


def metric_from_counts(TP, FP, FN, TN, metric: str, cost_fp: float = 1.0, cost_fn: float = 10.0):
    """Vectorised metrics from confusion counts (higher is better; cost is returned negated)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
        rec = np.where(TP + FN > 0, TP / (TP + FN), 0.0)
        spec = np.where(TN + FP > 0, TN / (TN + FP), 0.0)
        if metric == "F1":
            d = 2 * TP + FP + FN
            return np.where(d > 0, 2 * TP / np.where(d > 0, d, 1), 0.0)
        if metric == "F2":
            d = 4 * prec + rec
            return np.where(prec + rec > 0, 5 * prec * rec / np.where(d > 0, d, 1), 0.0)
        if metric == "G-Mean":
            return np.sqrt(rec * spec)
        if metric == "MCC":
            d = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            return np.where(d > 0, (TP * TN - FP * FN) / np.where(d > 0, d, 1), 0.0)
        if metric == "BalancedAcc":
            return 0.5 * (rec + spec)
        if metric == "cost":
            return -(cost_fp * FP + cost_fn * FN)
    raise ValueError(f"unknown metric {metric!r}")


def joint_alpha_tau_search_fast(
    y_val: np.ndarray,
    p_raw: np.ndarray,
    pi: float,
    metric: str = "F2",
    n_alpha: int = 27,
    n_tau: int = 20,
    alpha_range: Tuple[float, float] = (0.1, 3.0),
    tau_range: Tuple[float, float] = (0.01, 0.99),
) -> Tuple[float, float, float, int]:
    """
    Path A with the same 540 candidates and the same order (α outer, τ inner, first
    maximum kept) as :func:`joint_alpha_tau_search`, computed in O(A·n log n + A·T log n).
    ``metric`` is one of 'F1', 'F2', 'G-Mean', 'MCC', 'BalancedAcc'.
    When two candidates tie on the validation set, floating-point rounding can make the
    two implementations keep different (but equally scored) candidates.
    """
    from .prior_adjust import alpha_adjust_posterior
    y_val = np.asarray(y_val).astype(int)
    p_raw = np.asarray(p_raw, dtype=float)
    alpha_list = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    tau_list = np.linspace(tau_range[0], tau_range[1], n_tau)
    scores = np.vstack([metric_from_counts(*_counts_at(alpha_adjust_posterior(p_raw, a, pi), y_val, tau_list), metric)
                        for a in alpha_list])
    k = int(np.argmax(scores.ravel()))
    return float(alpha_list[k // n_tau]), float(tau_list[k % n_tau]), float(scores.ravel()[k]), n_alpha * n_tau
