"""
ABTC Prior Adjustment Module
============================
Posterior probability adjustment with parameter α for class imbalance correction.

Version: 1.2.0
"""

import numpy as np


def odds_correction(p: np.ndarray, pi_train: float, pi_deploy: float) -> np.ndarray:
    """
    Standard odds correction for prior shift.
    
    Parameters
    ----------
    p : array-like
        Original probabilities
    pi_train : float
        Training set prior P(Y=1)
    pi_deploy : float
        Deployment prior P(Y=1)
        
    Returns
    -------
    p_adjusted : ndarray
        Adjusted probabilities
    """
    p = np.clip(p, 1e-12, 1-1e-12)
    odds = p / (1 - p)
    k = (pi_deploy / (1 - pi_deploy)) / (pi_train / (1 - pi_train))
    odds_adj = odds * k
    return odds_adj / (1 + odds_adj)


def prior_adjust_posteriors(p: np.ndarray, pi_train: float, pi_deploy: float | None = None) -> np.ndarray:
    """
    Adjust posteriors for prior shift between training and deployment.
    
    Parameters
    ----------
    p : array-like
        Original probabilities
    pi_train : float
        Training set prior
    pi_deploy : float, optional
        Deployment prior (if None, no adjustment)
        
    Returns
    -------
    p_adjusted : ndarray
        Adjusted probabilities
    """
    if pi_deploy is None or abs(pi_deploy - pi_train) < 1e-12:
        return np.clip(p, 1e-12, 1-1e-12)
    return odds_correction(p, pi_train, pi_deploy)


def alpha_adjust_posterior(p: np.ndarray, alpha: float, pi: float) -> np.ndarray:
    """
    Apply the ABTC posterior adjustment with parameter α.

        p_adj = (p / π^α) / (p / π^α + (1-p) / (1-π)^α)

    Equivalently, in log-odds form,

        logit(p_adj) = logit(p) + α·L,   L = log((1-π)/π).

    Parameters
    ----------
    p : array-like
        Calibrated probabilities P(Y=1|x).
    alpha : float
        Adjustment exponent α ≥ 0 (the paper's grid uses [0.1, 3.0]).
        - α = 0: no adjustment (p_adj = p).
        - α = 1: if p is the true posterior of the training population, p_adj is the
          posterior under equal class priors (same class-conditional distributions).
        - For a minority positive class (π < 0.5, so L > 0), a larger α raises p_adj
          and therefore predicts the positive class more often.
        - If π = 0.5, α has no effect.
    pi : float
        Positive-class proportion π = P(Y=1) estimated from the training data.

    Returns
    -------
    p_adj : ndarray
        Adjusted probabilities.

    Notes
    -----
    Thresholding p_adj at τ is the same decision as thresholding p at the effective
    threshold t(α, τ) = σ(logit τ − α·L); see :func:`effective_threshold`.
    Interpretation (Supplementary Proposition S2): if p is the true posterior P(Y=1|x) of
    the training population and the deployment population differs only in its class prior
    (class-conditional distributions unchanged), then applying τ* = C_FP / (C_FP + C_FN)
    to p_adj is the Bayes rule for a deployment prior π' with logit π' = (1 − α)·logit π.
    Isotonic calibration does not guarantee these assumptions; with a calibrated score the
    rule is optimal only among rules that use that score.
    The 1e-12 guards reproduce the implementation used in the paper's experiments.

    Examples
    --------
    >>> p = np.array([0.3, 0.5, 0.7])
    >>> np.round(alpha_adjust_posterior(p, alpha=1.5, pi=0.2), 3)
    array([0.774, 0.889, 0.949])
    >>> np.round(alpha_adjust_posterior(p, alpha=0.0, pi=0.2), 3)
    array([0.3, 0.5, 0.7])
    """
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)

    numerator = p / (pi**alpha + eps)
    denominator = numerator + ((1 - p) / ((1 - pi)**alpha + eps))

    return numerator / (denominator + eps)


def effective_threshold(alpha: float, tau: float, pi: float) -> float:
    """
    Threshold on the unadjusted probability that gives the same decisions as
    ``alpha_adjust_posterior(p, alpha, pi) >= tau``:

        t(α, τ) = σ(logit τ − α·log((1-π)/π))

    Every (α, τ) pair of ABTC is therefore one threshold on the calibrated
    probability (Supplementary Proposition S1).
    """
    tau = np.clip(np.asarray(tau, dtype=float), 1e-15, 1 - 1e-15)
    z = np.log(tau / (1 - tau)) - np.asarray(alpha, dtype=float) * np.log((1 - pi) / pi)
    return 1.0 / (1.0 + np.exp(-z))


def compute_adjustment_strength(alpha: float, pi: float) -> dict:
    """
    Compute descriptive statistics about the adjustment strength.
    
    Parameters
    ----------
    alpha : float
        Adjustment parameter
    pi : float
        Minority class prior
        
    Returns
    -------
    info : dict
        Dictionary with adjustment characteristics
    """
    # Test on a standard probability range
    p_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    p_adj = alpha_adjust_posterior(p_test, alpha, pi)
    
    return {
        'alpha': alpha,
        'pi': pi,
        # For pi < 0.5 a larger alpha raises the adjusted probability of the positive class.
        'effect': 'none' if alpha == 0 or pi == 0.5 else ('raises_positive' if pi < 0.5 else 'lowers_positive'),
        'avg_shift': float(np.mean(p_adj - p_test)),
        'max_shift': float(np.max(np.abs(p_adj - p_test)))
    }
