"""
ABTC Prior Adjustment Module
============================
Posterior probability adjustment with parameter α for class imbalance correction.

Version: 1.0 (Single-Layer Search)
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
    Apply ABTC posterior adjustment with parameter α.
    
    This is the core ABTC transformation:
    
        p_adj = (p / π^α) / (p / π^α + (1-p) / (1-π)^α)
    
    Parameters
    ----------
    p : array-like
        Calibrated probabilities P(Y=1|x)
    alpha : float
        Adjustment parameter α ∈ [0.1, 3.0]
        - α < 1: Increases minority class predictions
        - α = 1: No adjustment (equivalent to standard posterior)
        - α > 1: Decreases minority class predictions
    pi : float
        Minority class prior π = P(Y=1) from training data
        
    Returns
    -------
    p_adj : ndarray
        Adjusted posterior probabilities
        
    Notes
    -----
    The adjustment is derived from Bayesian posterior recalibration:
    
    P(Y=1|x) ∝ P(x|Y=1) × P(Y=1)^α / P(x)
    
    where α controls the strength of prior influence.
    
    Examples
    --------
    >>> p = np.array([0.3, 0.5, 0.7])
    >>> pi = 0.2
    >>> alpha_adjust_posterior(p, alpha=1.5, pi=pi)
    array([0.182, 0.357, 0.583])
    """
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    
    numerator = p / (pi**alpha + eps)
    denominator = numerator + ((1 - p) / ((1 - pi)**alpha + eps))
    
    return numerator / (denominator + eps)


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
        'effect': 'boost_minority' if alpha < 1 else ('neutral' if alpha == 1 else 'suppress_minority'),
        'avg_shift': float(np.mean(p_adj - p_test)),
        'max_shift': float(np.max(np.abs(p_adj - p_test)))
    }
