"""
ABTC Thresholding Tests
=======================
Unit tests for single-layer grid search functions.

Test Coverage:
- tau_cost: Analytical cost-based threshold
- sweep_metric_threshold: Basic threshold sweep
- joint_alpha_tau_search: Path A (27α × 20τ)
- alpha_search_analytical_tau: Path B (27α with analytical τ*)
- alpha_adjust_posterior: Posterior adjustment
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from abtc.thresholding import (
    tau_cost,
    apply_threshold,
    operating_point,
    sweep_metric_threshold,
    joint_alpha_tau_search,
    alpha_search_analytical_tau,
    get_search_config,
    joint_alpha_tau_search_fast,
)
from abtc.prior_adjust import alpha_adjust_posterior, effective_threshold
from abtc.metrics import fbeta, gmean


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample validation data for testing."""
    np.random.seed(42)
    n = 200
    y_val = np.array([0] * 150 + [1] * 50)  # Imbalanced: 75% negative
    
    # Simulate calibrated probabilities
    p_raw = np.concatenate([
        np.random.beta(2, 5, 150),  # Negative class: lower probabilities
        np.random.beta(5, 2, 50)    # Positive class: higher probabilities
    ])
    
    pi = 0.25  # Training prior
    return y_val, p_raw, pi


@pytest.fixture
def simple_data():
    """Simple data for basic tests."""
    y_val = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_raw = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    pi = 0.5
    return y_val, p_raw, pi


# =============================================================================
# TAU_COST TESTS
# =============================================================================

class TestTauCost:
    """Tests for analytical cost-based threshold."""
    
    def test_tau_cost_equal_costs(self):
        """Equal costs should give τ* = 0.5."""
        tau = tau_cost(cfn=1.0, cfp=1.0)
        assert tau == pytest.approx(0.5, rel=1e-6)
    
    def test_tau_cost_high_fn_cost(self):
        """High FN cost should give low τ* (catch more positives)."""
        tau = tau_cost(cfn=10.0, cfp=1.0)
        expected = 1.0 / 11.0  # ≈ 0.0909
        assert tau == pytest.approx(expected, rel=1e-6)
        assert tau < 0.5
    
    def test_tau_cost_high_fp_cost(self):
        """High FP cost should give high τ* (avoid false positives)."""
        tau = tau_cost(cfn=1.0, cfp=10.0)
        expected = 10.0 / 11.0  # ≈ 0.909
        assert tau == pytest.approx(expected, rel=1e-6)
        assert tau > 0.5
    
    def test_tau_cost_extreme_ratio(self):
        """Extreme cost ratio."""
        tau = tau_cost(cfn=100.0, cfp=1.0)
        expected = 1.0 / 101.0
        assert tau == pytest.approx(expected, rel=1e-6)
    
    def test_tau_cost_invalid_costs(self):
        """Should raise assertion for non-positive costs."""
        with pytest.raises(AssertionError):
            tau_cost(cfn=0, cfp=1.0)
        with pytest.raises(AssertionError):
            tau_cost(cfn=1.0, cfp=-1.0)


# =============================================================================
# APPLY_THRESHOLD TESTS
# =============================================================================

class TestApplyThreshold:
    """Tests for threshold application."""
    
    def test_apply_threshold_basic(self):
        """Basic threshold application."""
        p = np.array([0.3, 0.5, 0.7])
        y_pred = apply_threshold(p, 0.5)
        expected = np.array([0, 1, 1])
        np.testing.assert_array_equal(y_pred, expected)
    
    def test_apply_threshold_boundary(self):
        """Values exactly at threshold should be positive."""
        p = np.array([0.49, 0.50, 0.51])
        y_pred = apply_threshold(p, 0.5)
        expected = np.array([0, 1, 1])
        np.testing.assert_array_equal(y_pred, expected)
    
    def test_apply_threshold_extreme(self):
        """Extreme thresholds."""
        p = np.array([0.1, 0.5, 0.9])
        
        y_low = apply_threshold(p, 0.0)
        np.testing.assert_array_equal(y_low, np.array([1, 1, 1]))
        
        y_high = apply_threshold(p, 1.0)
        np.testing.assert_array_equal(y_high, np.array([0, 0, 0]))


# =============================================================================
# OPERATING_POINT TESTS
# =============================================================================

class TestOperatingPoint:
    """Tests for confusion matrix statistics."""
    
    def test_operating_point_perfect(self):
        """Perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        stats = operating_point(y_true, y_pred)
        
        assert stats['tp'] == 2
        assert stats['tn'] == 2
        assert stats['fp'] == 0
        assert stats['fn'] == 0
        assert stats['tpr'] == 1.0
        assert stats['tnr'] == 1.0
    
    def test_operating_point_all_wrong(self):
        """All wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        stats = operating_point(y_true, y_pred)
        
        assert stats['tp'] == 0
        assert stats['tn'] == 0
        assert stats['fp'] == 2
        assert stats['fn'] == 2
        assert stats['tpr'] == 0.0
        assert stats['tnr'] == 0.0


# =============================================================================
# SWEEP_METRIC_THRESHOLD TESTS
# =============================================================================

class TestSweepMetricThreshold:
    """Tests for basic threshold sweep."""
    
    def test_sweep_returns_valid_tau(self, simple_data):
        """Should return τ in valid range."""
        y_val, p_raw, _ = simple_data
        metric = lambda yt, yp: fbeta(yt, yp, 1.0)
        
        tau, score = sweep_metric_threshold(y_val, p_raw, metric, grid=100)
        
        assert 0.0 <= tau <= 1.0
        assert 0.0 <= score <= 1.0
    
    def test_sweep_improves_metric(self, simple_data):
        """Optimized τ should improve over default 0.5."""
        y_val, p_raw, _ = simple_data
        metric = lambda yt, yp: fbeta(yt, yp, 2.0)
        
        tau_opt, score_opt = sweep_metric_threshold(y_val, p_raw, metric, grid=100)
        
        # Score at default threshold
        y_default = (p_raw >= 0.5).astype(int)
        score_default = metric(y_val, y_default)
        
        assert score_opt >= score_default


# =============================================================================
# ALPHA_ADJUST_POSTERIOR TESTS
# =============================================================================

class TestAlphaAdjustPosterior:
    """Tests for posterior adjustment with α."""
    
    def test_alpha_zero_no_change(self):
        """α = 0 leaves the probabilities unchanged."""
        p = np.array([0.2, 0.5, 0.8])
        p_adj = alpha_adjust_posterior(p, alpha=0.0, pi=0.1)
        np.testing.assert_array_almost_equal(p_adj, p, decimal=9)

    def test_balanced_prior_no_change(self):
        """With π = 0.5, α has no effect."""
        p = np.array([0.2, 0.5, 0.8])
        for alpha in [0.1, 1.0, 3.0]:
            np.testing.assert_array_almost_equal(alpha_adjust_posterior(p, alpha, 0.5), p, decimal=9)

    def test_alpha_range_preserved(self):
        """Adjusted probabilities should stay in [0, 1]."""
        np.random.seed(42)
        p = np.random.uniform(0.01, 0.99, 100)
        pi = 0.1
        
        for alpha in [0.1, 0.5, 1.0, 2.0, 3.0]:
            p_adj = alpha_adjust_posterior(p, alpha, pi)
            assert np.all(p_adj >= 0) and np.all(p_adj <= 1), f"Failed for α={alpha}"
    
    def test_higher_alpha_raises_minority(self):
        """For a minority positive class (π < 0.5), a larger α raises the adjusted probability."""
        p = np.array([0.3, 0.5, 0.7])
        pi = 0.1
        p_low = alpha_adjust_posterior(p, alpha=0.5, pi=pi)
        p_high = alpha_adjust_posterior(p, alpha=2.0, pi=pi)
        assert np.all(p_high > p_low)

    def test_effective_threshold_identity(self):
        """p_adj(α) >= τ  <=>  p >= t(α, τ)  (Supplementary Proposition S1)."""
        rng = np.random.default_rng(0)
        p = rng.uniform(0.001, 0.999, 2000)
        for pi in [0.05, 0.2, 0.4]:
            for alpha in [0.1, 1.0, 3.0]:
                for tau in [0.01, 0.3, 0.7, 0.99]:
                    t = effective_threshold(alpha, tau, pi)
                    lhs = alpha_adjust_posterior(p, alpha, pi) >= tau
                    rhs = p >= t
                    # decisions may differ only for scores within rounding distance of t
                    near = np.abs(p - t) < 1e-9
                    assert np.array_equal(lhs[~near], rhs[~near])


# =============================================================================
# JOINT_ALPHA_TAU_SEARCH TESTS (PATH A)
# =============================================================================

class TestJointAlphaTauSearch:
    """Tests for Path A: Single-layer joint (α, τ) grid search."""
    
    def test_path_a_evaluation_count(self, sample_data):
        """Should perform exactly n_alpha × n_tau evaluations."""
        y_val, p_raw, pi = sample_data
        metric_fn = lambda yt, yp: fbeta(yt, yp, 2.0)
        
        n_alpha, n_tau = 27, 20
        
        alpha, tau, score, n_eval = joint_alpha_tau_search(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            metric_fn=metric_fn,
            n_alpha=n_alpha,
            n_tau=n_tau
        )
        
        expected_evals = n_alpha * n_tau  # 540
        assert n_eval == expected_evals, f"Expected {expected_evals}, got {n_eval}"
    
    def test_path_a_default_config(self, sample_data):
        """Default configuration: 27α × 20τ = 540 evaluations."""
        y_val, p_raw, pi = sample_data
        metric_fn = lambda yt, yp: fbeta(yt, yp, 1.0)
        
        alpha, tau, score, n_eval = joint_alpha_tau_search(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            metric_fn=metric_fn
        )
        
        assert n_eval == 540, f"Default should be 540 evaluations, got {n_eval}"
    
    def test_path_a_returns_valid_params(self, sample_data):
        """Should return valid α and τ within specified ranges."""
        y_val, p_raw, pi = sample_data
        metric_fn = lambda yt, yp: fbeta(yt, yp, 2.0)
        
        alpha_range = (0.1, 3.0)
        tau_range = (0.01, 0.99)
        
        alpha, tau, score, n_eval = joint_alpha_tau_search(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            metric_fn=metric_fn,
            alpha_range=alpha_range,
            tau_range=tau_range
        )
        
        assert alpha_range[0] <= alpha <= alpha_range[1]
        assert tau_range[0] <= tau <= tau_range[1]
        assert 0.0 <= score <= 1.0
    
    def test_path_a_custom_grid(self, simple_data):
        """Custom grid sizes should work correctly."""
        y_val, p_raw, pi = simple_data
        metric_fn = lambda yt, yp: fbeta(yt, yp, 1.0)
        
        n_alpha, n_tau = 10, 5
        
        alpha, tau, score, n_eval = joint_alpha_tau_search(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            metric_fn=metric_fn,
            n_alpha=n_alpha,
            n_tau=n_tau
        )
        
        assert n_eval == n_alpha * n_tau  # 50


# =============================================================================
# ALPHA_SEARCH_ANALYTICAL_TAU TESTS (PATH B)
# =============================================================================

class TestAlphaSearchAnalyticalTau:
    """Tests for Path B: α search with analytical τ*."""
    
    def test_path_b_evaluation_count(self, sample_data):
        """Should perform exactly n_alpha evaluations."""
        y_val, p_raw, pi = sample_data
        
        def cost_metric(yt, yp):
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            return -(1.0 * fp + 10.0 * fn)  # Negative for maximization
        
        n_alpha = 27
        
        alpha, tau, score, n_eval = alpha_search_analytical_tau(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            cost_fn=10.0,
            cost_fp=1.0,
            metric_fn=cost_metric,
            n_alpha=n_alpha
        )
        
        assert n_eval == n_alpha, f"Expected {n_alpha}, got {n_eval}"
    
    def test_path_b_default_config(self, sample_data):
        """Default configuration: 27 evaluations."""
        y_val, p_raw, pi = sample_data
        
        def cost_metric(yt, yp):
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            return -(1.0 * fp + 10.0 * fn)
        
        alpha, tau, score, n_eval = alpha_search_analytical_tau(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            cost_fn=10.0,
            cost_fp=1.0,
            metric_fn=cost_metric
        )
        
        assert n_eval == 27, f"Default should be 27 evaluations, got {n_eval}"
    
    def test_path_b_analytical_tau(self, sample_data):
        """τ* should be calculated analytically from cost ratio."""
        y_val, p_raw, pi = sample_data
        
        cost_fn, cost_fp = 10.0, 1.0
        expected_tau = cost_fp / (cost_fp + cost_fn)  # 1/11 ≈ 0.0909
        
        def cost_metric(yt, yp):
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            return -(cost_fp * fp + cost_fn * fn)
        
        alpha, tau, score, n_eval = alpha_search_analytical_tau(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            metric_fn=cost_metric
        )
        
        assert tau == pytest.approx(expected_tau, rel=1e-6)
    
    def test_path_b_returns_valid_alpha(self, sample_data):
        """Should return valid α within specified range."""
        y_val, p_raw, pi = sample_data
        
        def cost_metric(yt, yp):
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            return -(1.0 * fp + 10.0 * fn)
        
        alpha_range = (0.1, 3.0)
        
        alpha, tau, score, n_eval = alpha_search_analytical_tau(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            cost_fn=10.0,
            cost_fp=1.0,
            metric_fn=cost_metric,
            alpha_range=alpha_range
        )
        
        assert alpha_range[0] <= alpha <= alpha_range[1]


# =============================================================================
# GET_SEARCH_CONFIG TESTS
# =============================================================================

class TestGetSearchConfig:
    """Tests for search configuration utility."""
    
    def test_path_a_config(self):
        """Path A configuration should show 540 evaluations."""
        config = get_search_config(path='A', n_alpha=27, n_tau=20)
        
        assert config['path'] == 'A (Metric-Driven)'
        assert config['total_evaluations'] == 540
        assert config['n_alpha'] == 27
        assert config['n_tau'] == 20
    
    def test_path_b_config(self):
        """Path B configuration should show 27 evaluations."""
        config = get_search_config(path='B', n_alpha=27)
        
        assert config['path'] == 'B (Cost-Sensitive)'
        assert config['total_evaluations'] == 27
        assert 'tau_formula' in config
    
    def test_custom_config(self):
        """Custom configuration should calculate correctly."""
        config = get_search_config(path='A', n_alpha=10, n_tau=5)
        
        assert config['total_evaluations'] == 50


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete ABTC workflow."""
    
    def test_path_a_improves_over_baseline(self, sample_data):
        """Path A should improve F2 over default threshold."""
        y_val, p_raw, pi = sample_data
        
        # Baseline: default τ = 0.5
        y_baseline = (p_raw >= 0.5).astype(int)
        f2_baseline = fbeta(y_val, y_baseline, 2.0)
        
        # Path A optimization
        metric_fn = lambda yt, yp: fbeta(yt, yp, 2.0)
        alpha, tau, f2_opt, _ = joint_alpha_tau_search(
            y_val=y_val,
            p_raw=p_raw,
            pi=pi,
            adjust_fn=alpha_adjust_posterior,
            metric_fn=metric_fn
        )
        
        # Apply optimal parameters
        p_adj = alpha_adjust_posterior(p_raw, alpha, pi)
        y_opt = (p_adj >= tau).astype(int)
        f2_final = fbeta(y_val, y_opt, 2.0)
        
        assert f2_final >= f2_baseline, f"Path A should improve: {f2_final} >= {f2_baseline}"
    
    def test_path_b_minimises_validation_cost_over_alpha_grid(self, sample_data):
        """Path B returns the α of its grid with the lowest validation cost at τ*."""
        y_val, p_raw, pi = sample_data
        cost_fn, cost_fp = 10.0, 1.0

        def calc_cost(y_true, y_pred):
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            return cost_fp * fp + cost_fn * fn

        alpha, tau, score, n_eval = alpha_search_analytical_tau(
            y_val=y_val, p_raw=p_raw, pi=pi, adjust_fn=alpha_adjust_posterior,
            cost_fn=cost_fn, cost_fp=cost_fp, metric_fn=lambda yt, yp: -calc_cost(yt, yp))

        grid_costs = [calc_cost(y_val, (alpha_adjust_posterior(p_raw, a, pi) >= tau).astype(int))
                      for a in np.linspace(0.1, 3.0, 27)]
        assert tau == pytest.approx(cost_fp / (cost_fp + cost_fn))
        assert -score == min(grid_costs)
        assert n_eval == 27


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestFastSearch:
    """The vectorised Path A search reaches the same best validation score."""

    def test_fast_equals_loop_score(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            n = int(rng.integers(40, 300))
            y = (rng.random(n) < 0.2).astype(int); y[0] = 1; y[1] = 0
            p = np.clip(0.3 * y + rng.normal(0.3, 0.2, n), 0.001, 0.999)
            pi = float(y.mean())
            _, _, s_loop, n1 = joint_alpha_tau_search(y, p, pi, alpha_adjust_posterior,
                                                      lambda yt, yp: fbeta(yt, yp, 2.0))
            _, _, s_fast, n2 = joint_alpha_tau_search_fast(y, p, pi, "F2")
            assert n1 == n2 == 540
            assert abs(s_loop - s_fast) < 1e-12
