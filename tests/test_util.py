from ranking.util import *
import numpy as np
import pytest

numerical_tolerance = 0.0001


def test_beta_to_theta():
    test_beta = 2
    beta_error = test_beta - theta_to_beta(beta_to_theta(test_beta))
    assert np.abs(beta_error) < numerical_tolerance


def test_theta_to_beta():
    test_theta = 1
    theta_error = test_theta - beta_to_theta(theta_to_beta(test_theta))
    assert np.abs(theta_error) < numerical_tolerance


def test_score_function_default():
    sig_error = score_function(2) - 0.880797  # From Mathematica
    assert np.abs(sig_error) < numerical_tolerance


def test_score_function():
    sig_error = score_function(0.2, 0.1, 10) - 0.842717  # From Mathematica
    assert np.abs(sig_error) < numerical_tolerance
