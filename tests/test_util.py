from ranking.util import *
from ranking.parsing import read_match_list_from_match_list
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


def test_make_match_list_hashable():
    match_list = read_match_list_from_match_list("data/match_lists/dogs.txt")
    match_list_hashable = make_match_list_hashable(match_list)
    print(type(match_list_hashable))
    assert type(match_list_hashable) is tuple

    match_list_recovered = undo_make_match_list_hashable(match_list_hashable)
    assert match_list == match_list_recovered
