from ranking.parsing import *
import pytest


def test_read_outcome_list_too_many_spaces():
    with pytest.raises(AssertionError):
        assert read_match_list_from_match_list("tests/bad_files/spaces_in_names.txt")
