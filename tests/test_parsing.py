from ranking.parsing import *
import pytest


def test_read_outcome_list_too_many_spaces():
    with pytest.raises(AssertionError):
        assert read_match_list_from_match_list(
            "tests/test_case_files/spaces_in_names.txt"
        )
