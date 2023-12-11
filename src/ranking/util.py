import numpy as np


def beta_to_theta(beta):
    """Convert beta to theta

    Args:
        beta (float): depth parameter

    Returns:
        float: theta (angle of score function at 0)
    """
    return np.arctan(beta / 4)


def theta_to_beta(theta):
    """Convert theta to beta

    Args:
        theta (float): angle of score function at 0 (when alpha = 0)

    Returns:
        float: beta (depth parameter)
    """
    return 4 * np.tan(theta)


def sigmoid(x):
    """Sigmoid function, overflow protected and vectorized

    Args:
        x (float):

    Returns:
        float:
    """
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + 1))


def score_function(s, alpha=0.0, beta=1.0):
    """Score function f(s) for luck and depth parameter values.

    Args:
        s (float): score difference
        alpha (float, optional): luck parameter. Defaults to 0.
        beta (float, optional): depth parameter. Defaults to 1.

    Returns:
        float: score function value
    """
    return 0.5 * alpha + (1 - alpha) * sigmoid(beta * s)


def get_string_indices_dict(match_list):
    """Generate a dict which maps each name (string) found to an index in the order they appear.

    Args:
        match_list (list): List of matches, each represented by a dict of the winner and loser.

    Returns:
        dict: Dict that associates to each found string a unique index.
    """
    string_indices_dict = {}  # Dictionary of {string: index} for quick access

    for match in match_list:
        # Find (or assign) indices to the new labels
        winner_label = match["winner"]
        loser_label = match["loser"]
        if winner_label not in string_indices_dict.keys():
            string_indices_dict.update({winner_label: len(string_indices_dict)})
        if loser_label not in string_indices_dict.keys():
            string_indices_dict.update({loser_label: len(string_indices_dict)})

    return string_indices_dict


def make_match_list_hashable(match_list):
    """Turn a match_list into a hashable flattened tuple for caching.

    Args:
        match_list (list): List of matches, each represented by a dict of the winner and loser.

    Returns:
        tuple: match_list_hashable, flattened version of match_list.
    """
    match_list_flat = []
    for match in match_list:
        match_list_flat.append(match["winner"])
        match_list_flat.append(match["loser"])
    return tuple(match_list_flat)


def undo_make_match_list_hashable(match_list_hashable):
    """Undo the make_match_list_hashable operation to recover match_list

    Args:
        match_list_hashable (tuple): Flattened version of match_list.

    Raises:
        AssertionError: match_list_hashable must have even length.

    Returns:
        list: List of matches, each represented by a dict of the winner and loser.
    """
    if not len(match_list_hashable) % 2 == 0:
        raise AssertionError(
            f"match_list_hashable does not have an even number of entries."
        )

    match_list = []
    match_index = 0
    while match_index < len(match_list_hashable):
        # Convert the next two entries into a match
        match = {
            "winner": match_list_hashable[match_index],
            "loser": match_list_hashable[match_index + 1],
        }
        match_list.append(match)
        # Increment to process next pair
        match_index += 2

    return match_list
