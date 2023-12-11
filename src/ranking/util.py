import numpy as np


def beta_to_theta(beta):
    """Convert beta to theta

    :param beta: depth parameter
    :type beta: float
    :return: theta (angle of score function at 0)
    :rtype: float
    """
    return np.arctan(beta / 4)


def theta_to_beta(theta):
    """Convert theta to beta

    :param theta: (angle of score function at 0)
    :type theta: float
    :return: beta depth parameter
    :rtype: float
    """
    return 4 * np.tan(theta)


def sigmoid(x):
    """Sigmoid function, overflow protected and vectorized.

    :param x: input
    :type x: float
    :return: sigmoid
    :rtype: float
    """
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + 1))


"""Score function f(s) for luck and depth parameter values.

    Args:
        s (float): score difference
        alpha (float, optional): luck parameter. Defaults to 0.
        beta (float, optional): depth parameter. Defaults to 1.

    Returns:
        float: score function value
    """


def score_function(s, alpha=0.0, beta=1.0):
    """Score function f(s) for luck and depth parameter values.

    :param s: score difference
    :type s: float
    :param alpha: luck parameter, defaults to 0.0
    :type alpha: float, optional
    :param beta: depth parameter, defaults to 1.0
    :type beta: float, optional
    :return: score function value
    :rtype: float
    """
    return 0.5 * alpha + (1 - alpha) * sigmoid(beta * s)


def get_string_indices_dict(match_list):
    """Generate a dict which maps each label found to an index in the order they appear.

    :param match_list: List of matches, each represented by a dict of the winner and loser.
    :type match_list: list
    :return: dict that associates to each found label a unique index.
    :rtype: dict
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

    :param match_list: List of matches, each represented by a dict of the winner and loser.
    :type match_list: list
    :return: Flattened version of match_list
    :rtype: tuple
    """
    match_list_flat = []
    for match in match_list:
        match_list_flat.append(match["winner"])
        match_list_flat.append(match["loser"])
    return tuple(match_list_flat)


def undo_make_match_list_hashable(match_list_hashable):
    """Undo the make_match_list_hashable operation to recover match_list

    :param match_list_hashable: Flattened version of match_list
    :type match_list_hashable: tuple
    :raises AssertionError: If match_list_hashable does not have even length.
    :return: List of matches, each represented by a dict of the winner and loser.
    :rtype: list
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
