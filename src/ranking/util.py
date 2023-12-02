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
    """Sigmoid function, overflow protected

    Args:
        x (float):

    Returns:
        float:
    """
    if x > 0:  # Normal sigmoid
        return 1 / (1 + np.exp(-x))
    else:  # Flip for large negative values
        return np.exp(x) / (np.exp(x) + 1)


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

def get_string_indices_dict_from_match_list(match_list):
    """Generate a dict which maps each name (string) found to an index in the order they appear

    Args:
        match_list (list): List of matches, each represented by a dict of the winner and loser 

    Returns:
        dict: Dict that associates to each found string a unique index
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