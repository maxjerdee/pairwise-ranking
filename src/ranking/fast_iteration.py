from . import util
import numpy as np

# Fast iterative method to find the MAP scores of the logistic_prior model


# TODO: Implement as a wrapper of a c++ function (and also make sure the rest of pipeline is efficient)
def logistic_prior_MAP(match_list, max_iters=1000, threshold=1e-8):
    """Get MAP estimates of the logistic_prior model with rapid iterative methods.

    :param match_list: List of matches, each represented by a dict of the winner and loser.
    :type match_list: list
    :param max_iters: Maximum number of iterations to perform, defaults to 1000
    :type max_iters: int, optional
    :param threshold: Average change in scores over an iteration at which to stop iterations, defaults to 1e-8
    :type threshold: float, optional
    :return: pandas DataFrame of labels, scores, and errors (inferred by change over an iteration).
    :rtype: DataFrame
    """
    # Convert match_list to a sparse format of neighbors for iteration
    string_indices_dict = util.get_string_indices_dict(match_list)
    n = len(string_indices_dict)

    # TODO: at least take advantage of the sparsity of A
    A = np.zeros((n, n))
    for match in match_list:
        winner_index = string_indices_dict[match["winner"]]
        loser_index = string_indices_dict[match["loser"]]
        A[winner_index, loser_index] += 1

    # Initialize choices of strengths (pi[i] = exp(s[i]))
    pis = np.ones(n)

    # Will estimate the errors in s simply by the change in each iteration
    estimated_s_errors = np.ones(n)

    # Implement the iterative method of Eq. (27) in Newman 2022
    # to find MAP scores for the Bradley-Terry with logistic prior
    # This is approximately our model with alpha = 0, beta = 2.56
    for iter in range(max_iters):
        # Important to perform the updates asynchronously,
        # the synchronous version does not necessarily converge
        for i in range(n):
            # 1/(pi_i + 1) + sum_j A_{ij} pi_j/(pi_i + pi_j)
            numerator = 1 / (pis[i] + 1) + np.dot(A[i, :], pis / (pis[i] + pis))
            # 1/(pi_i + 1) + sum_j A_{ji}/(pi_i + pi_j)
            denominator = 1 / (pis[i] + 1) + np.dot(A[:, i], 1 / (pis[i] + pis))
            new_pi = numerator / denominator

            # Estimate the error in s
            s_change = (new_pi - pis[i]) / pis[
                i
            ]  # Ds = log(pi + Dpi) - log(pi) \approx Dpi/pi
            pis[i] = new_pi
            estimated_s_errors[i] = np.abs(s_change)

        # Check if the average estimated error is below threshold
        RMS_s_error = np.sqrt(np.mean(estimated_s_errors**2))
        if RMS_s_error < threshold:
            break

    # Convert back to scores
    estimated_s = np.log(pis)

    return estimated_s, estimated_s_errors
