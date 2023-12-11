# Typical use case functions

import numpy as np
import pandas as pd
import warnings
import ranking
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from . import util


def scores(
    match_list,
    model_name="depth_and_luck",
    num_samples=10000,
    num_chains=4,
    force_mode=None,
):
    """Get fitted scores of players in a given model.

    :param match_list: List of matches, each represented by a dict of the winner and loser.
    :type match_list: list
    :param model_name: Model used for fitting. Defaults to 'depth_and_luck'. Options: {‘depth_and_luck’, ‘depth_only’, ‘luck_only’, ‘logistic_prior’}.
    :type model_name: str, optional
    :param num_samples: Number of samples used per chain for MCMC sampling, defaults to 10000
    :type num_samples: int, optional
    :param num_chains: Number of chains used for MCMC sampling, defaults to 4
    :type num_chains: int, optional
    :param force_mode: Optionally force the point estimate mode to be either 'average' or 'MAP', otherwise defaults to 'MAP' when more than 1000 players are present.
    :type force_mode: str or None, optional
    :return: pandas DataFrame with columns of the labels, inferred scores, and score errors for each player.
    :rtype: DataFrame
    """

    # Compute the number of participants present in match_list
    string_indices_dict = util.get_string_indices_dict(match_list)
    n = len(string_indices_dict)

    # Check the model_name is supported
    if model_name not in [
        "depth_and_luck",
        "depth_only",
        "luck_only",
        "logistic_prior",
    ]:
        raise AssertionError(f"model_name={model_name} not supported.")

    # Whether to find the expected values of the scores by sampling or use the MAP iteration method
    mode = "average"  # 'average' or 'MAP'
    if force_mode == None:
        if n < 1000:  # Prefer the sampling method if the data set is small enough
            mode = "average"
        else:  # If the data set is too large, default to using the simple iterative method
            warnings.warn(
                f"Data set contains {n} unique players, sampling may be slow for n > 1000.\n\
                        Defaulting to fast iterative method to find MAP estimates of the logistic_prior model.\n\
                            Can override this behavior with force_sampling=True."
            )
            mode = "MAP"
    elif force_mode == "average" or "MAP":
        mode = force_mode
    else:
        raise AssertionError(
            f"force_mode={force_mode} is not a valid option. \
                             Options are [None, 'average','MAP']"
        )

    if mode == "average":
        # Get samples from the posterior of this model fitted to this data set
        samples_df = ranking.samples(
            match_list,
            model_name=model_name,
            num_chains=num_chains,
            num_samples=num_samples,
        )

        # dict to be converted into scores DataFrame (more efficient to build this way)
        scores_df_dict = {"label": [], "score": [], "score_error": []}

        for column in samples_df.columns:  # Iterate over all sampled values
            if (
                column[-6:] == "_score"
            ):  # Check that this is a sampled score (not a parameter)
                # Extract the label name
                label = column[:-6]

                # Approximate the expected a posterori value and error by the sample mean and error
                sampled_values = samples_df[column]
                score_estimate = np.mean(sampled_values)
                score_error = np.std(sampled_values) / np.sqrt(len(sampled_values))

                # Write row to end of DataFrame
                scores_df_dict["label"].append(label)
                scores_df_dict["score"].append(score_estimate)
                scores_df_dict["score_error"].append(score_error)

        # Convert to DataFrame
        scores_df = pd.DataFrame(scores_df_dict)

        # Sort results by the estimated scores in descending order
        scores_df.sort_values(by="score", inplace=True, ascending=False)
        scores_df.reset_index(drop=True, inplace=True)

        # Set the index to start at 1 to nicely show the rank
        scores_df.index = range(1, len(scores_df) + 1)

        return scores_df
    else:
        # TODO: make logistic_prior_MAP faster
        MAP_scores, MAP_score_errors = ranking.logistic_prior_MAP(match_list)

        # Package information into a DataFrame
        labels = list(string_indices_dict.keys())
        scores_df_dict = {
            "label": labels,
            "score": MAP_scores,
            "score_error": MAP_score_errors,
        }

        # Convert to DataFrame
        scores_df = pd.DataFrame(scores_df_dict)

        # Sort results by the estimated scores in descending order
        scores_df.sort_values(by="score", inplace=True, ascending=False)
        scores_df.reset_index(drop=True, inplace=True)

        # Set the index to start at 1 to nicely show the rank
        scores_df.index = range(1, len(scores_df) + 1)

        return scores_df


def ranks(
    match_list,
    model_name="depth_and_luck",
    num_samples=10000,
    num_chains=4,
    force_mode=None,
):
    """Find the rankings of the players according to a specified model.

    :param match_list: List of matches, each represented by a dict of the winner and loser.
    :type match_list: list
    :param model_name: Model used for fitting. Defaults to 'depth_and_luck'. Options: {‘depth_and_luck’, ‘depth_only’, ‘luck_only’, ‘logistic_prior’}.
    :type model_name: str, optional
    :param num_samples: Number of samples used per chain for MCMC sampling, defaults to 10000
    :type num_samples: int, optional
    :param num_chains: Number of chains used for MCMC sampling, defaults to 4
    :type num_chains: int, optional
    :param force_mode: Optionally force the point estimate mode to be either 'average' or 'MAP', otherwise defaults to 'MAP' when more than 1000 players are present.
    :type force_mode: str or None, optional
    :return: Ranked list of the players in descending order of strength.
    :rtype: list
    """
    scores_df = scores(
        match_list,
        model_name=model_name,
        num_samples=num_samples,
        num_chains=num_chains,
        force_mode=force_mode,
    )

    return list(scores_df["label"])


def probability(
    match_list,
    winner_label,
    loser_label,
    model_name="depth_and_luck",
    num_samples=10000,
    num_chains=4,
    force_mode=None,
):
    """Inferred probability that one player will beat another, according to match_list data.

    :param match_list: List of matches, each represented by a dict of the winner and loser.
    :type match_list: list
    :param winner_label: Label of the desired winner
    :type winner_label: str
    :param loser_label: Label of the desired loser
    :type loser_label: str
    :param model_name: Model used for fitting. Defaults to 'depth_and_luck'. Options: {‘depth_and_luck’, ‘depth_only’, ‘luck_only’, ‘logistic_prior’}.
    :type model_name: str, optional
    :param num_samples: Number of samples used per chain for MCMC sampling, defaults to 10000
    :type num_samples: int, optional
    :param num_chains: Number of chains used for MCMC sampling, defaults to 4
    :type num_chains: int, optional
    :param force_mode: Optionally force the point estimate mode to be either 'average' or 'MAP', otherwise defaults to 'MAP' when more than 1000 players are present.
    :type force_mode: str or None, optional
    :raises AssertionError: If winner_label or loser_label is not present in match_list.
    :return: Tuple of the inferred probability and the error in the estimation.
    :rtype: tuple
    """
    # Compute the number of participants present in match_list
    string_indices_dict = util.get_string_indices_dict(match_list)
    n = len(string_indices_dict)

    # Check the model_name is supported
    if model_name not in [
        "depth_and_luck",
        "depth_only",
        "luck_only",
        "logistic_prior",
    ]:
        raise AssertionError(f"model_name={model_name} not supported.")

    # Check that the desired winner and loser are present in the data set
    if winner_label not in string_indices_dict.keys():
        raise AssertionError(f"{winner_label} not found in match_list.")
    if loser_label not in string_indices_dict.keys():
        raise AssertionError(f"{loser_label} not found in match_list.")

    # Whether to find the expected values of the scores by sampling or use the MAP iteration method
    mode = "average"  # 'average' or 'MAP'
    if force_mode == None:
        if n < 1000:  # Prefer the sampling method if the data set is small enough
            mode = "average"
        else:  # If the data set is too large, default to using the simple iterative method
            warnings.warn(
                f"Data set contains {n} unique players, sampling may be slow for n > 1000.\n\
                        Defaulting to fast iterative method to find MAP estimates of the logistic_prior model.\n\
                            Can override this behavior with force_sampling=True."
            )
            mode = "MAP"
    elif force_mode == "average" or "MAP":
        mode = force_mode
    else:
        raise AssertionError(
            f"force_mode={force_mode} is not a valid option. \
                             Options are [None, 'average','MAP']"
        )

    # Use sampling if the data set is not too large or the user has specified to
    if mode == "average":
        # Get samples from the posterior of this model fitted to this data set
        samples_df = ranking.samples(
            match_list,
            model_name=model_name,
            num_chains=num_chains,
            num_samples=num_samples,
        )

        # Get the total number of samples taken over all chains
        total_num_samples = len(samples_df)

        # Get the score differences between the relevant players
        # Need to add '_score' to the label for its column name
        sampled_score_differences = np.array(
            samples_df[f"{winner_label}_score"] - samples_df[f"{loser_label}_score"]
        )

        # Get parameter values. Depending on the model can set to different default values.
        match model_name:
            case "depth_and_luck":
                alphas = np.array(samples_df["alpha"])
                betas = np.array(samples_df["beta"])
            case "depth_only":
                alphas = np.zeros(total_num_samples)
                betas = np.array(samples_df["beta"])
            case "luck_only":
                alphas = np.array(samples_df["alpha"])
                betas = 100 * np.ones(
                    total_num_samples
                )  # We use this to approximate a true step function
            case "logistic_prior":
                alphas = np.zeros(total_num_samples)
                betas = np.ones(total_num_samples)
            case _:
                alphas = np.zeros(total_num_samples)
                betas = np.ones(total_num_samples)

        # Compute the probability of winning according to each sample
        probabilities = util.score_function(
            sampled_score_differences, alpha=alphas, beta=betas
        )

        # Probability is the mean over the samples
        probability = np.mean(probabilities)
        # Can estimate the error in the point estimate by the standard error
        probability_error = np.std(probabilities) / np.sqrt(total_num_samples)

        return (probability, probability_error)
    else:  # If the data set is too large, default to using the simple iterative method
        # TODO: make logistic_prior_MAP faster
        MAP_scores, MAP_score_errors = ranking.logistic_prior_MAP(match_list)

        winner_index = string_indices_dict[winner_label]
        loser_index = string_indices_dict[loser_label]
        score_difference = MAP_scores[winner_index] - MAP_scores[loser_index]
        # Adding the errors in quadrature as if they were independent
        score_difference_error = np.sqrt(
            MAP_score_errors[winner_index] ** 2 + MAP_score_errors[loser_index] ** 2
        )

        # Point estimate of the probability
        probability = util.score_function(score_difference, alpha=0, beta=1)
        # Estimate the error by changing he score difference by the score_error
        probability_upper = util.score_function(
            score_difference + score_difference_error, alpha=0, beta=1
        )
        probability_lower = util.score_function(
            score_difference - score_difference_error, alpha=0, beta=1
        )
        probability_error = (probability_upper - probability_lower) / 2

        return (probability, probability_error)


def depth_and_luck(
    match_list, model_name="depth_and_luck", num_samples=10000, num_chains=4
):
    """Get expected values of the depth and luck model parameters for a match_list by sampling. Returns appropriate stand-in parameters if model_name does not vary those parameters.

    :param match_list: List of matches, each represented by a dict of the winner and loser.
    :type match_list: list
    :param model_name: Model used for fitting. Defaults to 'depth_and_luck'. Options: {‘depth_and_luck’, ‘depth_only’, ‘luck_only’, ‘logistic_prior’}.
    :type model_name: str, optional
    :param num_samples: Number of samples used per chain for MCMC sampling, defaults to 10000
    :type num_samples: int, optional
    :param num_chains: Number of chains used for MCMC sampling, defaults to 4
    :type num_chains: int, optional
    :return: dict of depth (beta) and luck (alpha)
    :rtype: dict
    """
    # Get samples from the posterior of this model fitted to this data set
    samples_df = ranking.samples(
        match_list,
        model_name=model_name,
        num_chains=num_chains,
        num_samples=num_samples,
    )

    # Get the total number of samples taken over all chains
    total_num_samples = len(samples_df)

    # Estimate parameters by the sample means.
    # If a parameter is not sampled over return its appropriate fixed values
    # Note that for small data sets the expected value of beta can be very large,
    # since the first moment of the prior is unbounded: \int d\beta \beta P(\beta) = \infty
    match model_name:
        case "depth_and_luck":
            alphas = np.array(samples_df["alpha"])
            betas = np.array(samples_df["beta"])
        case "depth_only":
            alphas = np.zeros(total_num_samples)
            betas = np.array(samples_df["beta"])
        case "luck_only":
            alphas = np.array(samples_df["alpha"])
            betas = 100 * np.ones(
                total_num_samples
            )  # We use this to approximate a true step function
        case "logistic_prior":
            alphas = np.zeros(total_num_samples)
            betas = np.ones(total_num_samples)
        case _:
            alphas = np.zeros(total_num_samples)
            betas = np.ones(total_num_samples)

    alpha = np.mean(alphas)
    beta = np.mean(betas)
    return {"luck": alpha, "depth": beta}


def draw_depth_and_luck_posterior(match_list, num_samples=10000, num_chains=4):
    """Draw the posterior distribution of the luck and depth parameters from sampled values of the depth_and_luck model.

    :param match_list: List of matches, each represented by a dict of the winner and loser.
    :type match_list: list
    :param num_samples: Number of samples used per chain for MCMC sampling, defaults to 10000
    :type num_samples: int, optional
    :param num_chains: Number of chains used for MCMC sampling, defaults to 4
    :type num_chains: int, optional
    """
    # We can only use the depth_and_luck full model for this purpose
    samples_df = ranking.samples(
        match_list,
        model_name="depth_and_luck",
        num_chains=num_chains,
        num_samples=num_samples,
    )

    # Get the parameter samples
    alphas = samples_df["alpha"]
    betas = samples_df["beta"]

    # Show results uniform in theta = arctan(beta / 4) in order to show beta from 0 to infty
    alpha_spacing = 0.0025  # Setting the resolution
    theta_spacing = 0.005  # Setting the resolution
    # Define all possible values of alpha, beta, and theta (transformed beta) where we may evaluate the density estimate
    all_alphas = np.arange(0, 1, alpha_spacing)
    all_thetas = np.arange(0, np.pi / 2, theta_spacing)
    all_betas = util.theta_to_beta(all_thetas)

    # Restrict to the alphas and betas within the range of observed alphas and betas, this is where the grid will be computed
    compute_alphas = []
    for alpha in all_alphas:
        if alphas.min() < alpha and alpha < alphas.max():
            compute_alphas.append(alpha)
    compute_alphas = np.array(compute_alphas)
    compute_betas = []
    for beta in all_betas:
        if betas.min() < beta and beta < betas.max():
            compute_betas.append(beta)
    compute_betas = np.array(compute_betas)

    # Create (alpha,theta) pairs where we want to evaluate
    X, Y = np.meshgrid(compute_alphas, util.beta_to_theta(compute_betas))
    alpha_theta_pairs = np.vstack([X.ravel(), Y.ravel()])

    # Plot the samples and the Gaussian kernel smoothing (performed in (alpha,theta) space)
    thetas = util.beta_to_theta(betas)  # Compute the corresponding theta values

    # So that the heat map does not get cut off at alpha = 0, we mirror the points across
    # the alpha = 0 axis for density estimate calculation
    alphas_mirrored = []
    thetas_mirrored = []
    for i in range(len(alphas)):
        alphas_mirrored.append(alphas[i])
        thetas_mirrored.append(thetas[i])
        if alphas[i] < 0.03:  # Add an extra mirrored version
            alphas_mirrored.append(-alphas[i])
            thetas_mirrored.append(thetas[i])
    values = np.vstack([alphas_mirrored, thetas_mirrored])

    # Compute the gaussian kernel estimate
    kernel = gaussian_kde(values)
    kernal_vals = kernel(alpha_theta_pairs).T
    Z = np.reshape(kernal_vals, X.shape)

    # Plot the kernel estimate as a heatmap
    fig, ax = plt.subplots()
    ax.imshow(
        np.rot90(Z),
        cmap=plt.cm.gist_earth_r,
        extent=[np.min(thetas), np.max(thetas), np.min(alphas), np.max(alphas)],
    )

    # Plot the sampled points. Use less opacity when over 1000 sampled points.
    dot_alpha = min(1, float(1000) / len(alphas))
    ax.plot(thetas, alphas, "k.", markersize=2, alpha=dot_alpha)

    # Write in the tick values with the appropriate values of beta (ticks are uniform in theta)
    xtick_vals = []
    for x in np.arange(0, np.pi / 2, 0.1):
        if np.min(thetas) < x and x < np.max(thetas):
            xtick_vals.append(x)
    xtick_vals = np.array(xtick_vals)
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels([f"{beta:.2f}" for beta in util.theta_to_beta(xtick_vals)])

    # Other plotting information
    ax.set_xlabel(r"$\beta$ (depth)")
    ax.set_ylabel(r"$\alpha$ (luck)")
    ax.set_xlim([np.min(thetas), np.max(thetas)])
    ax.set_ylim([np.min(alphas), np.max(alphas)])
    ax.set_aspect(1)
