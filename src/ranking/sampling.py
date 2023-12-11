# Use pystan to take samples from the various models
import stan
import numpy as np
from functools import lru_cache
import warnings
import pandas as pd
from . import util  # Local import

# STAN model strings
depth_and_luck_STAN_string = """
data {
  int<lower = 0> N;                             // Number of players
  int<lower = 0> M;                             // Number of unique pairings
  array[M] int<lower=1, upper = N> player1;     // The index of "player1" in each pairing
  array[M] int<lower=1, upper = N> player2;     // The index of "player2" in each pairing
  array[M] real<lower = 0> A12;                 // Number of wins recorded by player1 against player2
}
parameters {
  vector[N] scores;                             // Score of each player
  real<lower = 0> beta;                         // Depth of competition
  real<lower = 0, upper = 1> alpha;             // Luck 
}
model {
  target += log(1/(4*(1 + beta^2/16)));         // P(beta), cauchy prior of width 4 (uniform over theta)
  for( k in 1:N){                               // P(s_i), prior on scores
    target += -(scores[k])^2;                   // Prior has variance 1/2 so that the difference between two scores is a Gaussian of width 1
  }
  for(i in 1:M){                                // P(A|s_i,alpha,beta), model likelihood
    real ds = scores[player1[i]] - scores[player2[i]];  // Score difference 
    target += A12[i]*log(alpha*0.5+(1-alpha)*1/(1+e()^(-beta*ds))); // Probability of times that player1 beat player2
  }
}
"""

depth_only_STAN_string = """
data {
  int<lower = 0> N;                             // Number of players
  int<lower = 0> M;                             // Number of unique pairings
  array[M] int<lower=1, upper = N> player1;     // The index of "player1" in each pairing
  array[M] int<lower=1, upper = N> player2;     // The index of "player2" in each pairing
  array[M] real<lower = 0> A12;                 // Number of wins recorded by player1 against player2
}
parameters {
  vector[N] scores;                             // Score of each player
  real<lower = 0> beta;                         // Depth of competition
}
model {
  target += log(1/(4*(1 + beta^2/16)));         // P(beta), cauchy prior of width 4 (uniform over theta)
  for( k in 1:N){                               // P(s_i), prior on scores
    target += -(scores[k])^2;                   // Prior has variance 1/2 so that the difference between two scores is a Gaussian of width 1
  }
  for(i in 1:M){                                // P(A|s_i,beta), model likelihood
    real ds = scores[player1[i]] - scores[player2[i]];  // Score difference 
    target += A12[i]*log(1/(1+e()^(-beta*ds))); // Probability of times that player1 beat player2
  }
}
"""

luck_only_STAN_string = """
data {
  int<lower = 0> N;                             // Number of players
  int<lower = 0> M;                             // Number of unique pairings
  array[M] int<lower=1, upper = N> player1;     // The index of "player1" in each pairing
  array[M] int<lower=1, upper = N> player2;     // The index of "player2" in each pairing
  array[M] real<lower = 0> A12;                 // Number of wins recorded by player1 against player2
}
parameters {
  vector[N] scores;                             // Score of each player
  real<lower = 0, upper = 1> alpha;             // Luck 
}
model {
  for( k in 1:N){                               // P(s_i), prior on scores
    target += -(scores[k])^2;                   // Prior has variance 1/2 so that the difference between two scores is a Gaussian of width 1
  }
  for(i in 1:M){                                // P(A|s_i,alpha), model likelihood. We slightly fake this by using beta = 100 to retain a continuous (and very nearly equivalent) model
    real ds = scores[player1[i]] - scores[player2[i]];  // Score difference 
    target += A12[i]*log(alpha*0.5+(1-alpha)*1/(1+e()^(-100*ds))); // Probability of times that player1 beat player2
  }
}
"""

logistic_prior_STAN_string = """
data {
  int<lower = 0> N;                             // Number of players
  int<lower = 0> M;                             // Number of unique pairings
  array[M] int<lower=1, upper = N> player1;     // The index of "player1" in each pairing
  array[M] int<lower=1, upper = N> player2;     // The index of "player2" in each pairing
  array[M] real<lower = 0> A12;                 // Number of wins recorded by player1 against player2
}
parameters {
  vector[N] scores;                             // Score of each player
}
model {
  for( k in 1:N){                               // P(s_i), prior on scores
    target += log(e()^(-scores[k])/(1+e()^(-scores[k]))^2); // Logistic prior
  }
  for(i in 1:M){                                // P(A|s_i), model likelihood
    real ds = scores[player1[i]] - scores[player2[i]];  // Score difference 
    target += A12[i]*log(1/(1+e()^(-ds)));      // Probability of times that player1 beat player2
  }
}
"""

# STAN language strings associated to each named model
STAN_strings_dict = {
    "depth_and_luck": depth_and_luck_STAN_string,
    "depth_only": depth_only_STAN_string,
    "luck_only": luck_only_STAN_string,
    "logistic_prior": logistic_prior_STAN_string,
}


def get_STAN_data(match_list, string_indices_dict):
    """Convert the match_list into a format for our STAN models.

    Args:
        match_list (list): List of matches, each represented by a dict of the winner and loser.
        string_indices_dict (dict): Dict that associates to each found string a unique index.

    Returns:
        dict: STAN_data representation of the match_list for STAN.
    """
    n = len(string_indices_dict)  # Number of players

    # Initializing STAN_data variables to be built up
    m = 0  # Number of unique pairings
    player1 = []  # The index of "player1" in each pairing
    player2 = []  # The index of "player2" in each pairing
    A12 = []  # Number of wins recorded by player1 against player2

    # Map the pairings present in the data to an index in the STAN_data arrays
    pairing_indices_dict = {}

    for match in match_list:
        winner_index = string_indices_dict[match["winner"]]
        loser_index = string_indices_dict[match["loser"]]

        # Store a pairing as the tuple [hashable!] (winner_index,loser_index)
        pairing = (winner_index, loser_index)

        # Assign this pairing an index if haven't already and add entries to
        # the STAN_data variables to represent this pairing.
        if pairing not in pairing_indices_dict.keys():
            pairing_indices_dict[pairing] = len(pairing_indices_dict)
            # Participants
            player1.append(pairing[0] + 1)  # STAN indices are unary
            player2.append(pairing[1] + 1)  # STAN indices are unary
            # Win counts
            A12.append(0)
            # Number of edges
            m += 1

        # Increment the STAN_data variables
        A12[pairing_indices_dict[pairing]] += 1

    STAN_data = {
        "N": int(n),
        "M": int(m),  # Number of unique pairings
        "player1": player1,  # The index of "player1" in each pairing
        "player2": player2,  # The index of "player2" in each pairing
        "A12": A12,
    }  # Number of wins recorded by player1 against player2

    return STAN_data


@lru_cache(maxsize=16)  # Cache samples taken for reuse (uses memory)
def _samples(
    match_list_hashable,
    model_name="depth_and_luck",
    num_chains=4,
    num_samples=10000,
    **kwargs,
):
    # Convert the hashable match_list back into our usual form
    match_list = util.undo_make_match_list_hashable(match_list_hashable)

    # STAN model string to be passed to pystan
    STAN_model_string = STAN_strings_dict[model_name]

    print(f"Using model {model_name}")

    # Assign an index to each string in the match list
    string_indices_dict = util.get_string_indices_dict(match_list)

    # Get data from each of the
    STAN_data = get_STAN_data(match_list, string_indices_dict)

    # Compile and sample from the model (with the given data) using pystan
    posterior = stan.build(STAN_model_string, data=STAN_data, **kwargs)
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)

    # Convert the resulting samples into a DataFrame with the original labels
    # DataFrame containing the STAN output (useful for diagnostics)
    df_original = fit.to_frame()
    # Dictionary to be converted into the final DataFrame
    df_dict = {}

    # Include the sampled model parameters if they are present
    for param in ["alpha", "beta"]:
        if param in df_original.columns:
            df_dict[param] = df_original[param]

    for name, index in string_indices_dict.items():
        STAN_index = index + 1  # STAN is unary
        STAN_label = f"scores.{STAN_index}"
        if STAN_label in df_original.columns:
            df_dict[f"{name}_score"] = df_original[STAN_label]
        else:
            raise AssertionError(f"{STAN_label} not found in pystan output.")

    # Convert to DataFrame
    df = pd.DataFrame(df_dict)

    return df


# TODO: switch to a permanent cache between executables. Store samples in local directory?
# Should probably add the option to not cache samples as well
def samples(
    match_list, model_name="depth_and_luck", num_chains=4, num_samples=10000, **kwargs
):
    """Get MCMC samples from the model fit to a given match_list.

    Parameters in ``kwargs`` will be passed to the python wrapper of
        ``stan::services::sample::hmc_nuts_diag_e_adapt`` in pystan.

    Args:
        match_list (list): List of matches, each represented by a dict of the winner and loser.
        model_name (str, optional): Model used for fitting. Defaults to "depth_and_luck". Options: {‘depth_and_luck’, ‘depth_only’, ‘luck_only’, ‘logistic_prior’}.
        num_chains (int, optional): Number of chains STAN will use for sampling. Defaults to 4.
        num_samples (int, optional): _description_. Defaults to 10000.
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # Give warning if the data set is too large for fast HMC sampling
    n = len(util.get_string_indices_dict(match_list))
    if n > 1000:
        # TODO: update warning with whatever we end up calling the fast iteration method
        warnings.warn(
            f"Data set contains {n} unique players, sampling may be slow for n > 1000.\n\
                      Consider using fast iterative MAP estimation methods. \n\
                        (Only available for the logistic_prior model)."
        )

    # Make the match_list hashable so that we can cache samples with lru_cache
    match_list_hashable = util.make_match_list_hashable(match_list)

    return _samples(
        match_list_hashable,
        model_name=model_name,
        num_chains=num_chains,
        num_samples=num_samples,
        **kwargs,
    )
