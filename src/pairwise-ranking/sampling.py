# `get_samples.py` uses STAN (through pystan) to generate samples from the posterior distributions 
# of 4 models on the full datasets.

# These models are:
# alpha_beta_model: scores, hyperparameters alpha and beta, Cauchy Prior of width 4 on beta. 
# bradley_terry: scores, hyperparameter beta (alpha = 0)
# step_function: scores, hyperparameter alpha. (step function is implemented as beta = 100)
# bradley_terry_eta: scores, fixed logisitic prior on the scores

# These samples are then written to the `/full_samples` directory.

# The script `get_samples.py` may also be used, for `crossvalidation = True` to generate samples for a 
# number of 5-fold cross-validation splits. That is, samples from the posterior distribution given the 
# training portion of each cross-validation split. These samples are then written to the `/crossvalidation_samples` directory. 

# Sample usage: 
# python get_samples.py data/test.txt

crossvalidation = False
# # num_samples = 4000 # Long
num_samples = 1000 # Default
num_chains = 4
# crossvalidation = True
# num_samples = 1000
# num_chains = 4
num_crossvalidations = 50
full_samples_folder = "full_samples"
crossvalidation_samples_folder = "crossvalidation_samples"

import json
import numpy as np
import util # local
import sys
import stan
import os

# STAN model strings for sampling from
alpha_beta_model = """
data {
  int<lower = 0> N;                     // players
  int<lower = 0> M;                     // matches
  array[M] int<lower=1, upper = N> player1;   // player 1 for interaction n
  array[M] int<lower=1, upper = N> player2;   // player 2 for interation n
  array[M] real<lower = 0> A12;                // Number of wins by player1 against player2
  array[M] real<lower = 0> A21;                // Number of wins by player2 against player1
}
parameters {
  vector[N] scores;                      // ability for player n
  real<lower = 0> beta; // Depth of competition
  real<lower = 0, upper = 1> alpha; // Luck (in the mixture model)
}
model {
  target += log(1/(4*(1 + beta^2/16))); // Becomes uniform over theta
  for( k in 1:N){ // P(s_i)
    target += -(scores[k])^2; // Prior has variance 1/2 so that the difference between two scores is a Gaussian of width 1
  }
  for(i in 1:M){
    real ds = scores[player1[i]] - scores[player2[i]];
    target += A12[i]*log(alpha*0.5+(1-alpha)*1/(1+e()^(-beta*ds))) + A21[i]*log(alpha*0.5+(1-alpha)*1/(1+e()^(beta*ds)));
  }
}
"""

bradley_terry_model = """
data {
  int<lower = 0> N;                     // players
  int<lower = 0> M;                     // matches
  array[M] int<lower=1, upper = N> player1;   // player 1 for interaction n
  array[M] int<lower=1, upper = N> player2;   // player 2 for interation n
  array[M] real<lower = 0> A12;                // Number of wins by player1 against player2
  array[M] real<lower = 0> A21;                // Number of wins by player2 against player1
}
parameters {
  vector[N] scores;                      // ability for player n
  real<lower = 0> beta; // Depth of competition
}
model {
  target += log(1/(4*(1 + beta^2/16))); // Becomes uniform over theta
  for( k in 1:N){ // P(s_i)
    target += -(scores[k])^2; // Prior has variance 1/2 so that the difference between two scores is a Gaussian of width 1
  }
  for(i in 1:M){
    real ds = scores[player1[i]] - scores[player2[i]];
    target += A12[i]*log(1/(1+e()^(-beta*ds))) + A21[i]*log(1/(1+e()^(beta*ds)));
  }
}
"""

step_function_model = """
data {
  int<lower = 0> N;                     // players
  int<lower = 0> M;                     // matches
  array[M] int<lower=1, upper = N> player1;   // player 1 for interaction n
  array[M] int<lower=1, upper = N> player2;   // player 2 for interation n
  array[M] real<lower = 0> A12;                // Number of wins by player1 against player2
  array[M] real<lower = 0> A21;                // Number of wins by player2 against player1
}
parameters {
  vector[N] scores;                      // ability for player n
  real<lower = 0, upper = 1> alpha; // Luck (in the mixture model)
}
model {
  for( k in 1:N){ // P(s_i)
    target += -(scores[k])^2; // Prior has variance 1/2 so that the difference between two scores is a Gaussian of width 1
  }
  for(i in 1:M){
    real ds = scores[player1[i]] - scores[player2[i]]; // Just use a very steep curve (beta = 100)
    target += A12[i]*log(alpha*0.5+(1-alpha)*1/(1+e()^(-100*ds))) + A21[i]*log(alpha*0.5+(1-alpha)*1/(1+e()^(100*ds)));
  }
}
"""

bradley_terry_eta_model = """
data {
  int<lower = 0> N;                     // players
  int<lower = 0> M;                     // matches
  array[M] int<lower=1, upper = N> player1;   // player 1 for interaction n
  array[M] int<lower=1, upper = N> player2;   // player 2 for interation n
  array[M] real<lower = 0> A12;                // Number of wins by player1 against player2
  array[M] real<lower = 0> A21;                // Number of wins by player2 against player1
}
parameters {
  vector[N] scores;                      // ability for player n
}
model {
  for( k in 1:N){ // P(s_i)
    target += log(e()^(-scores[k])/(1+e()^(-scores[k]))^2); // Logistic prior
  }
  for(i in 1:M){
    real ds = scores[player1[i]] - scores[player2[i]];
    target += A12[i]*log(1/(1+e()^(-ds))) + A21[i]*log(1/(1+e()^(ds)));
  }
}
"""

def write_to_file(filename,alphas,betas,scores):
  # Write the result to file
  f_out = open(filename,"w")
  header_row = "alpha,beta,"
  for i in range(n):
    header_row += f"scores_{i},"
  header_row = header_row[:-1] + "\n"# Remove last comma
  f_out.write(header_row)
  for t in range(len(scores)):
    row_string = f"{alphas[0][t]},{betas[0][t]},"
    for i in range(n):
      row_string += f"{scores[t][i]},"
    row_string = row_string[:-1] + "\n"
    f_out.write(row_string)

data_filename = sys.argv[1]
data_name = data_filename.split('/')[-1][:-4]

model_dict = {"alpha_beta":alpha_beta_model,"bradley_terry":bradley_terry_model,"step_function":step_function_model,"bradley_terry_eta":bradley_terry_eta_model} # names of STAN models and their strings

np.set_printoptions(suppress=True) # supress scientific notation

print(f"filename: {data_filename}, num_chains: {num_chains}")

player1, player2, A12, A21, labels = util.read_filename(data_filename)

# Computing basic graph data
m = len(player1)
n = len(labels)
m_a = np.sum(A12) + np.sum(A21) # Total number of matches
print(f"n: {n}, m: {m}, m_a: {m_a}")


if not crossvalidation:
  for model_name, model_string in model_dict.items():
    # if model_name == "alpha_beta":
    # if model_name != "alpha_beta":
    if True:
      data = {"N": int(n),
              "M": int(m),
              "player1":player1,
              "player2":player2,
              "A12":A12,
              "A21":A21} 
      posterior = stan.build(model_string, data=data, random_seed=1)
      fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
      scores = fit["scores"].T
      total_samples = len(scores)
      if model_name == "alpha_beta":
        alphas = fit["alpha"]
        betas = fit["beta"]
      if model_name == "bradley_terry":
        alphas = np.zeros((1,total_samples))
        betas = fit["beta"]
      if model_name == "step_function":
        alphas = fit["alpha"]
        betas = 100*np.ones((1,total_samples)) # Step function beta
      if model_name == "bradley_terry_eta":
        alphas = np.zeros((1,total_samples))
        betas = np.ones((1,total_samples)) # Typical sigmoid

      write_to_file(f"{full_samples_folder}/{data_name}_{model_name}.csv",alphas,betas,scores)
else: # Perform crossvalidation splits
  for t in range(num_crossvalidations): # Used to set the seed
  # for t in range(1,num_crossvalidations): # Used to set the seed, skipping first to check the chess
    for split in range(5): # 5-fold crossvalidations
      for model_name, model_string in model_dict.items():
        # Check if the file has not already been generated
        if f"{data_name}_{model_name}_{t}_{split}.csv" not in os.listdir(crossvalidation_samples_folder):
          player1, player2, A12, A21, labels = util.read_filename(data_filename,split=split,train_test="train",seed=t)
          m = len(player1)
          n = len(labels)
          data = {"N": int(n),
              "M": int(m),
              "player1":player1,
              "player2":player2,
              "A12":A12,
              "A21":A21} 
          posterior = stan.build(model_string, data=data, random_seed=1)
          fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
          scores = fit["scores"].T
          total_samples = len(scores)
          if model_name == "alpha_beta":
            alphas = fit["alpha"]
            betas = fit["beta"]
          if model_name == "bradley_terry":
            alphas = np.zeros((1,total_samples))
            betas = fit["beta"]
          if model_name == "step_function":
            alphas = fit["alpha"]
            betas = 100*np.ones((1,total_samples)) # Step function beta
          if model_name == "bradley_terry_eta":
            alphas = np.zeros((1,total_samples))
            betas = np.ones((1,total_samples)) # Typical sigmoid

          write_to_file(f"{crossvalidation_samples_folder}/{data_name}_{model_name}_{t}_{split}.csv",alphas,betas,scores)
        else:
          print(f"File {data_name}_{model_name}_{t}_{split}.csv already found in {crossvalidation_samples_folder}")
