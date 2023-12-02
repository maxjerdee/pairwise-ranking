# Use pystan to take samples from the various models
import stan
import numpy as np
import util # Local import

# STAN model strings
depth_and_luck_STAN_string = """
data {
  int<lower = 0> N;                             // Number of players
  int<lower = 0> M;                             // Number of unique pairings
  array[M] int<lower=1, upper = N> player1;     // The index of "player1" in each pairing
  array[M] int<lower=1, upper = N> player2;     // The index of "player2" in each pairing
  array[M] real<lower = 0> A12;                 // Number of wins recorded by player1 against player2
  array[M] real<lower = 0> A21;                 // Number of wins recorded by player2 against player1
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
    target += A21[i]*log(alpha*0.5+(1-alpha)*1/(1+e()^(beta*ds)));  // Probability of times that player2 beat player1
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
  array[M] real<lower = 0> A21;                 // Number of wins recorded by player2 against player1
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
    target += A21[i]*log(1/(1+e()^(beta*ds)));  // Probability of times that player2 beat player1
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
  array[M] real<lower = 0> A21;                 // Number of wins recorded by player2 against player1
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
    target += A21[i]*log(alpha*0.5+(1-alpha)*1/(1+e()^(100*ds)));  // Probability of times that player2 beat player1
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
  array[M] real<lower = 0> A21;                 // Number of wins recorded by player2 against player1
}
parameters {
  vector[N] scores;                             // Score of each player
  real<lower = 0> beta;                         // Depth of competition
}
model {
  for( k in 1:N){                               // P(s_i), prior on scores
    target += log(e()^(-scores[k])/(1+e()^(-scores[k]))^2); // Logistic prior
  }
  for(i in 1:M){                                // P(A|s_i), model likelihood
    real ds = scores[player1[i]] - scores[player2[i]];  // Score difference 
    target += A12[i]*log(1/(1+e()^(-ds)));      // Probability of times that player1 beat player2
    target += A21[i]*log(1/(1+e()^(ds)));       // Probability of times that player2 beat player1
  }
}
"""

