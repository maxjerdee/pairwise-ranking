# The script `tools.py` contains functions which a variety of the scripts take advantage of, 
# including the ability to read our data format into a sparse format easily used by STAN. 

import numpy as np
from scipy.special import loggamma
import matplotlib.pyplot as plt

# Read a file, format of each line is 
# winner_string loser_string type increment (space separated)
# where type = 0 is a win, type = 1 is a tie (counted as half win for each player)
# increment is the number of matches won (requires type = 0)
# typically will just provide the winner_string and loser_string.
# train_test can equal None if want to return the full dataset, ="train" if the training portion, ="test" if testing
# seed determines the split by seeding the randomness
# split is  anumber 0 to 4 which decides which of the splits to consider as the test dataset
def read_filename(filename,train_test=None,seed=0,split=0):
    """Read a filename containing a list of matches

    Args:
        filename (str): filename of the input file
        train_test (_type_, optional): _description_. Defaults to None.
        seed (int, optional): _description_. Defaults to 0.
        split (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """    
    # Potentially decide on which lines to consider if there is a 
    k_fold = 5 # 5-fold crossvalidation
    np.random.seed(seed)
    lines = []
    f = open(filename,"r")
    for line in f.readlines():
      lines.append(line)
    m_a = len(lines)
    indices = np.array(range(m_a))
    np.random.shuffle(indices)
    indices_split = np.array_split(indices, k_fold) 
    assert split >= 0 and split < k_fold
    test_indices = indices_split[split]
    train_indices = [index for index in range(m_a) if index not in test_indices]
    if train_test == None:
      use_indices = range(m_a)
    elif train_test == "train":
      use_indices = train_indices
    elif train_test == "test":
      use_indices = test_indices
    else:
      print("invalid train_test", train_test)
    
    labels_dict = {} # Dictionary of {string: index}
    player12_dict = {} # Dictionary of {(winner_index,loser_index): pair_index} where (winner_index,loser_index) is sorted
    A12 = []
    A21 = []
    num_lines = 0
    player_index = 1 # Start at 1 since STAN is unary
    pair_index = 0
    for line_index in range(len(lines)):
        line = lines[line_index]
        num_lines += 1
        if num_lines % 100000 == 0 and verbose:
           print(f"{num_lines} lines read")
        # Convert to label indexes
        pieces = line.split()
        winner = pieces[0]
        loser = pieces[1]
        increment = 1 # Amount to count a win for (default of 1, but could be more if file says so)
        if len(pieces) > 2:
          interaction_type = int(pieces[2]) # 0 means a win, 1 means a tie
        if len(pieces) > 3:
           increment = float(pieces[3])
        else:
          interaction_type = 0 # Assume a win report without the third column
        # Always fill out the label dictionary in the same way so that any training/testing crossvalidation split agrees on the identities of the players
        if winner not in labels_dict.keys():
            labels_dict.update({winner:player_index})
            player_index += 1
        if loser not in labels_dict.keys():
            labels_dict.update({loser:player_index})
            player_index += 1
          
        if line_index in use_indices:
          winner_ind = labels_dict[winner]
          loser_ind = labels_dict[loser]
          sorted_index_pair = (max(winner_ind,loser_ind),min(winner_ind,loser_ind)) # Tuple so hashable
          
          if sorted_index_pair not in player12_dict.keys():
            player12_dict.update({sorted_index_pair:pair_index}) # Create new assignment of index to pair
            pair_index += 1
            interaction_index = -1
            A12.append(0)
            A21.append(0)
          else:
            interaction_index = player12_dict[sorted_index_pair]
          if interaction_type == 1: # Tie
            A12[interaction_index] += 0.5
            A21[interaction_index] += 0.5
          elif interaction_type == 0: # Win
            if winner_ind > loser_ind: # player1 beat player2
              A12[interaction_index] += increment
            else: # player2 beat player1
              A21[interaction_index] += increment
          else:
            print(f"Bad interaction {interaction_index}")
    
    player1 = []
    player2 = []
    for player_pair in player12_dict.keys():
        player1.append(player_pair[0])
        player2.append(player_pair[1])
        
    labels = list(labels_dict.keys())
    return player1, player2, A12, A21, labels

# Define theta = arctan(beta/4). The betas at which we evaluate the posterior are uniform over these
# Define the parameter map
def beta_to_theta(beta):
    return np.arctan(beta/4)

def theta_to_beta(theta):
    return 4*np.tan(theta)


# log(f(s,alpha,beta)) = log(0.5*alpha + (1-alpha)*1/(1+E^(-beta s))), protected from underflows and overflows
def log_f(s,alpha,beta):
    m = -beta*s # f(s,d,u) = 0.5*u + (1-u)*1/(1+E^m)
    mp = np.clip(m, 0,None) # max(m,0) >= 0
    mm = np.clip(-m, 0,None) # max(-m,0) >= 0
    # m = mp - mm
    res = np.log(np.exp(-mp)*(1-alpha) + (np.exp(-mp)+np.exp(-mm))*alpha/2) - np.log(np.exp(-mp)+np.exp(-mm))
    return res 

# Cauchy prior of width 4, uniform over the angle theta = arctan(beta/4)
def logPbeta(beta):
    return np.log(1/(4*(1 + beta**2/16))) # This is just to return a number or array of the right shape

# Gaussian
def logPscores(scores):
    return -np.sum(scores**2 - 0.5*np.log(np.pi)) # This is just to return a number or array of the right shape

# Logistic
def logPscoresLogistic(scores):
  return np.sum(-scores-2*np.log(1+np.exp(-scores))) # log(e()^(-scores[k])/(1+e()^(-scores[k]))^2); // Logistic prior
  # = -s - 2(logsumexp(0,-s))

# log(sum(exp(array))), but protected from overflows
def logSumExp(array):
    xm = np.max(array)
    return xm + np.log(np.sum(np.exp(array - xm)))

# log of the binomial coefficient
def logBinom(a,b):
  return loggamma(a + 1) - loggamma(b + 1) - loggamma(a-b+1)


# Compute the per-match log likelihood probability of observing the given data
def compute_logP(scores, player1, player2, A12np, A21np, alpha, beta):
  score_diffs = []
  for i in range(len(player1)): # Iterate through the edges
      score_diffs.append(scores[player1[i] - 1]-scores[player2[i] - 1]) # convert back from unary stan input
  score_diffs = np.array(score_diffs)
  total_matches = np.sum(A12np) + np.sum(A21np)
  # return np.sum(logBinom(A12np+A21np,A12np) + A12np*log_f(score_diffs,alpha,beta) + A21np*log_f(-score_diffs,alpha,beta),axis=0)/total_matches
  # return np.sum(A12np*log_f(score_diffs,alpha,beta) + A21np*log_f(-score_diffs,alpha,beta),axis=0)/total_matches
  # Base 2 logarithm:
  return np.sum(A12np*log_f(score_diffs,alpha,beta) + A21np*log_f(-score_diffs,alpha,beta),axis=0)/(total_matches*np.log(2))

# Compute the accuracy of the direction of the scores
def compute_accuracy(scores, player1, player2, A12np, A21np):
  total_matches = np.sum(A12np) + np.sum(A21np)
  total_correct = 0
  for i in range(len(player1)): # Iterate through the edges
    if scores[player1[i] - 1] > scores[player2[i] - 1]: # Going to assume that noone has the same scores
      total_correct += A12np[i] # Player 1 supposed to win
    else:
      total_correct += A21np[i] # Player 2 supposed to win
  return total_correct/total_matches


# Which datasets to include, and how to label them
name_dict = {"ssbm":["Video games",0],"Franz_2015d":["Baboons",0],"Silk_2019a":["Dogs",2]\
             ,"Vilette_2020":["Vervet monkeys",3],\
             "Watt_1986f":["Sparrows",1],"Williamson_2016k":["Mice",0],\
                "atp_matches_agg":["Tennis",3],"basketball_all":["Basketball",2],\
                "friends_16":["Friends",4],"Strauss_2019c":["Hyenas",4],\
                    "cs_faculty":["CS depts.",3],"history_faculty":["History depts.",2],\
                    "soccer_agg_no_ties":["Soccer",1],"chess_tiny_no_ties":["Chess",4],"scrabble_sparse_old":["Scrabble",0]}

model_name_dict = {"step_function":"Luck only (min. violations)","springrank_unregularized":"SpringRank","springrank_regularized":"Regularized SpringRank","bradley_terry_ML":"Maximum likelihood","bradley_terry_eta":"Logistic prior","bradley_terry":"Depth only","alpha_beta":"This paper"}

# 0 - blue
# 1 - orange
# 2 - green
# 3 - red
# 4 - purple

# List of colors to use
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_cycle_length = 5
colors_RGB = []
for color in colors:
    colors_RGB.append(tuple(float(int(color[i+1:i+3], 16)/256) for i in (0, 2, 4)))
