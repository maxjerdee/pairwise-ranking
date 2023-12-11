from ranking.results import *
from ranking.parsing import read_match_list_from_adj_matrix

# Test against results obtained with Mathematica via 
# numerical optimization/integration on a small test case
# NOTE: these tests may randomly fail since they assess a MCMC process

# SCORE ESTIMATES

def test_logisticPriorMAP():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/logisticPriorMAP_1.txt","r")
    true_scores = []
    for line in f_in.readlines():
        true_scores.append(float(line))
    n = len(true_scores)

    # Call function
    scores_df = scores(match_list,force_mode='MAP')

    # Compare results
    for i in range(n):
        player_label = f'player_{i}'
        est_score = scores_df.loc[scores_df['label'] == player_label, 'score'].values[0]
        est_score_error = scores_df.loc[scores_df['label'] == player_label, 'score_error'].values[0]
        # Checking that the estimated error is realistic for each score
        if np.abs(true_scores[i]-est_score) > deviations_threshold*est_score_error:
            print(f"Failed on {i}")
            print("True scores:")
            print(true_scores)
            print("Results:")
            print(scores_df)
            assert False

    # All good
    assert True

def test_logisticPriorEs():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/logisticPriorEs_1.txt","r")
    true_scores = []
    for line in f_in.readlines():
        true_scores.append(float(line))
    n = len(true_scores)

    # Call function
    scores_df = scores(match_list,force_mode='average',model_name='logistic_prior')

    # Compare results
    for i in range(n):
        player_label = f'player_{i}'
        est_score = scores_df.loc[scores_df['label'] == player_label, 'score'].values[0]
        est_score_error = scores_df.loc[scores_df['label'] == player_label, 'score_error'].values[0]
        # Checking that the estimated error is realistic for each score
        if np.abs(true_scores[i]-est_score) > deviations_threshold*est_score_error:
            print(f"Failed on {i}")
            print("True scores:")
            print(true_scores)
            print("Results:")
            print(scores_df)
            assert False

    # All good
    assert True

def test_depthOnlyEs():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/depthOnlyEs_1.txt","r")
    true_scores = []
    for line in f_in.readlines():
        true_scores.append(float(line))
    n = len(true_scores)

    # Call function
    scores_df = scores(match_list,force_mode='average',model_name='depth_only')

    # Compare results
    for i in range(n):
        player_label = f'player_{i}'
        est_score = scores_df.loc[scores_df['label'] == player_label, 'score'].values[0]
        est_score_error = scores_df.loc[scores_df['label'] == player_label, 'score_error'].values[0]
        # Checking that the estimated error is realistic for each score
        if np.abs(true_scores[i]-est_score) > deviations_threshold*est_score_error:
            print(f"Failed on {i}")
            print("True scores:")
            print(true_scores)
            print("Results:")
            print(scores_df)
            assert False

    # All good
    assert True

def test_luckOnlyEs():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/luckOnlyEs_1.txt","r")
    true_scores = []
    for line in f_in.readlines():
        true_scores.append(float(line))
    n = len(true_scores)

    # Call function
    scores_df = scores(match_list,force_mode='average',model_name='luck_only')

    # Compare results
    for i in range(n):
        player_label = f'player_{i}'
        est_score = scores_df.loc[scores_df['label'] == player_label, 'score'].values[0]
        est_score_error = scores_df.loc[scores_df['label'] == player_label, 'score_error'].values[0]
        # Checking that the estimated error is realistic for each score
        if np.abs(true_scores[i]-est_score) > deviations_threshold*est_score_error:
            print(f"Failed on {i}")
            print("True scores:")
            print(true_scores)
            print("Results:")
            print(scores_df)
            assert False

    # All good
    assert True

def test_depthAndLuckEs():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/depthAndLuckEs_1.txt","r")
    true_scores = []
    for line in f_in.readlines():
        true_scores.append(float(line))
    n = len(true_scores)

    # Call function
    scores_df = scores(match_list,force_mode='average',model_name='depth_and_luck')

    # Compare results
    for i in range(n):
        player_label = f'player_{i}'
        est_score = scores_df.loc[scores_df['label'] == player_label, 'score'].values[0]
        est_score_error = scores_df.loc[scores_df['label'] == player_label, 'score_error'].values[0]
        # Checking that the estimated error is realistic for each score
        if np.abs(true_scores[i]-est_score) > deviations_threshold*est_score_error:
            print(f"Failed on {i}")
            print("True scores:")
            print(true_scores)
            print("Results:")
            print(scores_df)
            assert False

    # All good
    assert True

# PROBABILITY ESTIMATES

def test_logisticPriorMAP_probability():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/logisticPriorP_1.txt","r")
    line = f_in.readline()
    true_prob = float(line)

    # Call function
    prob, prob_error = probability(match_list,'player_0','player_1',force_mode='MAP')

    # Compare results
    if np.abs(true_prob-prob) > deviations_threshold*prob_error:
        assert False

    # All good
    assert True

def test_logisticPrior_probability():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/logisticPriorEP_1.txt","r")
    line = f_in.readline()
    true_prob = float(line)

    # Call function
    prob, prob_error = probability(match_list,'player_0','player_1',force_mode='average',model_name='logistic_prior')

    # Compare results
    if np.abs(true_prob-prob) > deviations_threshold*prob_error:
        assert False

    # All good
    assert True

def test_depthOnly_probability():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/depthOnlyEP_1.txt","r")
    line = f_in.readline()
    true_prob = float(line)

    # Call function
    prob, prob_error = probability(match_list,'player_0','player_1',force_mode='average',model_name='depth_only')

    # Compare results
    if np.abs(true_prob-prob) > deviations_threshold*prob_error:
        assert False

    # All good
    assert True

def test_luckOnly_probability():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/luckOnlyEP_1.txt","r")
    line = f_in.readline()
    true_prob = float(line)

    # Call function
    prob, prob_error = probability(match_list,'player_0','player_1',force_mode='average',model_name='luck_only')

    # Compare results
    if np.abs(true_prob-prob) > deviations_threshold*prob_error:
        assert False

    # All good
    assert True

def test_depthAndLuckOnly_probability():
    # How many times away from the claimed error we will allow
    deviations_threshold = 5 

    # Load test information
    match_list = read_match_list_from_adj_matrix("tests/test_case_files/adj_matrix_1.txt")
    f_in = open("tests/test_case_files/depthAndLuckEP_1.txt","r")
    line = f_in.readline()
    true_prob = float(line)

    # Call function
    prob, prob_error = probability(match_list,'player_0','player_1',force_mode='average',model_name='depth_and_luck')

    # Compare results
    if np.abs(true_prob-prob) > deviations_threshold*prob_error:
        assert False

    # All good
    assert True