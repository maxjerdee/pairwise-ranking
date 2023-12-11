# Parse the various supported file formats into a sparse adjecency matrix and list of labels
# TODO: Port to c++ to read large files faster
from scipy.sparse import lil_matrix
import networkx as nx
import numpy as np
from os.path import isfile

def read_match_list_from_match_list(filename):
    """Read list of matches from a file where each line represents a match in the format "winner loser".

    :param filename: Input filename
    :type filename: str
    :raises AssertionError: If a line is not in "winner loser" format. Labels should not themselves include spaces.
    """    
    # Store the tuples of winner and loser indices since we first need to find the number of players
    match_list = []

    f = open(filename, "r")
    for line in f.readlines():
        line_split = line.split()
        if not len(line_split) == 2:
            raise AssertionError(
                f'"{line.strip()}" does not have 2 space-separated entries. \
Format as "winner loser" (make sure the names don\'t have spaces)'
            )

        match_list.append({"winner": line_split[0], "loser": line_split[1]})

    return match_list

def read_match_list_from_gml(filename):
    """Read list of matches as the edges in a gml network file.

    :param filename: Input filename
    :type filename: str
    :return: List of matches, each represented by a dict of the winner and loser. 
    :rtype: list
    """    
    # Read in with networkX
    G = nx.read_gml(filename)
    edges = list(G.edges())

    # Convert to our list of dicts format
    match_list = []
    for edge in edges:
        match_list.append({"winner": edge[0], "loser": edge[1]})

    return match_list


def read_match_list_from_adj_matrix(filename):
    """Read list of matches from an adjacency matrix which counts the number of times each player beats another. In lieu of labels, assign in the form player_i.

    :param filename: Input filename
    :type filename: str
    :return: List of matches, each represented by a dict of the winner and loser. 
    :rtype: list
    """    
    adj_matrix = np.loadtxt(filename,dtype=int)
    n = len(adj_matrix)

    match_list = []
    for i in range(n):
        for j in range(n):
            # Give each player a label from player_0 to player_(n-1) depending on where they appear in the given matrix
            player_i_label = f"player_{i}"
            player_j_label = f"player_{j}"
            for t in range(adj_matrix[i,j]): # Add the number of matches which occured
                match_list.append({"winner": player_i_label, "loser": player_j_label})

    return match_list

def read_match_list(filename):
    """Read list of matches from a file, attempting to detect the appropriate file format among gml, match_list, or adjacency_matrix formats.

    :param filename: Input filename
    :type filename: str
    :return: List of matches, each represented by a dict of the winner and loser. 
    :rtype: list
    """    
    # First check that the file exists
    f_in = open(filename)

    # Attempt to read the file in the other of: gml, match_list, adj_matrix
    try:
        return read_match_list_from_gml(filename)
    except Exception:
        pass
    try:
        return read_match_list_from_match_list(filename)
    except Exception:
        pass
    try:
        return read_match_list_from_adj_matrix(filename)
    except Exception:
        pass

    # If all methods have failed, raise error
    raise AssertionError(
        f"Unable to parse {filename} as a gml, match list, or adjacency matrix."
    )

        
