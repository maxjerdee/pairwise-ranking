# Parse the various supported file formats into a sparse adjecency matrix and list of labels
# TODO: Port to c++ to read large files faster
from scipy.sparse import lil_matrix
import networkx as nx


def read_match_list_from_match_list(filename):
    """Read list of matches from a file with each line in "winner loser" format.

    Args:
        filename (string): Input filename.

    Raises:
        AssertionError: Each line must be in "winner loser" format.

    Returns:
        list: List of matches, each represented by a dict of the winner and loser.
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
    """Read list of matches as the edges in a gml file.

    Args:
        filename (string): Input filename.

    Returns:
        list: List of matches, each represented by a dict of the winner and loser.
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
    pass


def read_match_list(filename):
    pass
