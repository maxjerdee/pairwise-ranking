# Parse the various supported file formats into a sparse adjecency matrix and list of labels
from scipy.sparse import lil_matrix


# TODO: Port to c++ to read large files faster
def read_match_list(filename):    
    """Read a file with a list of individual matches in "winner loser" format on each line.

    Args:
        filename (string): input filename

    Raises:
        AssertionError: Each line must be in "winner loser" format.

    Returns:
        list: List of matches, each represented by a dict of the winner and loser 
    """
    
    # Store the tuples of winner and loser indices since we first need to find the number of players
    match_list = []

    f = open(filename, "r")
    for line in f.readlines():
        line_split = line.split()
        if len(line_split) != 2:
            raise AssertionError(
                f'"{line.strip()}" does not have 2 space-separated entries. \
Format as "winner loser" (make sure the names don\'t have spaces)'
            )
        
        match_list.append({"winner":line_split[0], "loser":line_split[1]})

    return match_list
