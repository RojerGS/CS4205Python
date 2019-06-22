# import statements
import numpy as np
from collections import OrderedDict

class IndexMapping(object):
    """
    An object which stores the mapping of genotype indices to subgenotype indices.
    """
    def __init__(self, input_from, train_from):
        """
        Initialize the IndexMapping by mapping the indices from the genotype
        to the indices of the subgenotype

        Args:
            input_from (list-like): a list of GLOBAL indices that are seen by the function
            train_from (list-like): a list of GLOBAL indices of the genotype which are
            being modified by the genetic algorithm.
        """
        """ Validate arguments """
        if not (set(train_from).issubset(set(input_from))):
            print(train_from)
            print(input_from)
            raise Exception("The train indices are not a subset of the input indices!")

        ### Input mapping is a dictionary that maps the input global indices into
        ## the locally constructed input
        self._input_mapping = OrderedDict(zip(input_from, range(len(input_from))))
        ### Train mapping is a dictionary that maps the species subgenotype into
        ## the locally constructed input
        self._train_mapping = OrderedDict([(idx, self._input_mapping[train_from[idx]])
                                for idx in range(len(train_from))])
        ### Lift mapping lifts the local species subgenotype into the global indices
        self._lift_mapping = OrderedDict([(idx, train_from[idx]) for idx in range(len(train_from))])

    def get_input_mapping(self):
        """
        Return: The dictionary which maps which indices of the whole genotype go into
        building the input for the function
        """
        return self._input_mapping

    def get_train_mapping(self):
        """
        Return: The dictionary that maps the indices of the local subgenotype into
        the indices of the genotype that should go into the function
        """
        return self._train_mapping

    def get_lift_mapping(self):
        """
        Return: the dictionary that maps the indices of the local genotype
        into the global indices
        """
        return self._lift_mapping


"""
Helper functions which abstract the construction of genotypes from a subspace
into a more general space.
"""
def extrapolate_values(subgenotype, genotype, index_mapping):
    """
    Returns a vector containing the values of a subgenotype as well as other values
    in the genotype needed for function evaluation.
    Args:
        subgenotype (list-like): a subgenotype which contains values used for training.
        genotype (list-like): a complete genotype from which we will pull other values.
        value_indices (list-like): a set of indices which describe which values we want.
    """

    """ Modify the values of subgenotype """
    ret = np.array(genotype)
    lift_mapping = index_mapping.get_lift_mapping()
    ret[list(lift_mapping.values())] = subgenotype

    return ret

def wrap_function(f, inps):
    """
    Returns a function g such that
    g(L) = f(L[inps])
        i.e. creates an intermediate function that receives a more general
        input but then only uses a subset of said input to compute f
    For indexing matters, assume L is of type np.ndarray
    """
    return lambda L: f(L[inps])

if __name__ == "__main__":
    input_space = [1, 3, 4, 7, 8]
    local_space = [3, 7]
    im = IndexMapping(input_space, local_space)

    local_values = ["a", "b"]
    global_values = [10,11,12,13,14,15,16,17,18,19,20]
    ret = extrapolate_values(local_values, global_values, im)
    # this should give [11, "a", 14, "b", 18]
    print(ret)
    assert len(ret) == len(input_space)
    for idx, val in enumerate(ret):
        if input_space[idx] in local_space:
            assert val == local_values[local_space.index(input_space[idx])]
        else:
            assert val == global_values[input_space[idx]]