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

        print("-"*30)
        print("global genotype indices of function input {}".format(input_from))
        print("global genotype indices of species variables {}".format(train_from))
        ### Input mapping is a dictionary that maps the input global indices into
        ## the locally constructed input
        self._input_mapping = OrderedDict(zip(input_from, range(len(input_from))))
        ### Train mapping is a dictionary that maps the species subgenotype into
        ## the locally constructed input
        self._train_mapping = OrderedDict([(idx, self._input_mapping[train_from[idx]])
                                for idx in range(len(train_from))])
        ### Lift mapping lifts the local species subgenotype into the global indices
        self._lift_mapping = OrderedDict([(idx, train_from[idx]) for idx in range(len(train_from))])
        print("input mapping")
        print(self._input_mapping)
        print("train mapping")
        print(self._train_mapping)
        print("lift mapping")
        print(self._lift_mapping)
        print("-"*30)

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
    ret = np.zeros(len(index_mapping.get_input_mapping()), dtype=np.double)

    input_mapping = index_mapping.get_input_mapping()
    for i in input_mapping:
        ret[input_mapping[i]] = genotype[i]

    train_mapping = index_mapping.get_train_mapping()
    for j in train_mapping:
        ret[train_mapping[j]] = subgenotype[j]

    return ret

# def extract_values(genotype, index_mapping):
#     """
#     Collects the values of a genotype into a subgenotype, as described by the index_mapping.

#     Returns:
#         list-like: the subgenotype
#     """
#     input_mapping = index_mapping.get_input_mapping()
#     subgenotype = [None]*len(input_mapping)
#     for i in input_mapping:
#         subgenotype[input_mapping[i]] = genotype[i]

#     return subgenotype

# def aggregate_values(subgenotypes, index_mappings):
#     """
#     Takes a list of subgenotypes as well as the indices which each genotype contributes
#     to the output, and produces a genotype that encompasses all genotypes.

#     Args:
#         subgenotypes (list-like): a list of subgenotypes to extract values from.
#         index_partitions (list-like): a list of index mappings which store the
#         respective subgenotypes' value locations in the genotype.

#     Returns:
#         numpy array: a complete genotype pulling values from all subgenotypes.
#     """
#     """ Validate Input"""
#     #verify that there is an index_mapping for each subgenotype
#     if len(subgenotypes) != len(index_mappings):
#         raise Exception("The inputs aren't of matching sizes!")

#     """ Collect values from the index mappings """
#     length = len([len(im.get_train_mapping()) for im in index_mappings])
#     aggregation = [None]*length

#     for (subgenotype, mapping) in zip(subgenotypes, index_mappings):
#         train_mapping = mapping.get_train_mapping()
#         for k in train_mapping:
#             aggregation[train_mapping[k]] = subgenotype[k]

#     """ Return the aggregated genotype """
#     # first, make sure that the partitions covered the whole genotype
#     if None in aggregation:
#         raise Exception("Not the whole genotype was accounted for by the train partitions!")

#     return np.array(aggregation)

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