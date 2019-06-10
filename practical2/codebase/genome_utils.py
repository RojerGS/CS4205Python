# import statements
import numpy as np

class IndexMapping(object):
    """
    An object which stores the mapping of genotype indices to subgenotype indices.
    """
    def __init__(self, input_from, train_from, input_to):
        """
        Initialize the IndexMapping by mapping the indices from the genotype
        to the indices of the subgenotype

        Args:
            input_from (list-like): a list of indices that are seen in the genotype.
            train_from (list-like): a list of indices of the genotype which are
            being modified by the genetic algorithm.
            input_to (list-like): the set of indices to which the values from the genotype correspond.
        """
        """ Validate arguments """
        if (len(input_from) != len(input_to)):
            raise Exception("The sizes of the inputs are uneven!")
        if not (set(train_from).issubset(set(input_from))):
            raise Exception("The train indices are not a subset of the input indices!")

        self._mapping = dict(zip(input_from, input_to))
        self._train_mapping = dict(zip(train_from, range(len(train_from))))
        # self._train_mapping = {tf: self._mapping[tf] for tf in train_from}

    def get_input_mapping(self):
        """
        Return: The dictionary which maps genotype indicies to subgenotype indices.
        """
        return self._mapping

    def get_train_mapping(self):
        """
        Return: The dictionary which maps genotype indices to subgenotype indicies,
        specifying only the indices which are being used for training.
        """
        return self._train_mapping

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
    ret = np.array([None]*len(index_mapping.get_input_mapping()))

    input_mapping = index_mapping.get_input_mapping()
    for i in input_mapping:
        ret[input_mapping[i]] = genotype[i]

    train_mapping = index_mapping.get_train_mapping()
    for j in train_mapping:
        ret[train_mapping[j]] = subgenotype[train_mapping[j]]

    return ret


def extract_values(genotype, index_mapping):
    """
    Collects the values of a genotype into a subgenotype, as described by the index_mapping.

    Returns:
        list-like: the subgenotype
    """
    input_mapping = index_mapping.get_input_mapping()
    subgenotype = [None]*len(input_mapping)
    for i in input_mapping:
        subgenotype[input_mapping[i]] = genotype[i]

    return subgenotype

def aggregate_values(subgenotypes, index_mappings):
    """
    Takes a list of subgenotypes as well as the indices which each genotype contributes
    to the output, and produces a genotype that encompasses all genotypes.

    Args:
        subgenotypes (list-like): a list of subgenotypes to extract values from.
        index_partitions (list-like): a list of index mappings which store the
        respective subgenotypes' value locations in the genotype.

    Returns:
        numpy array: a complete genotype pulling values from all subgenotypes.
    """
    """ Validate Input"""
    #verify that there is an index_mapping for each subgenotype
    if len(subgenotypes) != len(index_mappings):
        raise Exception("The inputs aren't of matching sizes!")

    """ Collect values from the index mappings """
    length = len([len(im.get_train_mapping()) for im in index_mappings])
    aggregation = [None]*length

    for (subgenotype, mapping) in zip(subgenotypes, index_mappings):
        train_mapping = mapping.get_train_mapping()
        for k in train_mapping:
            aggregation[train_mapping[k]] = subgenotype[k]

    """ Return the aggregated genotype """
    # first, make sure that the partitions covered the whole genotype
    if None in aggregation:
        raise Exception("Not the whole genotype was accounted for by the train partitions!")

    return np.array(aggregation)
