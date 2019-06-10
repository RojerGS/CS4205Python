"""Implements gray- and black-box cooperative coevolution,
based on the implementations of the real-valued optimizations algorithms provided.
In black-box cooperative coevolution, we want to minimize a function f,
where f is a function of n variables, but we know that it makes sense to divide
the problem into several populations, each concerned with
????????????????????"""

import numpy as np
from copy import deepcopy as dc
from genome_utils import *

class GrayBoxOptimizer(object):
    """
    An object which trains a genotype on optimization of a set of
    hand-partitioned functions, as evaluated on a subset of input variables.
    Functions are assumed to be real-valued with the goal of minimization.
    """

    class Species(object):
        """
        An object to abstract the necessary collection of features and such
        from species which are being trained in a GBO.
        """
        def __init__(self, function, input_space, train_part,
                     genetic_algorithm, genetic_algorithm_arguments,
                     initial_genotype):
            """
            Construct a species upon the fitness function, GA, and input space.

            Args:
                function (object): a function used to evaluate the fitness
                of an individual in the species.
                input_space (list): list of indices which tells you what parts
                of the genotype are used in the function evaluation.
                train_part (list): list of indices which are modified
                by the GA that we're using for optimization.
                genetic_algorithm (object): The type of object used to abstract
                the training of our individuals.
                genetic_algorithm_arguments (dict): a dictionary listing
                the additional arguments we might need in the genetic algorithm.
            """
            self._function = function
            self._input_space = input_space
            self._index_mapping = IndexMapping(input_from = input_space,
                                              train_from = train_part,
                                              input_to = list(range(len(input_space))))

            self._optimizer = genetic_algorithm(fitness_function = self._function,
                                                genome_length = len(train_part),
                                                initial_genotype = initial_genotype,
                                                index_mapping = self._index_mapping,
                                                **genetic_algorithm_arguments)


    def __init__(self, functions, input_spaces, train_partition,
                 genetic_algorithms, genetic_algorithm_arguments,
                 genome_length = None,
                 max_generations = float('inf'),
                 max_evaluations = float('inf'),
                 goal_fitness = float('-inf')):
        """
        Initialize the Gray Box Optimizer. Initializes all GAs with their
        respective partitions upon the genotype.

        Args:
            functions (list-like): a list of functions which will be used
            for evaluations of parts of the genotype.
            inputs (list-like): a list of sets of indices which will be used
            as inputs for the listed function or functions.
            train_partition (list-like): a partition of indices on the genotype
            input space which will decide which genetic algorithms will train
            which variables.
            genetic_algorithms (list-like): a list of objects containing
            genetic_algorithms used for training the partition of input variables.
            genetic_algorithm_arguments (list-like): a list of dictionaries
            detailing the sets of parameters used in the construction of the
            used genetic algorithms.
        """

        """ Validate arguments """
        # make sure inputs have the same sizes
        if not (len(input_spaces) == len(functions) == len(train_partition) \
                == len(genetic_algorithms) == len(genetic_algorithm_arguments)):
            raise Exception("The arguments of the Gray Box Optimizer did not have the same sizes!")

        # check that the variable partitions don't actually overlap, but they cover everything
        indices = [i for part in train_partition for i in part]
        if genome_length == None: genome_length = max(indices)+1
        if genome_length != max(indices)+1 or set(indices) != set(range(genome_length)):
            raise Exception("The provided index partitions do not align with the input space!")

        # make sure that the indices are a partition
        if len(indices) != len(set(indices)):
            raise Exception("The index partitions overlap!")

        """ Initializing subpopulations and member variables """
        initial_genotype = np.random.rand(genome_length)

        self._subpopulations = []
        for i in range(len(train_partition)):
            species = GrayBoxOptimizer.Species(function = functions[i],
                                               input_space = input_spaces[i],
                                               train_part = train_partition[i],
                                               genetic_algorithm = genetic_algorithms[i],
                                               genetic_algorithm_arguments = genetic_algorithm_arguments[i],
                                               initial_genotype = initial_genotype)
            self._subpopulations.append(species)

        self._generations = 0
        self._elite_fitness = float('inf')
        self._elite_genotype = None
        self._genome_length = genome_length

        self._max_generations = max_generations
        self._goal_fitness = goal_fitness

    def get_aggregate_genotype(self):
        """
        Collect the parts of the genotype from the respective subpopulations
        which are training the respective values.

        Returns:
            list-like: the genotype
        """
        genotype = [None]*self._genome_length
        for subpopulation in self._subpopulations:
            elite_genotype = subpopulation._optimizer.get_best_genotype(n=1)
            train_mapping = subpopulation._index_mapping.get_train_mapping()
            for i in train_mapping:
                genotype[i] = elite_genotype[train_mapping[i]]

        return genotype

    def evaluate(self, genotype):
        """
        Evaluate the fitness of an individual as the summation of the evaluations
        of all functions being optimized in this optimizer. Updates the elite and
        elite fitness accordingly.
        """
        subgenotypes = [extract_values(genotype, subpopulation._index_mapping)\
                        for subpopulation in self._subpopulations]
        fitness = 0
        for i in range(len(self._subpopulations)):
            fitness += self._subpopulations[i]._function(subgenotypes[i])

        if fitness < self._elite_fitness:
            self._elite_finess = fitness
            self._elite_genotype = dc(genotype)

    def evolve(self):
        """
        Evolve each subpopulation for one iteration
        """
        # first, get the aggregated genotype representing the elites of all subpopulations
        genotype = [None]*self._genome_length
        for subpopulation in self._subpopulations:
            elite_genotype = subpopulation._optimizer.get_best_genotype(n=1)[0]
            train_mapping = subpopulation._index_mapping.get_train_mapping()
            for i in train_mapping:
                genotype[i] = elite_genotype[train_mapping[i]]

        # evaluate current state of our genotype
        self.evaluate(genotype)

        # now project all subgenotypes onto the current optimal genotype where applicable
        for subpopulation in self._subpopulations:
            subpopulation._optimizer.evolve(genotype, subpopulation._index_mapping)

        self._generations = sum(map(lambda x: x._optimizer._generations, self._subpopulations))

    def has_converged(self):
        """
        Returns:
            boolean: whether or not our optimizer has converged.
        """
        if self._elite_genotype == None: return False
        return (self._generations >= self._max_generations\
                or self.get_elite_fitness() <= self._goal_fitness)

    def get_elite_genotype(self):
        """
        Returns:
            list-like: an array representing the genotype of the elite
        """
        return self._elite_genotype

    def get_elite_fitness(self):
        """
        Returns:
            float: the fitness of our elite individual
        """
        return self._elite_finess

if __name__ == "__main__":
    from fitness_functions import FunctionFactory as FF
    from particle_swarm_optimization import ParticleSwarmOptimization as PSO
    f = lambda x: sum((np.array(x)-2)**6)
    gbo = GrayBoxOptimizer(functions = [f, f, f],
                           input_spaces = [[0,1,2],[1,2,3],[2,3,4]],
                           train_partition = [[0,1],[2,3],[4]],
                           genetic_algorithms = [PSO, PSO, PSO],
                           genetic_algorithm_arguments = [{}]*3,
                           max_generations = 1000)

    while not (gbo.has_converged()):
        gbo.evolve()
        print(gbo.get_elite_fitness())
        print(gbo.get_elite_genotype())
