# import statements
import numpy as np
from abc import ABC, abstractmethod
from genome_utils import *

class GeneticAlgorithm(ABC):
    """
    Abstract Base Class which defines components necessary
    to the definition of a genetic algorithm
    """
    def __init__(self, fitness_function, genome_length, population_size,
                 lower_bounds = -1, upper_bounds = 1,
                 initial_genotype = None, index_mapping = None):
        """
        initialize the properties of a genetic algorithm.
        Make sure to assess the fitness of the initial population upon construction.

        Args:
            fitness_function (function): the function being optimized by this GA.
            genome_length (int): the length of the vector of variables being trained
            by this GA.
            population_size (int): number of members of the population of the GA.
            lower_bounds (int, list-like): the lower bound of all or each dimension
            in the training space.
            upper_bounds (int, list-like): the upper bound of all or each dimension
            in the training space.
            initial_genotype (list-like): the genotype from which extra values are pulled
            for the first evaluation of the fitness function.
            index_mapping (IndexMapping): object storing the indexing from the space of the
            genotypes in a Cooperative Coevolution to the train space.
        """
        self._fitness_function = fitness_function
        self._genome_length = genome_length
        self._population_size = population_size
        self._generations = 0
        self._evaluations = 0

        # if the given bounds are constants, turn them into vectors
        if not hasattr(lower_bounds, "__iter__"):
            lower_bounds = lower_bounds*np.ones(genome_length, dtype=np.double)
        if not hasattr(upper_bounds, "__iter__"):
            upper_bounds = upper_bounds*np.ones(genome_length, dtype=np.double)

        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    @abstractmethod
    def init_population(self, initial_genotype = None, index_mapping = None):
        """
        create the initial group of individuals of the population.

        Args:
            initial_genotype (list-like): the genotype from which extra values are pulled
            for the first evaluation of the fitness function.
            index_mapping (IndexMapping): object storing the indexing from the space of the
            genotypes in a Cooperative Coevolution to the train space.
        """
        pass

    @abstractmethod
    def evolve(self, genotype=None, index_mapping=None):
        """
        perform one iteration of selection, variation, and mutation.

        Args:
            genotype (list-like): the genotype from which extra values are pulled
            for the evaluation of the fitness function.
            index_mapping (IndexMapping): object storing the indexing from the space of the
            genotypes in a Cooperative Coevolution to the train space.
        """
        pass

    @abstractmethod
    def has_converged(self):
        """
        check if our population has met some convergence condition.
        """
        pass

    def generations(self):
        """
        Return the number of generations this algorithm has run for.
        """
        return self._generations

    def evaluations(self):
        """
        Return how many times the fitness function has been evaluated.
        """
        return self._evaluations

    def get_best_genotype(self, n):
        """
        Return the genotype of the best individual in the population.
        """
        indices = np.argsort(self._fitnesses)
        return np.copy(self._population[indices[:n], :])

    def get_best_fitness(self, n=1):
        """
        Return the best n values of the fitness.
        """
        return np.sort(self._fitnesses)[:n]
