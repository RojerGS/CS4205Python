# import statements
import numpy as np
from abc import ABC, abstractmethod

class GeneticAlgorithm(ABC):
    """
    Abstract Base Class which defines components necessary
    to the definition of a genetic algorithm
    """
    def __init__(self, fitness_function, genome_length, population_size):
        """
        initialize the properties of a genetic algorithm
        """
        self._fitness_function = fitness_function
        self._genome_length = genome_length
        self._population_size = population_size
        self._generations = 0
        self._evaluations = 0

    @abstractmethod
    def init_population(self):
        """
        create the initial group of individuals of the population
        """
        pass

    @abstractmethod
    def evolve(self):
        """
        perform one iteration of selection, variation, and mutation
        """
        pass

    @abstractmethod
    def has_converged(self):
        """
        check if our population has met some convergence condition
        """
        pass

    def generations(self):
        """
        Return the number of generations this algorithm has run for
        """
        return self._generations

    def evaluations(self):
        """
        Return how many times the fitness function has been evaluated
        """
        return self._evaluations

    def get_best(self, n):
        """
        Return the genotype of the best individual in the population
        """
        indices = np.argsort(self._fitnesses)
        return np.copy(self._population[indices[:n], :])

    def get_best_fitness(self, n=1):
        """
        Return the best n values of the fitness
        """
        return np.sort(self._fitnesses)[:n]
