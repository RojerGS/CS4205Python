"""
Implements the evolution strategies variant with different variances but no covariances
(case 2 in lecture slides)
"""

from genetic_algorithm import GeneticAlgorithm
from genome_utils import *
import numpy as np

class EvolutionStrategies(GeneticAlgorithm):

    class Individual(object):
        """
        Abstraction of an individual of the ES population,
        an individual is described by its position/genotype (mean of a normal distribution),
        and the variances of said distribution which define how the individual is mutated.
        """

        def __init__(self, genome_length, fitness_function,
                     lower_bounds, upper_bounds, mean = None, variance = None, eps_sigma = 0):

            self._genome_length = genome_length
            self._fitness_function = fitness_function
            self._fitness = None
            self._eps_sigma = eps_sigma
            self._lower_bounds = lower_bounds
            self._upper_bounds = upper_bounds

            if(mean is None):
                self.mean = np.array([np.random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(genome_length)])
            else:
                self.mean = mean

            if(variance is None):
                self.variance = np.array([1 for i in range(genome_length)])
            else:
                self.variance = variance

        def evaluate(self, genotype, index_mapping):

            if genotype is not None:
                subgenotype = extrapolate_values(subgenotype = self.mean,
                                                 genotype = genotype, index_mapping = index_mapping)
            else:
                subgenotype = self.mean

            self._fitness =  self._fitness_function(subgenotype)

            return self._fitness

        def mutate(self):

                self._has_mutated = True

                #Mutate variances
                z = np.random.normal(0, 1/(2 * self._genome_length))
                for i in range(self._genome_length):
                    z_i = np.random.normal(0, 1/(2 * np.sqrt(self._genome_length)))
                    self.variance[i] = max(self._eps_sigma, self.variance[i]*np.exp(z + z_i))

                #Mutate mean
                for i in range(self._genome_length):
                    self.mean[i] += np.random.normal(0, self.variance[i])
                    self.mean[i] = np.clip(self.mean, self._lower_bounds[i], self._upper_bounds[i])

        def crossover(self, other):
            """
            Crossover this individual with another one to produce single offspring
            """
            tmp = np.random.randint(0,1,self._genome_length)
            offspring_mean = np.multiply(self.mean, tmp) + np.multiply(other.mean, np.abs(tmp-1))
            offspring_variance = 0.5*(self.variance + other.variance)

            offspring = EvolutionStrategies.Individual(self._genome_length, self._fitness_function,
                                                       self._lower_bounds, self._upper_bounds,
                                                       offspring_mean, offspring_variance)
            return offspring

    def __init__(self, fitness_function, genome_length, population_size=25,
                 lower_bounds=-3.0, upper_bounds=3.0, initial_genotype = None, index_mapping = None,
                 max_generations = float('inf'),  max_evaluations = float('inf'), goal_fitness = float('-inf'),*,
                 offspring_size = None):
        """
        Args:
            fitness_function (function): The function to minimize.
            genome_length (int): The number of variables in the genome.
            population_size (int): the number of individuals in the population.
            offspring_size (int): NUmber of offspring generated each generation
        """
        super(EvolutionStrategies, self).__init__(
              fitness_function=fitness_function,
              genome_length=genome_length,
              population_size=population_size,
              upper_bounds = upper_bounds,
              lower_bounds = lower_bounds)

        self._offspring_size = 5*population_size if offspring_size is None else offspring_size
        self.init_population(initial_genotype, index_mapping)


    def init_population(self, initial_genotype = None, index_mapping = None):
        """
        create the initial group of individuals of the population.

        Args:
            initial_genotype (list-like): the genotype from which extra values are pulled
            for the first evaluation of the fitness function.
            index_mapping (IndexMapping): object storing the indexing from the space of the
            genotypes in a Cooperative Coevolution to the train space.
        """
        self._population = [EvolutionStrategies.Individual(self._genome_length, self._fitness_function, self._lower_bounds, self._upper_bounds)
                            for _ in range(self._population_size)]

        for i in self._population:
            i.evaluate(initial_genotype, index_mapping)


    def evolve(self, genotype=None, index_mapping=None):
        """
        perform one iteration of selection, variation, and mutation.

        Args:
            genotype (list-like): the genotype from which extra values are pulled
            for the evaluation of the fitness function.
            index_mapping (IndexMapping): object storing the indexing from the space of the
            genotypes in a Cooperative Coevolution to the train space.
        """
        self._generations += 1
        self._evaluations += self._offspring_size

        offspring = []

        for _ in range(self._offspring_size):
            parent1, parent2 = np.random.choice(self._population, 2)
            child = parent1.crossover(parent2)
            child.mutate()
            child.evaluate(genotype, index_mapping)
            offspring.append(child)

        self._population = sorted(self._population + offspring, key = lambda x: x._fitness)[:self._population_size]

    def has_converged(self):
        """
        check if our population has met some convergence condition.
        """
        return False

    def get_best_genotypes(self, n=1):
        """
        Return the genotype of the best n individuals in the population.
        """
        return [ind.mean for ind in sorted(self._population, key = lambda x: x._fitness)[:n]]

    def get_best_fitness(self, n=1):
        """
        Return the best n values of the fitness.
        """
        return [ind._fitness for ind in sorted(self._population, key = lambda x: x._fitness)[:n]]

if __name__ == "__main__":
    from fitness_functions import FunctionFactory as FF
    from matplotlib import pyplot as plt

    f = FF.get_sphere()
    #f = FF.get_rosenbrock()

    pop = EvolutionStrategies(fitness_function = f,
                                genome_length = 5,
                                population_size = 200)

    fit = []
    for i in range(100):
        pop.evolve()
        fit.append(pop.get_best_fitness(1)[0])

    print(pop.get_best_fitness(1)[0])
    print(pop.get_best_genotypes(1)[0])

    plt.plot(fit)
    plt.show()
