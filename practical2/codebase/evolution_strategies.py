"""
Implements the evolution strategies variant with different variances but no covariances
(case 2 in lecture slides)
"""

from genetic_algorithm import GeneticAlgorithm
from genome_utils import *
import numpy as np

class EvolutionStrategies(GeneticAlgorithm):

    def __init__(self, fitness_function, genome_length, population_size=20,
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

        self._max_generations = max_generations
        self._max_evaluations = max_evaluations
        self._goal_fitness = goal_fitness
        self._best_finess = float('inf')
        self._best_genotype = None
        self._offspring_size = 7*population_size if offspring_size is None else offspring_size
        self.init_population(initial_genotype, index_mapping)

    def evaluate(self, mean, genotype=None, index_mapping=None):
        """
        reevaluate the fitness of the particle, accounting for additional values
        that need to be used in evaluation as necessary.
        """
        if (genotype is not None) and (index_mapping is not None):
            genotype = extrapolate_values(subgenotype = mean,
                                             genotype = genotype, index_mapping = index_mapping)
        else:
            genotype = mean

        self._evaluations += 1
        return self._fitness_function(genotype)


    def init_population(self, initial_genotype = None, index_mapping = None):
        """
        create the initial group of individuals of the population.

        Args:
            initial_genotype (list-like): the genotype from which extra values are pulled
            for the first evaluation of the fitness function.
            index_mapping (IndexMapping): object storing the indexing from the space of the
            genotypes in a Cooperative Coevolution to the train space.
        """
        self._population_means = np.random.uniform(self._lower_bounds, self._upper_bounds, size=(self._population_size, self._genome_length))
        self._population_fitnesses = np.array([self.evaluate(x, initial_genotype, index_mapping) for x in self._population_means])
        #self._population_variances = np.random.uniform(0, 0.1*(self._upper_bounds - self._lower_bounds), size=(self._population_size, self._genome_length))
        self._population_variances = np.random.uniform(0.01*(self._upper_bounds - self._lower_bounds), 0.05*(self._upper_bounds - self._lower_bounds), size=(self._population_size, self._genome_length))

        self._best_fitness = np.min(self._population_fitnesses)
        self._best_genotype = self._population_means[np.argmin(self._population_fitnesses)]

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

        #Randomly select parent from current population, parent_set_1[i] and parent_set_2[i] are
        #recombined to get offspring number i
        parent_set_1 = np.random.randint(0, self._population_size, size = self._offspring_size)
        parent_set_2 = np.random.randint(0, self._population_size, size = self._offspring_size)

        #Get means of parents
        parent_means_1 = self._population_means[parent_set_1]
        parent_means_2 = self._population_means[parent_set_2]

        #Get variances of parents
        parent_variances_1 = self._population_variances[parent_set_1]
        parent_variances_2 = self._population_variances[parent_set_2]

        #Recombine means of parents
        recombination_matrix = np.random.randint(0, 1, size = (self._offspring_size, self._genome_length))
        offspring_means = np.multiply(parent_means_1, recombination_matrix) - np.multiply(parent_means_2, recombination_matrix - 1)

        #Recombine variances of parents
        offspring_variances = 0.5*(parent_variances_1 + parent_variances_2)

        #Mutate offspring variances
        z_vector = np.random.normal(0, np.sqrt(1/(2 * self._genome_length)), size = (self._offspring_size, 1))
        zi_matrix = np.random.normal(0, np.sqrt(1/(2 * np.sqrt(self._genome_length))), size = (self._offspring_size, self._genome_length))
        offspring_variances = np.multiply(offspring_variances, np.exp(zi_matrix + z_vector))

        #Mutate offspring means
        offspring_means += np.array([np.random.multivariate_normal(mean = np.zeros(self._genome_length), cov = np.diag(var)) for var in offspring_variances])
        offspring_fitnesses = np.array([self.evaluate(x, genotype, index_mapping) for x in offspring_means])

        """
        all_fitnesses = np.concatenate((self._population_fitnesses, offspring_fitnesses))
        all_means = np.concatenate((self._population_means, offspring_means))
        all_variances = np.concatenate((self._population_variances, offspring_variances))
        most_fit = np.argsort(all_fitnesses)[:self._population_size]
        self._population_means = all_means[most_fit]
        self._population_variances = all_variances[most_fit]
        self._population_fitnesses = all_fitnesses[most_fit]
        """

        #Perform selection among offspring
        most_fit = np.argsort(offspring_fitnesses)[:self._population_size]
        self._population_means = offspring_means[most_fit]
        self._population_variances = offspring_variances[most_fit]
        self._population_fitnesses = offspring_fitnesses[most_fit]

        if(self._population_fitnesses[0] < self._best_fitness):
            self._best_fitness = self._population_fitnesses[0]
            self._best_genotype = self._population_means[0,:]

    def has_converged(self):
        """
        check if our population has met some convergence condition.
        """
        return (self._generations >= self._max_generations
                or self._evaluations >= self._max_evaluations
                or self.get_best_fitness() <= self._goal_fitness)

    def get_best_genotypes(self, n=1):
        """
        Return the genotype of the best n individuals in the population.
        """
        tmp = np.argsort(self._population_fitnesses)[:n]

        return self._population_means[tmp]

    def get_best_fitness(self, n=1):
        """
        Return the best n values of the fitness.
        """
        if(n == 1):
            return [self._best_fitness]

        return sorted(self._population_fitnesses)[:n]

if __name__ == "__main__":
    from fitness_functions import FunctionFactory as FF
    from matplotlib import pyplot as plt

    #f = FF.get_sphere()
    f = FF.get_rosenbrock()
    #f = FF.get_soreb()

    pop = EvolutionStrategies(fitness_function = f,
                                genome_length = 3,
                                population_size=10,
                                lower_bounds=-100, upper_bounds=100)


    fit = []

    for _ in range(1000):
        pop.evolve()
        fit.append(pop.get_best_fitness(1)[0])

    print(pop.get_best_genotypes(10))
    print(pop.get_best_fitness(10))
    print(pop._best_fitness)
    plt.plot(fit)
    plt.show()
