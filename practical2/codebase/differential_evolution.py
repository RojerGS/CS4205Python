"""Implements the "classic DE" algorithm, the Differential Evolution
algorithm, a real-valued evolutionary algorithm.
This implementation follows Price K.V., Storn R.M., Lampinen J.A. (2005). The Differential Evolution Algorithm. Differential Evolution: A Practical Approach to Global Optimization, 37-134
In particular, the details in pages 37-50"""
import numpy as np
import random as rnd
from math import floor
from copy import deepcopy as dc
from genetic_algorithm import GeneticAlgorithm



class DifferentialEvolution(GeneticAlgorithm):
    """This genetic algorithm encapsulates a population that evolves
    according to the Differential Evolution algorithm."""
    def __init__(self, fitness_function, genome_length, *,
                 population_size = 50, lower_bounds=0, upper_bounds=1,
                 crossover_probability=0.5, f_weight=0.1,
                 initial_genotype = None, index_mapping = None):
        """Initialize a population that will evolve according to the DE.
            The genome_length is the number of parameters of each individual,
                the population_size and fitness_function are self_explanatory;
            The algorithm will try to minimize the fitness_function, which is
                a function of genome_length variables.
            The lower/upper_bounds is a vector of length genome_length encoding
                the lower/upper bound of each variable; the lower bound
                defaults to 0 and the upper bound defaults to 1;
                Alternatively, a constant can be passed in, which will then be
                taken as the corresponding bound for all the features."""
        super(DifferentialEvolution, self).__init__(
              fitness_function=fitness_function,
              genome_length=genome_length,
              population_size=population_size,
              lower_bounds = lower_bounds,
              upper_bounds = upper_bounds)

        self._crossover_probability = crossover_probability
        self._evaluations = 0   # number of times the fitness function was evaluated
        self._generations = 0   # number of generations ran
        self._f_weight = f_weight
        self.init_population(initial_genotype, index_mapping)

    def init_population(self, initial_genotype, index_mapping):
        """Initializes the population for the algorithm"""
        # each row is an individual, each column a feature
        self._population = np.random.rand(self._population_size,
                                            self._genome_length)
        # enforce the lower and upper bounds
        for i, (lb, ub) in enumerate(zip(self._lower_bounds, self._upper_bounds)):
            self._population[:, i] = lb + (ub - lb)*self._population[:, i]

        self._fitnesses = np.array([float('inf')]*self._population_size, dtype=np.double)
        self.evaluate(initial_genotype, index_mapping)

    def evaluate(self, genotype=None, index_mapping=None):
        if genotype is None:
            for i in range(self._population_size):
                self._fitnesses[i] = self._fitness_function(self._population[i, :])
        else:
            im = index_mapping.get_input_mapping()
            subgenotypes = dc(np.array([dc(genotype)[list(im.keys())] for _ in range(self._population_size)]))

            tm = index_mapping.get_train_mapping()
            subgenotypes[:, list(tm.values())] = self._population

            for i in range(self._population_size):
                self._fitnesses[i] = self._fitness_function(subgenotypes[i, :])

        self._evaluations += self._population_size

    def _evaluate_mutants(self, mutants, genotype=None, index_mapping=None):
        """
        Evaluates the fitness of _population_size possible offspring
        Returns numpy array with corresponding fitnesses
        """
        new_fitnesses = np.zeros(self._fitnesses.shape, dtype=np.double)
        if genotype is None:
            for i in range(self._population_size):
                new_fitnesses[i] = self._fitness_function(mutants[i, :])
        else:
            # the fitness function expects things of length len(index_mapping.get_input_mapping().keys())
            im = index_mapping.get_input_mapping()
            # from the genotype, extract the values we would care about
            subgenotype = np.array(genotype)[list(im.keys())]
            # replicate the subgenotype enough times
            extended_mutants = np.tile(subgenotype, (self._population_size, 1))
            # find the indices where we want to put the features of the mutants
            indices = list(index_mapping.get_train_mapping().keys())
            extended_mutants[:, indices] = mutants

            for i in range(self._population_size):
                new_fitnesses[i] = self._fitness_function(extended_mutants[i, :])

        self._evaluations += self._population_size
        return new_fitnesses

    def evolve(self, genotype=None, index_mapping=None):
        """Create next generation"""
        self._generations += 1
        Ns = list(range(self._population_size))
        mutpop = np.copy(self._population)
        new_fitnesses = np.zeros(self._fitnesses.shape, dtype=np.double)
        for i in range(self._population_size):
            # take r0, r1 and r2
            r0, r1, r2 = rnd.sample(Ns[:i]+Ns[i+1:], 3)
            # create a mutation
            mutant = self._population[r0, :] + self._f_weight*(self._population[r1, :] - self._population[r2, :])
            # crop to fit inside the bounds
            leftMask = mutant < self._lower_bounds
            mutant[leftMask] = self._lower_bounds[leftMask]
            rightMask = self._upper_bounds < mutant
            mutant[rightMask] = self._upper_bounds[rightMask]
            # crossover mask
            mask = np.random.rand(self._genome_length) <= self._crossover_probability
            # feature that flips for sure
            mask[floor(self._genome_length*rnd.random())] = True
            mutpop[i, mask] = mutant[mask]
        new_fitnesses = self._evaluate_mutants(mutpop, genotype=genotype, index_mapping=index_mapping)
        # replace those who were surpassed by their children
        # sbm stands for "the Student Becomes the Master"
        sbm = (new_fitnesses <= self._fitnesses)
        self._population[sbm, :] = mutpop[sbm, :]
        self._fitnesses[sbm] = new_fitnesses[sbm]

    def has_converged(self):
        return np.all(self._population[0] == self._population[1:])

    def get_best_genotypes(self, n=1):
        """
        get a list of the genotypes of the n best individuals in the population.
        """
        indices = np.argsort(self._fitnesses)
        return np.copy(self._population[indices[:n], :])

    def get_best_fitness(self, n=1):
        """
        get a list of the n best fitnesses of the population.
        """
        return np.sort(self._fitnesses)[:n]



if __name__ == "__main__":
    # Solve the "sphere" problem
    from fitness_functions import FunctionFactory as FF
    f = FF.get_sphere()
    rnd.seed(0)
    np.random.seed(0)

    print("################")
    print("#### START #####")
    print("################")
    de = DifferentialEvolution(f, 10, population_size=100, lower_bounds=-3, upper_bounds=3)
    for i in range(100):
        de.evolve()
        print(de.get_best_fitness(1))
    print(de.get_best_genotypes(1))
