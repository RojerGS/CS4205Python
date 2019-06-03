"""Implements the "classic DE" algorithm, the Differential Evolution
algorithm, a real-valued evolutionary algorithm.
This implementation follows Price K.V., Storn R.M., Lampinen J.A. (2005). The Differential Evolution Algorithm. Differential Evolution: A Practical Approach to Global Optimization, 37-134
In particular, the details in pages 37-50"""

import numpy as np
import random as rnd
from math import floor

class DE(object):
    """This genetic algorithm encapsulates a population that evolves
    according to the Differential Evolution algorithm."""
    def __init__(self, fitness_function, genome_length, population_size, *,
                    lower_bounds=None, upper_bounds=None,
                    crossover_probability=0.5, Fweight=0.1):
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
        self._genome_length = genome_length
        self._population_size = population_size
        self._fitness_function = fitness_function

        # check if we got any lower bounds
        if lower_bounds is None:
            lower_bounds = np.zeros(genome_length, dtype=np.double)
        # this might happen if the lower bound is given as just a constant
        elif not hasattr(lower_bounds, "__iter__"):
            lower_bounds = lower_bounds*np.ones(genome_length, dtype=np.double)
        self._lower_bounds = lower_bounds
        if upper_bounds is None:
            upper_bounds = np.ones(genome_length, dtype=np.double)
        elif not hasattr(upper_bounds, "__iter__"):
            upper_bounds = upper_bounds*np.ones(genome_length, dtype=np.double)
        self._upper_bounds = upper_bounds
        self._crossover_prob = crossover_probability
        # number of times the fitness function was evaluated
        self._evaluations = 0
        # number of generations ran
        self._generations = 0
        self._fweight = Fweight
        self.init_population()

    def init_population(self):
        """Initializes the population for the algorithm"""
        # each row is an individual, each column a feature
        self._population = np.random.rand(self._population_size,
                                            self._genome_length)
        # enforce the lower and upper bounds
        for i, (lb, ub) in enumerate(zip(self._lower_bounds, self._upper_bounds)):
            self._population[:, i] = lb + (ub - lb)*self._population[:, i]

        self._fitnesses = np.zeros(self._population_size, dtype=np.double)
        for i in range(self._population_size):
            self._fitnesses[i] = self._fitness_function(self._population[i, :])
        self._evaluations += self._population_size

        # # sort the individuals
        # indices = np.argsort(self._fitnesses)
        # self._population[::, ::] = self._population[indices, ::]
        # self._fitnesses = self._fitnesses[indices]

    def evolve(self):
        """Create next generation"""
        self._generations += 1
        Ns = list(range(self._population_size))
        mutpop = np.copy(self._population)
        new_fitnesses = np.zeros(self._fitnesses.shape, dtype=np.double)
        for i in range(self._population_size):
            # take r0, r1 and r2
            r0, r1, r2 = rnd.sample(Ns[:i]+Ns[i+1:], 3)
            # create a mutation
            mutant = self._population[r0, :] + self._fweight*(self._population[r1, :] - self._population[r2, :])
            # crop to fit inside the bounds
            leftMask = mutant < self._lower_bounds
            mutant[leftMask] = self._lower_bounds[leftMask]
            rightMask = self._upper_bounds < mutant
            mutant[rightMask] = self._upper_bounds[rightMask]
            # crossover mask
            mask = np.random.rand(self._genome_length) <= self._crossover_prob
            # feature that flips for sure
            mask[floor(self._genome_length*rnd.random())] = True
            mutpop[i, mask] = mutant[mask]
            new_fitnesses[i] = self._fitness_function(mutpop[i, :])
            self._evaluations += 1
        # replace those who were surpassed by their children
        # sbm stands for "the Student Becomes the Master"
        sbm = (new_fitnesses <= self._fitnesses)
        self._population[sbm, :] = mutpop[sbm, :]
        self._fitnesses[sbm] = new_fitnesses[sbm]

    def has_converged(self):
        return np.all(self._population[0] == self._population[1:])

    def generations(self):
        """Returns the number of generations this algorithm has run for"""
        return self._generations

    def evaluations(self):
        """Returns how many times the fitness function has been evaluated"""
        return self._evaluations

    def get_best(self, n):
        indices = np.argsort(self._fitnesses)
        return np.copy(self._population[indices[:n], :])

    def get_best_fitness(self, n):
        """Returns the best n values of the fitness"""
        sorted_fits = np.sort(self._fitnesses)
        return sorted_fits[:n]

if __name__ == "__main__":
    # Solve the "sphere" problem
    def f(l):
        array = np.array(l)
        return np.linalg.norm(array)

    rnd.seed(0)
    np.random.seed(0)

    print("################")
    print("#### START #####")
    print("################")
    de = DE(f, 10, 100, lower_bounds=-3, upper_bounds=3)
    for i in range(100):
        de.evolve()
        print(de.get_best_fitness(1))
    print(de.get_best(1))