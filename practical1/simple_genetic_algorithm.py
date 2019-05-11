import time, logging, sys
import numpy as np
from random import shuffle
from math import sqrt, ceil

from variation import Variation
from selection import Selection
from fitness_function import FitnessFunction
from individual import Individual

# set up the logger, log file and stream handler
logger = logging.getLogger('simple_genetic_algorithm')
fh = logging.FileHandler('simple_genetic_algorithm.log')
ch = logging.StreamHandler()
logger.setLevel(logging.INFO)
fh.setLevel(logging.INFO)
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)


class SimpleGeneticAlgorithm:
    """ Object to wrap the SGA to maximize the fitness function
    """
    def __init__(self,population_size, m, k, d, crossover_type):
        """ Create the SGA, describing parameters of our fitness function
        as well as the type of crossover to be used
        """
        self.population_size = population_size
        if (self.population_size % 2 == 1):
            print("Error: population size must be multiple of 2")
            sys.exit()

        self.fitness_function = FitnessFunction(m,k,d)
        self.genotype_length = m*k
        self.variation = Variation(crossover_type)
        self.selection = Selection()
        self.population = []
        self.peak_fitnesses = []


    def checkTerminationCondition(self, generation_limit, evaluations_limit, time_limit):
        """ function to decide when to stop running the SGA
        """
        # terminate if most fitnesses are equal
        fitnesses = np.array(list(map(lambda x: x.fitness, self.population)))
        if len(np.unique(fitnesses)) == 1: return True
        # if len(list(filter(lambda x: x.fitness == max(fitnesses), self.population))) > .9*len(self.population): return True

        # terminate if the population contains very few unique genotypes
        # strings = list(map(lambda x: x.__repr__(), self.population))
        # if len(set(strings)) < len(self.population) / (self.genotype_length): return True

        # terminate if the fitness of the elite has stopped increasing

        if(generation_limit >0 and self.generation >= generation_limit):
            return True

        elif(evaluations_limit > 0 and self.fitness_function.evaluations >= evaluations_limit):
            return True

        elapsed_time = time.time() - self.start_time
        if(time_limit > 0 and elapsed_time >= time_limit):
            return True

        return False


    def run(self, generation_limit, evaluations_limit, time_limit):
        """ perform the SGA
        """

        # set the starting point of the algorithm
        self.generation = 0
        self.start_time = time.time()

        # create the iniitial population and evaluate them
        for i in range(self.population_size):
            individual = Individual(self.genotype_length)
            self.fitness_function.evaluate(individual)
            self.population.append(individual)


        # evolutionary loop
        while (not self.checkTerminationCondition(generation_limit, evaluations_limit, time_limit)):
            # log info

            # create permutation of indices
            offspring = []
            perm = list(range(self.population_size))
            shuffle(perm)

            # generate offspring
            for i in range(self.population_size // 2):
                offspring += self.variation.perform_crossover(self.population[perm[2 * i]], self.population[perm[2 * i + 1]])

            # evaluate offspring
            for o in offspring:
                self.fitness_function.evaluate(o)

            # join parents and offspring
            p_and_o = []
            p_and_o += self.population
            p_and_o += offspring

            # select out offspring
            self.population = self.selection.tournamentSelect(p_and_o)
            self.generation += 1

            # update list of peak fitnesses
            self.peak_fitnesses.append(self.fitness_function.elite.fitness)
