import time
from random import shuffle
from variation import Variation
from selection import Selection
from fitness_function import FitnessFunction


class SimpleGeneticAlgorithm:
    def __init__(self,population_size, m, k, d, crossover_type):
        self.population_size = population_size
        if (self.population_size % 2 == 1):
            print("Error: population size must be multiple of 2")
            sys.exit()
        
        self.fitness_function = FitnessFunction(m,k,d)
        self.genotype_length = m*k
        self.variation = Variation(crossover_type)
        self.selection = Selection()
        self.population = []
        
    def run(generation_limit, evaluations_limit, time_limit):
        self.generation = 0
        self.start_time = time.time()
        
        for i in range(self.population_size):
            individual = Individual(self.genotype_length)
            self.fitness_function.evaluate(individual)
            self.population.append(individual)
        
        #utility stuff
        
        while (not check_Termination_Condition(generation_limit, evaluations_limit, time_limit)):
            print( "> Generation", self.generation, "- best fitness found: ", self.fitness_function.elite.fitness)
            offspring = []
            perm = list(range(self.population_size))
            shuffle(perm)
           
            for i in range(population_size // 2):
                offspring += self.variation.perform_crossover(self.population[perm[2 * i]], self.population[perm[2 * i + 1]])

            for o in offspring:
                self.fitness_function.evaluate(o)
            
            p_and_o = []
            p_and_o += self.population
            p_and_o += self.offspring
         
            self.population = self.selection.tournamentselect(p_and_o)
            
            generation += 1
            
            #log generation ALEXXXXXXXXX
            
            def check_Termination_Condition(generation_limit, evaluations_limit, time_limit):
                if(generation_limit >0 and self.generation >= generation_limit):
                    return True
                
                elif(evaluations_limit > 0 and self.fitness_function.evaluations >= evaluations_limit):
                    return True
                
                elapsed_time = time.time() - start_time
                
                if(time_limit > 0 and elapsed_time >= time_limit):
                    return True
            return False
        
          
            
            
            
         
            
            
        
        
        