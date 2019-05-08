from enum import IntEnum, auto
import random

class CrossoverType(IntEnum):
    Uniform = auto()
    OnePoint = auto()

class Variation:
    """ Variation object to store the variation being used
    """
    def __init__(self,crossover_type):
        """ Variation constructor which stores the type of crossover being used
        """
        self.crossover_type = crossover_type


    def perform_crossover(self, parent1, parent2):
        """ Wrapper function to use the appropriate variation mechanism
        """
        if (self.crossover_type == CrossoverType.OnePoint):
            return self.onePointCrossover(parent1,parent2)
        else:
            return self.uniformCrossover(parent1, parent2)


    def onePointCrossover(self, parent1, parent2):
        """ Perform one point crossover of individuals, return children
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        rn = randit(1,parent1.genotype_length)
        child1.genotype[0:rn] = parent2[0:rn]
        child2.genotype[rn:(parent1.genotype_length)] = parent1[rn:(parent1.genotype_length)]
       
        # TODO: remember to use the rand ng to sample random numbers

        result = []
        result.append(child1)
        result.append(child2)

        return result

    def uniformCrossover(self, parent1, parent2):
        """ Perform uniform crossover of individuals, return children
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        for i in range(parent1.genotype_length):
            randn = random.uniform(0,1)
            temp=0
            if(randn >= 0.5):
                temp=child1.genotype[i]
                child1.genotype[i]=child2.genotype[i]
                child2.genotype[i]= temp
            else:
                continue
               

        # TODO: remember to use the rand ng to sample random numbers

        result = []
        result.append(child1)
        result.append(child2)

        return result
