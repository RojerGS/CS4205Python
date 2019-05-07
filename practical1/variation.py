from enum import Enum

class CrossoverType(Enum):
    Uniform=1
    OnePoint=2
    
class Variation:
    def __init__(self,crossover_type):
        self.crossover_type = crossover_type
    
    def perform_crossover(parent1, parent2):
        if (self.crossover_type == CrossoverType.OnePoint):
            return onePointCrossover(parent1,parent2)
        else:
            return uniformCrossover(parent1, parent2)
       
    def onePointCrossover(parent1, parent2):
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # TODO: remember to use the rand ng to sample random numbers
        
        result = []
        result.append(child1)
        result.append(child2)
        
        return result
        
    def uniformCrossover(parent1, parent2):
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # TODO: remember to use the rand ng to sample random numbers
        
        result = []
        result.append(child1)
        result.append(child2)
        
        return result
        