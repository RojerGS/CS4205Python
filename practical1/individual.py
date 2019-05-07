import numpy as np
from copy import deepcopy

class Individual(object):
    genotype = []
    fitness = 0.0

    # constructors
    def __init__(self):
        pass
    
    def __init__(self, genotype_length):
        for i in range(genotype_length):
            genotype.append(np.random.randint(2))
 
    def clone(self):
        retClone = Individual()
        for i in range(len(self.genotype)):
            retClone.genotype.append(self.genotype[i]) #maybe change to deep copy later
        retClone.fitness = self.fitness   
        return retClone
    
    def __repr__(self):
        return ''.join(list(map(lambda x: str(x), self.genotype)))
        
        
        
        
