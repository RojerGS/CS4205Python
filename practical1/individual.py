import numpy as np
from copy import deepcopy

class Individual(object):
    genotype = None
    fitness = 0.0


    def __init__(self, genotype_length = None):
        """ individual constructor
        If we don't give it a genotype length, the individual will be empty and need copying
        """
        if genotype_length != None:
            print("hoesgf")
            self.genotype = []
            for i in range(genotype_length):
                self.genotype.append(np.random.randint(2))


    def clone(self):
        """ Create a new individual that is a deep copy of this individual
        """
        retClone = Individual()
        retClone.genotype = []

        for i in range(len(self.genotype)):
            retClone.genotype.append(self.genotype[i]) #maybe change to deep copy later

        retClone.fitness = self.fitness
        return retClone


    def __repr__(self):
        """ Create a text representation of an individual, which is the genotype
        """
        return ''.join(list(map(lambda x: str(x), self.genotype)))
