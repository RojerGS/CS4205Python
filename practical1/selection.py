import numpy as np
from random import shuffle

class Selection:
    """ Static class which deals with Selection
    Making this a class for consistency with assignment code
    """

    def getBest(self, candidates):
        """ return the individual in a list with the highest fitness
        """
        fitnesses = np.array(list(map(lambda x: x.fitness, candidates)))
        return candidates[fitnesses.argmax()]


    def tournamentSelect(self, individuals):
        """ use tournament selection on a list of individuals, return the selection
        """
        result = []
        n = len(individuals) // 2

        for j in range(2):
            perm = list(range(2*n))
            shuffle(perm)

            for i in range (n//2):
                candidates = []

                for k in range(4):
                    candidates.append(individuals[perm[4*i + k]])

                result.append(self.getBest(candidates))

        return result
