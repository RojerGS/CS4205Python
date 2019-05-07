import numpy as np
from random import shuffle

 
def getbest(candidates):
    best = candidates[0]
    
    for i in range(len(candidates)):
        if (candidates[i].fitness >= best.fitness):
            best = candidates[i]
    return best

def tournamentselect(individuals):
    result = []
    n = len(individuals) // 2
    
    for j in range(2):
        perm = list(range(2*n))
        shuffle(perm)
        
        for i in range (n//2):
            candidates = []
            
            for k in range(4):
                candidates.append(individuals[perm[4*i + k]])
            
            result.append(getbest(candidates))
            
    return result


                
                
                
                
                
       