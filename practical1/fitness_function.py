
class FitnessFunction:
    
    def __init__(self,m,k,d):
        self.m = m
        self.k = k
        self.d = d
        self.evaluations = 0
        self.optimum = m * k #TODO : this is the optimum for OneMax, not your function
        self.elite = None
        
    #The purpose of this custom exception is to perform a naughty trick: halt the GA as soon as the optimum is found
    #do not modify it
        
    class OptimumFoundCustomException(Exception):
        def __init__(self,message,errors):
            super().__init__(message)
            self.errors = errors
         
    def evaluate(individual):
        self.evaluations += 1
        #TODO : You have to implement the correct evaluation function. The following is OneMax (counts 1s). Remember to modify the optimum as well
        result = 0.0
        
        for i in range(len(individual.genotype)):
            result += individual.genotype[i]
        
        
        individual.fitness = result 
        
        if(self.elite == None or self.elite.fitness < individual.fitness):
            self.elite = individual.clone()
        
        if (result == optimum):
            raise OptimumFoundCustomException("GG EZ")
            
        
        
        
        
        
        
      