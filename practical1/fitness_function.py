
class FitnessFunction:

    def __init__(self,m,k,d):
        """ construct the fitness function to be maximized
        """
        self.m = m
        self.k = k
        self.d = d
        self.evaluations = 0
        self.optimum = m * k #TODO : this is the optimum for OneMax, not your function
        self.elite = None

    class OptimumFoundCustomException(Exception):
        """ The purpose of this custom exception is to perform a naughty trick: halt the GA as soon as the optimum is found
        """
        def __init__(self,message,errors):
            super().__init__(message)
            self.errors = errors

    def evaluate(self, individual):
        """ function to evaluate the fitness of the Individual
        """
        #TODO : You have to implement the correct evaluation function. The following is OneMax (counts 1s). Remember to modify the optimum as well
       
        self.evaluations += 1
        result = 0.0  
        
        def u(b):
            sum(b)
    
        def fsub(b,k,d):
            if (u(b) == k):
                return 1
            else:
                return ((1 - d)*(k - 1 - u(b)))/(k - 1)
      
        def f(x,k,m,d):
            temp = 0
            b = []
            for i in range(m):
                b = x[(i * k):((i+1) * k)]
                temp += fsub(b,k,d)
            return temp
            
        result = f(individual.genotype, self.k, self.m, self.d)
   
        individual.fitness = result
        
        if (self.elite == None or self.elite.fitness < individual.fitness):
            self.elite = individual.clone()
        if (result == self.optimum):
            raise OptimumFoundCustomException("GG EZ")
