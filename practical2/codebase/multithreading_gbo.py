from cooperative_evolution_optimizers import GrayBoxOptimizer
import multiprocessing
import numpy as np

class MTGrayBoxOptimizer(GrayBoxOptimizer):
    def __init__(self, *args, **kwargs):
        super(MTGrayBoxOptimizer, self).__init__(*args, **kwargs)

    def _sub_evolve(self, subpop):
        subpop._optimizer.evolve(self._genotype[::], subpop._index_mapping)

    def evolve(self):
        """
        Evolve each subpopulation for one iteration
        """
        self._generations += 1
        # first, get the aggregated genotype representing the elites of all subpopulations
        self._genotype = np.zeros(self._genome_length, dtype=np.double)
        for subpopulation in self._subpopulations:
            elite_genotype = subpopulation._optimizer.get_best_genotypes(n=1)[0]
            lift_mapping = subpopulation._index_mapping.get_lift_mapping()
            for i in lift_mapping:
                self._genotype[lift_mapping[i]] = elite_genotype[i]

        # evaluate current state of our genotype
        self.evaluate(self._genotype)

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pool.map(self._sub_evolve, self._subpopulations)

if __name__ == "__main__":
    from fitness_functions import FunctionFactory as FF
    from differential_evolution import DifferentialEvolution as DE

    # small test with decoupled, non-aligned sphere problems
    f1 = FF.get_sphere()
    functions = [f1, f1]
    input_spaces = [[0,1,2,3], [4,5,6]]
    train_partition = [[0,1,2], [3,4], [5,6]]
    # de-center the bounds to introduce some additional bias
    lower_bounds = [-3]*7
    upper_bounds = [4]*7
    genetic_algorithms = [DE, DE, DE]
    genetic_algorithm_arguments = [
        {'crossover_probability': 0.25, 'f_weight': .1},
        {'crossover_probability': 0.25, 'f_weight': .1},
        {'crossover_probability': 0.25, 'f_weight': .1}
    ]
    
    gbo = MTGrayBoxOptimizer(functions = functions,
                           input_spaces = input_spaces,
                           train_partition = train_partition,
                           lower_bounds = lower_bounds, upper_bounds = upper_bounds,
                           genetic_algorithms = genetic_algorithms,
                           genetic_algorithm_arguments = genetic_algorithm_arguments,
                           max_generations = 100)

    while not (gbo.has_converged()):
        gbo.evolve()
    print(gbo.get_elite_fitness())
    print(gbo.get_elite_genotype())