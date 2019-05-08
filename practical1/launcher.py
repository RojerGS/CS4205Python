import os, logging, time
from variation import CrossoverType, Variation
from simple_genetic_algorithm import SimpleGeneticAlgorithm
from fitness_function import FitnessFunction
from itertools import product as cp

def main():
    generations_limit = -1
    evaluations_limit = -1
    time_limit = 1.0 # in seconds

    if 'experiments' not in os.listdir('.'):
        os.mkdir('experiments')

    population_sizes = [100]
    ms = [8]
    ks = [5]
    ds = [1]
    crossoverTypes = list(CrossoverType)

    # set up file outputs
    f = open("experiments/results.csv", "w+")
    f.write("crossover_type, population_size, m, k, d, generations, evaluations, time, best_fitness, optimal\n")

    i = -1
    for (ct, p, m, k, d) in cp(crossoverTypes, population_sizes, ms, ks, ds):

        # run genetic algorithm
        i += 1
        print("Starting run {run} with pop_size={p}, m={m}, k={k}, d={d}, crossover_type={ct}".format(
              run=i, p=p, m=m, k=k, d=d, ct=ct.name
        ))
        sga = SimpleGeneticAlgorithm(p, m, k, d, ct)
        optimumFound = False
        try:
            sga.run(generations_limit, evaluations_limit, time_limit)
        except FitnessFunction.OptimumFoundCustomException:
            optimumFound = True

        print("""{outcome} {bf} found at
        generation\t{gen}
        evaluations\t{evals}
        time\t{time}
        elite\t{elite}
        """.format(outcome = "Optimum" if optimumFound is False else "Best Fitness",
                   bf = sga.fitness_function.elite.fitness,
                   gen = sga.generation,
                   evals = sga.fitness_function.evaluations,
                   time = time.time() - sga.start_time,
                   elite = sga.fitness_function.elite
        ))

        f.write("{ct}, {p}, {m}, {k}, {d}, {gen}, {evals}, {time}, {bf}, {found}\n".format(
            ct = ct.name, p=p, m=m, k=k, d=d,
            gen = sga.generation, evals = sga.fitness_function.evaluations,
            time = time.time() - sga.start_time, bf = sga.fitness_function.elite.fitness,
            found = optimumFound
        ))
        if optimumFound == True: f.write("Optimum found!\n")

    # set up logging

if __name__ == "__main__":
    main()
