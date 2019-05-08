import os, logging, time
from variation import CrossoverType, Variation
from simple_genetic_algorithm import SimpleGeneticAlgorithm
from fitness_function import FitnessFunction

def main():
    generations_limit = -1
    evaluations_limit = -1
    time_limit = 3.0 # in seconds

    if 'experiments' not in os.listdir('.'):
        os.mkdir('experiments')

    population_sizes = [100]
    ms = [8]
    ks = [5]
    ds = [1]

    crossoverTypes = list(CrossoverType)

    i = -1
    for ct in crossoverTypes:
        for p in population_sizes:
            for m in ms:
                for k in ks:
                    for d in ds:
                        # set up file outputs
                        i += 1
                        output_file_name = "experiments/log_p{p}_m{m}_k{k}_d{d}_c{ct}_run{run}.txt".format(
                            p=p, m=m, k=k, d=d, ct=ct.name, run=i
                        )

                        # run genetic algorithm
                        print("Starting run {run} with pop_size={p}, m={m}, k={k}, d={d}, crossover_type={ct}".format(
                              run=i, p=p, m=m, k=k, d=d, ct=ct.name
                        ))
                        sga = SimpleGeneticAlgorithm(p, m, k, d, ct)
                        optimumFound = True
                        try:
                            sga.run(generations_limit, evaluations_limit, time_limit)
                        except FitnessFunction.OptimumFoundCustomException:
                            optimumFound = False

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

    # set up logging

if __name__ == "__main__":
    main()
