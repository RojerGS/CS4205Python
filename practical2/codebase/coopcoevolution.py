"""Implements gray- and black-box cooperative coevolution,
based on the implementations of the real-valued optimizations algorithms provided.
In black-box cooperative coevolution, we want to minimize a function f,
where f is a function of n variables, but we know that it makes sense to divide
the problem into several populations, each concerned with 
????????????????????"""

from de import DE
import numpy as np

class GrayBoxCoopCoev(object):
    def __init__(self, fs, variable_partition, ga, pop_size):
        self.fs = fs
        self.variable_partition = variable_partition
        self.ga = ga
        self.population_size = pop_size

        self.subsimulations = []
        self.total_genome_length = -1
        for f, partition in zip(self.fs, self.variable_partition):
            self.total_genome_length = max(self.total_genome_length,
                                            max(partition))
            self.subsimulations.append(self.ga(f,
                                            len(partition),
                                            pop_size, lower_bounds=-10, upper_bounds=10))
        self.total_genome_length += 1

    def evolve(self):
        for ga in self.subsimulations:
            ga.evolve()

    def get_best(self, n):
        best_inds = np.zeros((n, self.total_genome_length))
        for sim, part in zip(self.subsimulations, self.variable_partition):
            sub_best = sim.get_best(n)
            best_inds[:, part] = sub_best
        return best_inds

    def get_best_fitness(self, n):
        acc = self.subsimulations[0].get_best_fitness(n)
        for sim in self.subsimulations[1:]:
            acc += sim.get_best_fitness(n)
        return acc

if __name__ == "__main__":
    # ### decompose the sphere problem
    # def f(l):
    #     return l[0]**2
    # gbcc = GrayBoxCoopCoev([f,f,f,f], [[0],[1],[2],[3]], DE, 100)
    # for _ in range(10):
    #     gbcc.evolve()
    #     print(gbcc.get_best_fitness(1))

    ### decoupled rosenbrock
    def ros(l):
        return 100*(l[0]**2 - l[1])**2 + (l[0] - 1)**2
    gbcc = GrayBoxCoopCoev([ros, ros, ros], [[0,1],[2,3],[4,5]], DE, 100)
    for _ in range(100):
        gbcc.evolve()
        print(gbcc.get_best_fitness(1))