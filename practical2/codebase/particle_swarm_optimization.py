# imports
from genetic_algorithm import GeneticAlgorithm
import numpy as np
from enum import Enum
from copy import deepcopy as dc
from genome_utils import *



class PSOTopologies(Enum):
    """
    Enumerated type which represents the topology strategy.
    Topology describes the way that particles observe the performance
    of their "neighbors" to influence their trajectories.
    """
    GBEST = 1       # fully connected graph
    LBEST = 2       # one big cycle
    VONNEUMANN = 3  # single component with vertices of degree 4

class PSOInteractions(Enum):
    """
    Enumerated type which represents interaction type.
    Interaction types are the ways that particles calculate their
    new trajectories from their neighbors.
    """
    NORMAL = 1  # read the best position of the most fit neighbor
    FIPS = 2    # move your trajectory towards the best positions
                # of all of your neightbors

class PSOVelocityCap(Enum):
    """
    Enumerated type which represents the option to cap velocity of a particle.
    """
    UNCAPPED = 1    # velocity can increase forever
    MAXCAP = 2      # the magnitude of the velocity on a dimension, at its cap,
                    # is half of the size of the dimension's domain




class ParticleSwarmOptimization(GeneticAlgorithm):
    """
    Class abstraction of Particle Swarm Optimization, as described in
    "Particle swarm optimization" by Poli et al., 2007
    algorithm description starts on page 35
    """
    class Particle(object):
        """
        Inner class and abstraction for the "particle" in PSO
        """
        def __init__(self, genome_length, fitness_function,
                     lower_bounds, upper_bounds,
                     interaction = PSOInteractions.NORMAL,
                     velocity_cap_type = PSOVelocityCap.MAXCAP,
                     phi_i = 2.0, phi_g = 2.0):
            """
            generate a particle with a random position and velocity

            Args:
                genome_length (int): The length of the vector passed into the fitness function.
                fitness_function (function): The function to minimize.
                interaction (PSOInteractions): The method used for calculating the velocity
                of a particle during each step of the algorithm.
                lower_bounds (float, list-like): the lower bounds for the values in the genotype.
                upper_bounds (float, list-like): the upper bounds for the values in the genotype.
            """

            self._fitness_function = fitness_function
            self._interaction = interaction
            self._genome_length = genome_length

            self._velocity_cap_type = velocity_cap_type
            self._phi_i = phi_i
            self._phi_g = phi_g

            self._lower_bounds = lower_bounds
            self._upper_bounds = upper_bounds
            self._speed_cap = (self._upper_bounds-self._lower_bounds)/2

            self._curr_position = np.random.rand(self._genome_length)*(self._upper_bounds-self._lower_bounds)+self._lower_bounds
            self._best_position = dc(self._curr_position)
            self._velocity = np.random.rand(genome_length)*self._speed_cap*np.random.choice([-1,1])
            self._best_fitness = float('inf')

        def evaluate(self, genotype, index_mapping):
            """
            reevaluate the fitness of the particle, accounting for additional values
            that need to be used in evaluation as necessary.
            """
            if genotype is not None:
                genotype = extrapolate_values(subgenotype = self._curr_position,
                                                 genotype = genotype, index_mapping = index_mapping)
            else:
                genotype = self._curr_position

            current_fitness = self._fitness_function(genotype)
            if current_fitness < self._best_fitness:
                self._best_fitness = current_fitness
                self._best_position = dc(self._curr_position)

        def update_velocity(self, neighbors):
            """
            Update the velocity of the particle based on its interaction type
            and the neighbors which have been given to it.

            Args:
                neighbors (list-like): a list of neighbors which will influence
                the movement of the particle.
            """
            if self._interaction == PSOInteractions.NORMAL:
                self.update_velocity_normal(neighbors)
            elif self._interaction == PSOInteractions.FIPS:
                self.update_velocity_FIPS(neighbors)

            #cap the velocity as necessary
            if self._velocity_cap_type == PSOVelocityCap.MAXCAP:
                for i in range(self._genome_length):
                    if self._velocity[i] < -self._speed_cap[i]:
                        self._velocity[i] = -self._speed_cap[i]
                    elif self._velocity[i] > self._speed_cap[i]:
                        self._velocity[i] = self._speed_cap[i]

        def update_velocity_normal(self, neighbors):
            """
            Update the velocity of the particle with the deterministic form of
            velocity recalculation.

            Args:
                neighbors (list-like): a list of neighbors which will influence
                the movement of the particle.
            """
            best_neighbor_fitness = float('inf')
            best_neighbor = None
            for neighbor in neighbors:
                if neighbor._best_fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor._best_fitness

            u_i = np.random.uniform(0.0, self._phi_i, size=self._genome_length)
            u_g = np.random.uniform(0.0, self._phi_g, size=self._genome_length)

            self._velocity += u_i*(self._best_position-self._curr_position) + u_g*(best_neighbor._best_position-self._curr_position)

        def update_velocity_FIPS(self, neighbors):
            """
            Update the velocity of the particle with the "Fully Informed
            Particle Swarm" method.

            Args:
                neighbors (list-like): a list of neighbors which will influence
                the movement of the particle.
            """
            pass
            phi = self._phi_g + self._phi_i
            chi = 2/(phi-2+np.sqrt(phi**2-4*phi))
            d_avg = np.mean([np.random.uniform(0.0, phi, size=self._genome_length) * (neighbor._best_position-self._curr_position)
                            for neighbor in neighbors], axis=0)
            self._velocity = chi*(self._velocity + d_avg)

        def move(self):
            """
            Update the position of the particle with respect to the velocity,
            capping the position of the particle as needed.
            """
            self._curr_position += self._velocity
            for i in range(self._genome_length):
                if self._curr_position[i] < self._lower_bounds[i]:
                    self._curr_position[i] = self._lower_bounds[i]
                elif self._curr_position[i] > self._upper_bounds[i]:
                    self._curr_position[i] = self._upper_bounds[i]



    def __init__(self, fitness_function, genome_length, population_size=25,
                 *, phi_i=2.0, phi_g=2.0, lower_bounds=-3.0, upper_bounds=3.0,
                 initial_genotype = None, index_mapping = None,
                 topology = PSOTopologies.VONNEUMANN,
                 interaction = PSOInteractions.FIPS,
                 velocity_cap_type = PSOVelocityCap.MAXCAP,
                 max_generations = float('inf'),
                 max_evaluations = float('inf'),
                 goal_fitness = float('-inf')):
        """
        Initialize the PSO with parameters.

        Args:
            fitness_function (function): The function to minimize.
            genome_length (int): The number of variables in the genome.
            population_size (int): the number of individuals in the population.
            topology (PSOTopologies): The method used for linking particles together
            in the swarm.
            interaction (PSOInteractions): The method used for calculating the velocity
            of a particle during each step of the algorithm.
        """
        super(ParticleSwarmOptimization, self).__init__(
              fitness_function=fitness_function,
              genome_length=genome_length,
              population_size=population_size,
              upper_bounds = upper_bounds,
              lower_bounds = lower_bounds)

        self._topology = topology
        self._interaction = interaction
        self._velocity_cap_type = velocity_cap_type

        self._max_generations = max_generations
        self._max_evaluations = max_evaluations
        self._goal_fitness = goal_fitness
        self._elite_finess = float('inf')

        self.init_population(initial_genotype, index_mapping)

    def init_population(self, initial_genotype, index_mapping):
        """
        create the initial group of individuals of the population
        and link them together.
        """
        self._population = [ParticleSwarmOptimization.Particle(
                            genome_length = self._genome_length,
                            fitness_function = self._fitness_function,
                            interaction=self._interaction, velocity_cap_type=self._velocity_cap_type,
                             lower_bounds=self._lower_bounds, upper_bounds=self._upper_bounds)
                            for _ in range(self._population_size)]

        self.init_topology()

        # get initial fitnesses for the first evaluation step
        for individual in self._population:
            individual.evaluate(initial_genotype, index_mapping)

    def init_topology(self):
        """
        define a topology, that is, which particles look at which other particles
        when they recalculate their velocities.
        """
        if self._topology == PSOTopologies.GBEST:
            self.init_topology_gbest()
        elif self._topology == PSOTopologies.LBEST:
            self.init_topology_lbest()
        elif self._topology == PSOTopologies.VONNEUMANN:
            self.init_topology_vonneumann()

    def init_topology_gbest(self):
        """
        define the topology as the gbest topology, in which each particle
        looks at all other particles, so the graph is fully connected.
        A bit hacky, but I'll set the neighborhood to point to the first individual,
        and I'll update the pointer accordingly when evolve is called.
        """
        self._neighborhoods = [[self._population[0]] for _ in range(self._population_size)]

    def init_topology_lbest(self):
        """
        define the topology as the gbest topology, in which each particle
        looks at adjacently indexed particles, so our topology is one big cycle.
        """
        self._neighborhoods = [[self._population[i-1], self._population[0 if i==self._population_size else i]]
                               for i in range(self._population_size)]

    def init_topology_vonneumann(self):
        """
        define the topology as the von neumann topology, in which each particle
        looks at 4 other particles.
        I think it's supposed to be lbest with 2 random additional edges.
        """
        self.init_topology_lbest()
        for i in range(len(self._neighborhoods)):
            neighborhood = set(self._neighborhoods[i])
            while len(neighborhood) < 4:
                j = np.random.randint(0, self._population_size)
                neighborhood.add(self._population[j])
            self._neighborhoods[i] = list(neighborhood)

    def evolve(self, genotype=None, index_mapping=None):
        """
        recalculate velocities, move, and evaluate fitness.
        """
        for i in range(len(self._population)):
            individual = self._population[i]
            individual.update_velocity(self._neighborhoods[i])

        for individual in self._population:
            individual.move()
            individual.evaluate(genotype, index_mapping)
            if individual._best_fitness < self._elite_finess:
                self._elite = individual
            self._evaluations += 1

        if self._topology == PSOTopologies.GBEST:
            self._neighborhoods = [[self._elite] for _ in range(self._population_size)]

        self._generations += 1

    def get_curr(self, n=1):
        return [i._curr_position for i in self._population][:n]

    def get_best_genotypes(self, n=1):
        """
        Return the genotype of the best individual in the population.
        """
        best_individuals = sorted(self._population, key=lambda x: x._best_fitness)
        return [bi._best_position for bi in best_individuals][:n]

    def get_best_fitness(self, n=1):
        """
        Return the best n values of the fitness.
        """
        fitnesses = [i._best_fitness for i in self._population]
        return np.sort(fitnesses)[:n]

    def has_converged(self):
        """
        check if our population has met some convergence condition.
        """
        return (self._generations >= self._max_generations
                or self._evaluations >= self._max_evaluations
                or self.get_best_fitness() <= self._goal_fitness)



if __name__ == "__main__":
    from random import seed
    seed(100)

    from fitness_functions import FunctionFactory as FF
    f = FF.get_sphere()

    pso = ParticleSwarmOptimization(fitness_function = f,
                                    genome_length = 10,
                                    population_size = 10,
                                    max_generations = 100,
                                    interaction = PSOInteractions.FIPS,
                                    topology = PSOTopologies.VONNEUMANN,
                                    velocity_cap_type = PSOVelocityCap.MAXCAP)

    while not (pso.has_converged()):
        pso.evolve()
        print(pso.get_best_fitness())
    print(pso.get_best_genotypes(n=5))
