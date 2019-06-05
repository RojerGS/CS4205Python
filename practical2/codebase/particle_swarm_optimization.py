# imports
from genetic_algorithm import GeneticAlgorithm
import numpy as np
from enum import Enum

class PSOTopologies(Enum):
    GBEST = 1
    LBEST = 2
    VONNEUMANN = 3


class PSOInteractions(Enum):
    NORMAL = 1
    FIPS = 2       # I don't feel like implementing this one

class PSOVelocityCap(Enum):
    UNCAPPED = 1
    MAXCAP = 2

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
                     interaction = PSOInteractions.NORMAL,
                     velocity_cap = PSOVelocityCap.UNCAPPED,
                     lower_bounds=-3.0, upper_bounds=3.0,
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

            self._velocity_cap = velocity_cap
            self._phi_i = phi_i
            self._phi_g = phi_g

            # if the given bounds are constants, turn them into vectors
            if not hasattr(lower_bounds, "__iter__"):
                lower_bounds = lower_bounds*np.ones(genome_length, dtype=np.double)
            if not hasattr(upper_bounds, "__iter__"):
                upper_bounds = upper_bounds*np.ones(genome_length, dtype=np.double)
            self._lower_bounds = lb = lower_bounds
            self._upper_bounds = ub = upper_bounds

            self._best_position = self._curr_position = np.random.rand(self._genome_length)*(ub-lb)+lb
            self._velocity = np.random.rand(genome_length)
            self._best_fitness = float('inf')

        def evaluate(self):
            """
            reevaluate the fitness of the particle
            """
            current_fitness = self._fitness_function(self._curr_position)
            if current_fitness < self._best_fitness:
                self._best_fitness = current_fitness
                self._best_position = self._curr_position

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
            if self._velocity_cap == PSOVelocityCap.MAXCAP:
                for i in range(self._genome_length):
                    if self._velocity[i] < self._lower_bounds[i]:
                        self._velocity[i] = self._lower_bounds[i]
                    elif self._velocity[i] > self._upper_bounds[i]:
                        self._velocity[i] = self._upper_bounds[i]

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

            u_i = np.random.normal(0.0, self._phi_i, size=self._genome_length)
            u_g = np.random.normal(0.0, self._phi_g, size=self._genome_length)

            self._velocity += u_i*(self._best_position-self._curr_position) + u_g*(best_neighbor._best_position-self._curr_position)

        def update_velocity_FIPS(self, neighbors):
            """
            Update the velocity of the particle with the "Fully Informed
            Particle Swarm" method

            Args:
                neighbors (list-like): a list of neighbors which will influence
                the movement of the particle.
            """
            pass
            phi = self._phi_g + self._phi_i
            chi = 2/(phi-2+np.sqrt(phi**2-4*phi))
            d_avg = np.mean([np.random.normal(0.0, phi, size=self._genome_length) * (neighbor._best_position-self._curr_position)
                            for neighbor in neighbors])

        def move(self):
            self._curr_position += self._velocity
            for i in range(self._genome_length):
                if self._curr_position[i] < self._lower_bounds[i]:
                    self._curr_position[i] = self._lower_bounds[i]
                elif self._curr_position[i] > self._upper_bounds[i]:
                    self._curr_position[i] = self._upper_bounds[i]



    def __init__(self, fitness_function, genome_length, population_size=25,
                 *, phi_i=2.0, phi_g=2.0, lower_bounds=-3.0, upper_bounds=3.0,
                 topology = PSOTopologies.VONNEUMANN,
                 interaction = PSOInteractions.NORMAL,
                 max_generations = float('inf'),
                 max_evaluations = float('inf'),
                 goal_fitness = float('-inf')):
        """
        Initialize the PSO with parameters

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
              population_size=population_size)

        self._topology = topology
        self._interaction = interaction
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        self._max_generations = max_generations
        self._max_evaluations = max_evaluations
        self._goal_fitness = goal_fitness
        self._elite_finess = float('inf')

        self.init_population()

    def init_population(self):
        """
        create the initial group of individuals of the population
        and link them together.
        """
        self._population = [ParticleSwarmOptimization.Particle(
                            genome_length = self._genome_length,
                            fitness_function = self._fitness_function,
                            interaction=self._interaction,
                             lower_bounds=self._lower_bounds, upper_bounds=self._upper_bounds)
                            for _ in range(self._population_size)]
        self.init_topology()

        for individual in self._population:
            individual.evaluate()

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
        self._neighborhoods = [[self._population[0]] for _ in self._population_size]

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

    def evolve(self):
        """
        recalculate velocities, and move, and evaluate fitness.
        """
        if self._topology == PSOTopologies.GBEST:
            self._neighborhoods = [self._elite for _ in range(self._population_size)]

        for i in range(len(self._population)):
            individual = self._population[i]
            individual.update_velocity(self._neighborhoods[i])

        for individual in self._population:
            individual.move()
            individual.evaluate()
            if individual._best_fitness < self._elite_finess:
                self._elite = individual
            self._evaluations += 1

        self._generations += 1

    def get_best(self, n):
        """
        Return the genotype of the best individual in the population
        """
        return self._elite._best_position

    def get_best_fitness(self, n=1):
        """
        Return the best n values of the fitness
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
    from fitness_functions import decoupled_rosenbrock
    pso = ParticleSwarmOptimization(fitness_function = decoupled_rosenbrock,
                                    genome_length = 100,
                                    population_size = 25,
                                    max_generations = 10,
                                    interaction = PSOInteractions.FIPS)
    while not (pso.has_converged()):
        pso.evolve()
        print(pso.get_best_fitness(1))
    print(pso.get_best(1))