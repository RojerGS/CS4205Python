import numpy as np

def sphere(genotype):
    """
    Sphere Problem.
    """
    array = np.array(genotype)
    return np.linalg.norm(array)

def decoupled_rosenbrock(genotype):
    """
    Decoupled Rosenbrock Problem.
    """
    return 100*(genotype[0]**2 - genotype[1])**2 + (genotype[0] - 1)**2
