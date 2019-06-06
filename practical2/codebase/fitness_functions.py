import numpy as np
from math import sin, cos, pi


"""
Defining fitness functions outlined in "Exploiting Linkage Information..."
"""
def sphere(x):
    """
    Sphere problem.
    """
    return sum(np.array(x)**2)

def rosenbrock(x):
    """
    Rosenbrock problem.
    """
    return sum([(100*(x[i+1]-x[i]**2) + (1-x[i])**2) for i in range(len(x)-1)])

def rastrigin(x):
    """
    Rastrigin problem.
    """
    return (10*len(x) + sum([xi**2 - 10*cos(2*pi*xi) for xi in x]))

def michalewicz(x):
    """
    Michalewicz problem.
    """
    return sum([-sin(x[i]) * (sin((i+1)*x[i]**2/pi)**20) for i in range(len(x))])

def ellipsoid(x):
    """
    Ellipsoid problem.
    """
    return sum([10**((6*i)/(len(x)-1)) * (x[i]**2) for i in range(len(x))])

def soreb(x, k=5, theta=45.0):
    """
    SoREB problem.
    """
    # TODO finish this mess of a function
    raise Exception("This fitness function hasn't been implemented yet!")
    if (len(x) % k != 0): raise Exception("The value of k might need to be a factor of the length of x?")
    # check if theta should be in radians or degrees
    # r_theta
    # return sum([ellipsoid(r_theta(x[k*i:k*(i+1)])) for i in range(len(x)/k)])



if __name__ == "__main__":
    x = np.random.rand(10)
    print(sphere(x))
    print(rosenbrock(x))
    print(rastrigin(x))
    print(michalewicz(x))
    print(ellipsoid(x))
    # soreb(x)
