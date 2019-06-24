import numpy as np
from math import sin, cos, pi



class FunctionFactory(object):
    """
    Class to produce functions which will be used for the evaluation of fitness
    in our evolutionary algorithms. The functions are outlined in
    "Exploiting Linkage Information...".
    """

    @staticmethod
    def get_sphere():
        """
        Sphere function
        """
        return lambda x: sum(np.array(x)**2)

    @staticmethod
    def get_rosenbrock():
        """
        Rosenbrock function
        """
        return lambda x: sum([(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2) for i in range(len(x)-1)])

    @staticmethod
    def get_rastrigin():
        """
        Rastrigin function
        """
        return lambda x: (10*len(x) + sum([xi**2 - 10*cos(2*pi*xi) for xi in x]))

    @staticmethod
    def get_michalewicz():
        """
        Michalewicz function
        """
        return lambda x: sum([-sin(x[i]) * (sin((i+1)*x[i]**2/pi)**20) for i in range(len(x))])

    @staticmethod
    def get_ellipsoid():
        """
        Ellipsoid function
        """
        return lambda x: sum([10**((6*i)/(len(x)-1)) * (x[i]**2) for i in range(len(x))])

    @staticmethod
    def get_rotated_ellipsoid(theta = 45.0):
        """
        Rotated ellipsoid function
        """
        theta_rad = np.radians(theta)
        rotation_matrix = np.array(
            [[np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]]
        )
        def rotated_ellipsoid(x):
            for i in range(len(x)-1):
                x[i:i+2] = np.dot(rotation_matrix, x[i:i+2])
            return sum([10**((6*i)/(len(x)-1)) * (x[i]**2) for i in range(len(x))])

        return rotated_ellipsoid

    @staticmethod
    def get_soreb(k=5, theta=45.0):
        """
        SoREB function
        """
        t = theta_rad = np.radians(theta)
        c = np.cos; s = np.sin;
        inner_matrix = np.array([[c(t), -s(t)], [s(t), c(t)]])

        final_matrix = np.identity(k)
        for i in range(k-1):
            mult_matrix = np.identity(k)
            mult_matrix[i:(i+2), i:(i+2)] = inner_matrix
            final_matrix = np.dot(final_matrix, mult_matrix)

        ellipsoid = FunctionFactory.get_ellipsoid()

        def soreb(x):
            if (len(x) % k != 0): raise Exception("The value of k might need to be a factor of the length of x?")
            return sum([ellipsoid(final_matrix.dot(np.array(x[k*i:k*(i+1)]))) for i in range(len(x)//k)])

        return soreb



if __name__ == "__main__":
    x = np.random.rand(10)
    print(FunctionFactory.get_sphere()(x))
    print(FunctionFactory.get_rosenbrock()(x))
    print(FunctionFactory.get_rastrigin()(x))
    print(FunctionFactory.get_michalewicz()(x))
    print(FunctionFactory.get_ellipsoid()(x))
    print(FunctionFactory.get_soreb()(x))
