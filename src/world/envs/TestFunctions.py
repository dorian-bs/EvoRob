import numpy as np


def f_reversed_ackley(x, y):
    return -1 * (
        -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )

def f_rosenbrock(x, y):
    return 100*(y-x**2)**2+(x-1)**2

def f_rastrigin2d(x, y):
    return (x**2 - 10 * np.cos(2 * np.pi * x)) +  (y**2 - 10 * np.cos(2 * np.pi * y)) + 20

def f_schaffer1d(x):
    f1 = -np.square(x)
    f2 = -np.square(x-2)
    return f1, f2

def f_custom_benchmark(x, y):
    """
    Custom optimization benchmark function.

    :param x: First parameter (float or array-like).
    :param y: Second parameter (float or array-like).
    :return: Fitness value(s) for the given input(s).
    """
    # Quadratic bowl term
    bowl_term = (x**2 + y**2)

    # Sinusoidal oscillations
    oscillation_term = 10 * (np.sin(5 * x) + np.sin(5 * y))

    # Combine terms
    fitness = bowl_term + oscillation_term

    return fitness