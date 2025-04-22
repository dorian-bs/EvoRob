import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

from src.EA.ES import ES, ES_opts
from src.world.World import World
from src.world.envs.TestFunctions import f_reversed_ackley, f_rosenbrock, f_rastrigin2d, f_schaffer1d, f_custom_benchmark
from src.utils.Filesys import get_project_root

""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.
    
    Exercise0 warm-up: This exercise is a warm-up to understanding the flow of information in the software. 
    In the previous exercise you built your own Evolutionary Strategy which will be used to optimise the parameters
    in the reversed Ackley environment. Additionally, we will integrate the original cmaes code built by Hansen et al.:
    https://cma-es.github.io/index.html
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'InverseAckley'

def plot_fitness_curve(fitnesses_full):
    """
    Plots the best fitness and average fitness for each generation.

    :param fitnesses_full: A 2D array where each row corresponds to a generation,
                           and each column corresponds to the fitness of an individual.
    """
    best_fitness = np.max(fitnesses_full, axis=1)
    avg_fitness = np.mean(fitnesses_full, axis=1)

    plt.plot(best_fitness, label='Best Fitness', color='blue')
    plt.plot(avg_fitness, label='Average Fitness', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Curve')
    plt.legend()
    plt.grid()
    plt.show()


def plot_3d_landscape_with_means(fitness_function, mean_positions, bounds=(-5, 5), resolution=100):
    """
    Plots the 3D landscape of a fitness function and overlays the mean positions of each generation.

    :param fitness_function: The fitness function to plot (e.g., f_custom_benchmark).
    :param mean_positions: A list or array of mean positions for each generation (shape: [n_generations, 2]).
    :param bounds: Tuple specifying the (min, max) bounds for x and y axes.
    :param resolution: The resolution of the grid for plotting the landscape.
    """
    # Create a grid of x and y values
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute the fitness values for the grid
    Z = fitness_function(X, Y)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the fitness landscape
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    # Overlay the mean positions of each generation
    mean_positions = np.array(mean_positions)
    ax.scatter(mean_positions[:, 0], mean_positions[:, 1], 
               fitness_function(mean_positions[:, 0], mean_positions[:, 1]),
               color='red', label='Mean Positions', s=50)

    # Add labels and title
    ax.set_title("3D Fitness Landscape with Mean Positions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Fitness")
    ax.legend()

    # Show the plot
    plt.show()


#%% Q0.1
#TODO: understanding the world
class AckleyWorld(World):
    def __init__(self):
        self.n_params = 2

    def geno2pheno(self, genotype):
        x, y = genotype
        return np.array([x, y])

    def evaluate_individual(self, genotype):        
        x, y = self.geno2pheno(genotype)
        fitness = f_reversed_ackley(x, y)
        return fitness


class MyWorld(World):
    def __init__(self):
        self.n_params = 2

    def geno2pheno(self, genotype):
        x, y = genotype
        return np.array([x, y])

    def evaluate_individual(self, genotype):
        x, y = self.geno2pheno(genotype)
        fitness = -f_reversed_ackley(x, y)
        return fitness


def run_EA(ea, world):
    # TODO: Understand the EA ask/tell interface.
    for _ in range(ea.n_gen):
        pop = ea.ask()
        fitnesses_gen = np.empty(ea.n_pop)
        for index, individual in enumerate(pop):
            fit_ind = world.evaluate_individual(individual)
            fitnesses_gen[index] = fit_ind
        ea.tell(pop, fitnesses_gen)


def main():
    #%% Q0.1
    world = AckleyWorld()
    n_parameters = world.n_params  # only x,y params are optimised

    #%% Q0.2
    ES_opts["min"] = -10
    ES_opts["max"] = 10
    ES_opts["num_parents"] = 100
    ES_opts["num_generations"] = 50
    ES_opts["mutation_sigma"] = 2.5
    population_size = ES_opts["num_parents"]
    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'ES')

    # Create the EA instance for AckleyWorld
    ea = ES(population_size, n_parameters, ES_opts, results_dir)

    #%% Report results for AckleyWorld
    # run_EA(ea, world)
    # fitnesses_full = np.load(os.path.join(results_dir, 'full_f.npy'))
    # plot_fitness_curve(fitnesses_full)
    # plot_3d_landscape_with_means(
    #     f_reversed_ackley,
    #     ea.mean_positions,
    #     bounds=(-4, 4),
    #     resolution=100
    # )

    #%% Change the World
    myworld = MyWorld()

    # Create a new EA instance for MyWorld
    # results_dir_myworld = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'MyWorld')
    # ea_myworld = ES(population_size, n_parameters, ES_opts, results_dir_myworld)
    # run_EA(ea_myworld, myworld)
    # fitnesses_full_myworld = np.load(os.path.join(results_dir_myworld, 'full_f.npy'))
    # plot_fitness_curve(fitnesses_full_myworld)


    #%% Change the EA
    #TODO: Change the ea function
    from src.EA.CMAES import CMAES, CMAES_opts
    CMAES_opts["min"]= -40
    CMAES_opts["max"]= 40
    CMAES_opts["num_generations"]= 50
    CMAES_opts["mutation_sigma"]= 3
    results_dir_cmaes = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'CMAES')
    ea_cmaes = CMAES(population_size, n_parameters, CMAES_opts, results_dir_cmaes)

    run_EA(ea_cmaes, myworld)
    fitnesses_full_cmaes = np.load(os.path.join(results_dir_cmaes, 'full_f.npy'))
    plot_fitness_curve(fitnesses_full_cmaes)
    plot_3d_landscape_with_means(
        f_reversed_ackley,
        ea_cmaes.mean_positions,
        bounds=(-50, 50),
        resolution=500
    
    )


if __name__ == '__main__':
    main()

