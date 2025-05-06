import os
from typing import Dict

import numpy as np

from src.utils.Filesys import search_file_list

ES_opts = {
    "min": -4,
    "max": 4,
    "num_parents": 16,
    "num_generations": 100,
    "mutation_sigma": 0.3,
    "sigma_limit": 0.1,
}


class ES:

    def __init__(self, n_pop, n_params, opts: Dict=ES_opts, output_dir: str="./results/ES"):
        """
        Evolutionary Strategy [INCOMPLETE]

        :param n_pop: population size
        :param n_params: number of parameters
        :param opts: algorithm options
        :param output_dir: output directory Default = "./results/ES"
        """
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.n_parents = opts["num_parents"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.current_mean = self.initialise_x0()  # Initialize the mean vector
        self.current_sigma = opts["mutation_sigma"]
        self.sigma_limit = opts["sigma_limit"]

        #% bookkeeping
        self.directory_name = output_dir
        self.full_x = []
        self.full_fitness = []
        self.mean_positions = []  # New attribute to store mean positions
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None

    def ask(self):
        if self.current_gen==0:
            new_population = self.initialise_x0()
        else:
            new_population = self.generate_mutated_offspring(self.n_pop)
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=True):
        parents_population, parents_fitness = self.sort_and_select_parents(
            solutions, function_values, self.n_parents
        )
        self.current_mean = self.update_population_mean(parents_population, parents_fitness)
        self.current_sigma = self.update_sigma()

        # Store the current mean position
        self.mean_positions.append(self.current_mean)

        #% Some bookkeeping
        self.full_fitness.append(function_values)
        self.full_x.append(solutions)
        self.x = parents_population
        self.f = parents_fitness

        if np.max(function_values) > self.f_best_so_far:
            best_index = np.argmax(function_values)
            self.f_best_so_far = function_values[best_index]
            self.x_best_so_far = solutions[best_index]

        if self.current_gen % 5 == 0:
            print(f"Generation {self.current_gen}:\t{self.f_best_so_far}\n"
                  f"Mean fitness:\t{self.f.mean()} +- {self.f.std()}\n"
                  f"Sigma: {self.current_sigma} \n"
                  )

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def initialise_x0(self):
        """
        Initializes the starting population for the evolutionary strategy.

        :return: A 2D array where each row is an individual and each column is a parameter.
        """
        mean_vector = np.random.uniform(low=self.min, high=self.max, size=(self.n_pop, self.n_params))
        return mean_vector

    def generate_mutated_offspring(self, population_size):
        """
        Generates a mutated offspring population based on the current mean vector and mutation sigma.

        :param population_size: Number of individuals in the population.
        :return: A 2D array representing the mutated offspring population.
        """
        # Duplicate the current mean vector along the population dimension
        population = np.tile(self.current_mean, (population_size, 1))

        # Compute multivariate Gaussian noise
        mutation = np.random.normal(
            loc=0.0, scale=1.0, size=(population_size, self.n_params)
        )

        # Compute offspring by adding noise scaled by the mutation sigma
        mutated_population = population + self.current_sigma * mutation

        return mutated_population

    def sort_and_select_parents(self, population, fitness, num_parents):
        """
        Sorts the population by fitness and selects the top parents.

        :param population: A 2D array representing the population.
        :param fitness: A 1D array of fitness values corresponding to the population.
        :param num_parents: Number of parents to select.
        :return: A tuple of (selected parent population, selected parent fitness).
        """
        # Sort indices by fitness in descending order
        sorted_indices = np.argsort(fitness)[::-1]

        # Select the top parents based on the number of parents
        sorted_indices = sorted_indices[:num_parents]

        parent_population = population[sorted_indices]
        parent_fitness = fitness[sorted_indices]

        return parent_population, parent_fitness

    def update_population_mean(self, parent_population, parent_fitness):
        """
        Updates the mean vector of the population based on the fitness-weighted average of the parents.

        :param parent_population: A 2D array representing the selected parent population.
        :param parent_fitness: A 1D array of fitness values corresponding to the parents.
        :return: The updated mean vector.
        """
        # Normalize parent fitness values to be non-negative
        min_parent_fitness = np.min(parent_fitness)
        max_parent_fitness = np.max(parent_fitness)
        normed_parents_fitness = (parent_fitness - min_parent_fitness) / (
            max_parent_fitness - min_parent_fitness
        )

        # Normalize weights to sum to 1
        weights = normed_parents_fitness / np.sum(normed_parents_fitness)

        # Compute the weighted average of the parent population
        updated_mean_vector = np.average(parent_population, axis=0, weights=weights)

        return updated_mean_vector

    def update_sigma(self):
        """
        Updates the mutation step size (sigma) for the evolutionary strategy.
        Ensures sigma does not fall below a predefined minimum value.
        """
        # Gradually reduce sigma over generations
        decay_rate = 0.95  # Example decay rate (adjustable)
        sigma = self.current_sigma * decay_rate

        # Ensure sigma does not fall below the minimum threshold
        sigma = max(sigma, self.sigma_limit)

        return sigma

    def save_checkpoint(self):
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        os.makedirs(curr_gen_path, exist_ok=True)
        np.save(os.path.join(self.directory_name, 'full_f'), np.array(self.full_fitness))
        np.save(os.path.join(self.directory_name, 'full_x'), np.array(self.full_x))
        np.save(os.path.join(self.directory_name, 'mean_positions'), np.array(self.mean_positions))  # Save mean positions
        np.save(os.path.join(curr_gen_path, 'f_best'), np.array(self.f_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x_best'), np.array(self.x_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x'), np.array(self.x))
        np.save(os.path.join(curr_gen_path, 'f'), np.array(self.f))

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, 'f_best.npy')
        assert len(dir_path) > 0;
        "No files are here, check the directory_name!!"

        self.current_gen = int(dir_path[-1].split('/')[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))

        self.full_fitness = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.mean_positions = np.load(os.path.join(self.directory_name, 'mean_positions.npy')).tolist()  # Load mean positions
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))

