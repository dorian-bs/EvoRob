from src.EA.NSGA import NSGAII, NSGA_opts
import numpy as np
import os
from src.utils.Filesys import get_project_root

class SimpleTestWorld:
    def __init__(self):
        self.n_params = 10
        
    def evaluate_individual(self, genotype):
        # Simple multi-objective function:
        # f1 = sum of parameters (maximize)
        # f2 = negative variance of parameters (maximize)
        f1 = np.sum(genotype)
        f2 = -np.var(genotype)
        return f1, np.array([f1, f2])
    
    def geno2pheno(self, genotype):
        return genotype  # Identity mapping for this simple test

def run_test():
    # Initialize test world
    world = SimpleTestWorld()
    
    # Set up NSGA-II parameters
    population_size = 50
    NSGA_opts["min"] = -5
    NSGA_opts["max"] = 5
    NSGA_opts["num_parents"] = 20
    NSGA_opts["num_generations"] = 10
    
    # Set up output directory
    ROOT_DIR = get_project_root()
    results_dir = os.path.join(ROOT_DIR, 'results', 'test_nsga')
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize NSGA-II
    ea = NSGAII(population_size, world.n_params, NSGA_opts, results_dir)
    
    # Run evolution
    for gen in range(ea.n_gen):
        print(f"Generation {gen}...")
        pop = ea.ask()
        
        # Evaluate each individual
        fitnesses = np.empty((len(pop), 2))
        for i, genotype in enumerate(pop):
            _, fit = world.evaluate_individual(genotype)
            fitnesses[i] = fit
        
        # Update algorithm with results
        ea.tell(pop, fitnesses)
    
    print("Optimization complete!")
    print(f"Best solution: {ea.x_best_so_far}")
    print(f"Best fitness: {ea.f_best_so_far}")

if __name__ == "__main__":
    run_test()
