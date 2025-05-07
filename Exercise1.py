from src.EA.CMAES import CMAES, CMAES_opts
from src.world.robot.controllers import MLP
from src.utils.Filesys import get_project_root
from src.world.World import World

from stable_baselines3.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

import torch
import gymnasium as gym
import numpy as np
import os
import time

ROOT_DIR = get_project_root()
ENV_NAME = 'HalfCheetah-v5'


class PPO_controller():
    def __init__(self, ppo: PPO):
        self.ppo = ppo
        self.state_space = ppo.observation_space
        self.action_space = ppo.action_space

    def get_action(self, state):
        state_tensor = torch.tensor(state[np.newaxis, :])
        action = self.ppo.policy(state_tensor)[0].squeeze().detach()
        return action.numpy()


class CheetahWorld(World):
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        action_space = self.env.action_space.shape[0]  # https://gymnasium.farama.org/environments/mujoco/half_cheetah/#action-space
        state_space = self.env.observation_space.shape[0]  # https://gymnasium.farama.org/environments/mujoco/half_cheetah/#observation-space
        self.controller = MLP.NNController(state_space, action_space)
        self.dt = self.env.get_wrapper_attr('dt')
        self.n_params = self.controller.n_params

    def geno2pheno(self, genotype):
        self.controller.geno2pheno(genotype)
        return self.controller

    def evaluate_individual(self, genotype):
        trial_time = 50  # seconds in simulation
        n_sim_steps = int(trial_time / self.dt)

        self.geno2pheno(genotype)

        rewards_list = []
        observations, info = self.env.reset(seed=42)
        for step in range(n_sim_steps):
            action = self.controller.get_action(observations)
            observations, rewards, terminated, truncated, info = self.env.step(action)
            rewards_list.append(rewards)
        return np.sum(rewards_list)


def run_EA(ea, world):
    env = gym.make(ENV_NAME)
    for gen in range(ea.n_gen):
        pop = ea.ask()
        fitnesses_gen = np.empty(ea.n_pop)
        # Use a different seed for each generation to improve robustness
        random_seed = np.random.randint(0, 1000)
        env.reset(seed=random_seed)
        for index, genotype in enumerate(pop):
            fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea.tell(pop, fitnesses_gen)
    env.close()


def generate_best_individual_video(controller, video_name: str = 'EvoRob1_video.mp4'):
    # TODO: Make a video of the best individual, and plot the fitness curve.
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    rewards_list = []
    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()


def make_env(env_name, rank=0, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_name: (str) Name of the environment
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :return: (Callable)
    """
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed + rank)  # Set different seeds for each environment
        return env
    set_random_seed(seed)
    return _init


def train_accelerated_ppo(env_name, total_steps, verbose=1):
    """
    Train PPO using acceleration techniques (vectorized environments and optimized parameters).
    
    :param env_name: Name of the gym environment
    :param total_steps: Total timesteps to train
    :param verbose: Verbosity level
    :return: Trained PPO model, training time, controller, vectorized environment
    """
    start_time = time.time()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training accelerated PPO using {device}")
    
    # Use multiple parallel environments to speed up data collection
    n_envs = min(8, os.cpu_count() or 1)
    print(f"Creating {n_envs} parallel environments")
    
    # Create vectorized environment with proper seeding
    vec_env = SubprocVecEnv([make_env(env_name, i, seed=42) for i in range(n_envs)])
    
    # Apply observation normalization for better training stability
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # Adjusted steps for parallel environments
    adjusted_steps = total_steps // n_envs
    print(f"Training with {adjusted_steps} steps ({total_steps} total environment interactions)")
    
    # Create PPO with optimized hyperparameters
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        device=device, 
        verbose=verbose,
        # Optimized hyperparameters for faster training
        n_steps=1024,       # Larger batch of steps for more stable gradients
        batch_size=64,      # Smaller batch size for faster updates
        n_epochs=10,        # More epochs for better learning with fewer steps
        learning_rate=3e-4, # Slightly higher learning rate for faster convergence
        gamma=0.99,         # Discount factor
        gae_lambda=0.95,    # GAE lambda parameter
        clip_range=0.2,     # Clip parameter for PPO
        ent_coef=0.0,       # No entropy coefficient for faster convergence
        # Use larger networks since we're training faster
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
    )
    
    # Train the model
    model.learn(total_timesteps=adjusted_steps)
    training_time = time.time() - start_time
    
    # Create and return controller
    controller = PPO_controller(model)
    return model, training_time, controller, vec_env


def main():
    # TODO: understanding the world
    world = CheetahWorld()
    n_parameters = world.n_params

    # TODO: improve the ES settings
    CMAES_opts["min"] = -15
    CMAES_opts["max"] = 15
    CMAES_opts["num_parents"] = 100
    CMAES_opts["num_generations"] = 100
    CMAES_opts["mutation_sigma"] = 5

    population_size = 50

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'CMAES')
    ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir)

    start_time = time.time()
    run_EA(ea, world)
    print(f"Optimisation time : {(time.time() - start_time):.2f} [s]")

    # %% Make video of best behaviour
    best_individual = np.load(os.path.join(results_dir, f"{CMAES_opts['num_generations']-1}", "x_best.npy"))
    world.controller.geno2pheno(best_individual)

    generate_best_individual_video(world.controller, 'CMAES_best.mp4')
    
    # Test best individual against different seeds
    print("\nTesting best individual across multiple seeds:")
    num_seeds = 10
    rewards_across_seeds = []
    test_env = gym.make(ENV_NAME)
    trial_time = 50  # seconds in simulation
    n_sim_steps = int(trial_time / world.dt)
    
    for seed in range(num_seeds):
        rewards_list = []
        observations, info = test_env.reset(seed=seed)
        for step in range(n_sim_steps):
            action = world.controller.get_action(observations)
            observations, rewards, terminated, truncated, info = test_env.step(action)
            rewards_list.append(rewards)
            if terminated:
                break
        total_reward = np.sum(rewards_list)
        rewards_across_seeds.append(total_reward)
        print(f"Seed {seed}: Reward = {total_reward:.2f}")
    
    mean_reward = np.mean(rewards_across_seeds)
    std_reward = np.std(rewards_across_seeds)
    print(f"Performance across {num_seeds} seeds - Mean: {mean_reward:.2f}, Std: {std_reward:.2f}")
    test_env.close()
    
    ea_mean_reward = mean_reward
    ea_std_reward = std_reward

    # %% Compare with PPO
    print("\n=== Training Accelerated PPO ===")
    
    # Calculate the same number of steps as used by EA for fair comparison
    trial_time = 50  # seconds in simulation
    n_sim_steps = int(trial_time / world.dt)
    n_total_steps = population_size * CMAES_opts["num_generations"] * n_sim_steps
    
    try:
        # Train accelerated PPO
        ppo, ppo_training_time, ppo_controller, vec_env = train_accelerated_ppo(
            ENV_NAME, 
            n_total_steps
        )
        print(f"PPO training time: {ppo_training_time:.2f} seconds ({ppo_training_time/60:.2f} minutes)")
        
        # Test PPO across multiple seeds for consistent comparison
        print("\nTesting PPO across multiple seeds:")
        ppo_rewards_across_seeds = []
        test_env = gym.make(ENV_NAME)
        
        for seed in range(num_seeds):
            rewards_list = []
            observations, info = test_env.reset(seed=seed)
            for step in range(n_sim_steps):
                action = ppo_controller.get_action(observations)
                observations, rewards, terminated, truncated, info = test_env.step(action)
                rewards_list.append(rewards)
                if terminated:
                    break
            total_reward = np.sum(rewards_list)
            ppo_rewards_across_seeds.append(total_reward)
            print(f"Seed {seed}: Reward = {total_reward:.2f}")
        
        ppo_mean_reward = np.mean(ppo_rewards_across_seeds)
        ppo_std_reward = np.std(ppo_rewards_across_seeds)
        print(f"PPO performance across {num_seeds} seeds - Mean: {ppo_mean_reward:.2f}, Std: {ppo_std_reward:.2f}")
        test_env.close()
        
        # Generate video for PPO
        generate_best_individual_video(ppo_controller, 'PPO_best.mp4')
        
        # Print comparison summary
        print("\n=== EA vs PPO Comparison Summary ===")
        print(f"Environment: {ENV_NAME}")
        print(f"Total environment steps: {n_total_steps}")
        print(f"EA performance - Mean reward: {ea_mean_reward:.2f}, Std: {ea_std_reward:.2f}")
        print(f"PPO performance - Mean reward: {ppo_mean_reward:.2f}, Std: {ppo_std_reward:.2f}")
        print(f"Training times - EA: {(time.time() - start_time):.2f}s, PPO: {ppo_training_time:.2f}s")
        print(f"Relative performance (EA/PPO): {ea_mean_reward/ppo_mean_reward:.2f}x")
        
        # Close environment
        vec_env.close()
    except Exception as e:
        print(f"Error during PPO training: {e}")


if __name__ == '__main__':
    main()
