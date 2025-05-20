from src.EA.CMAES import CMAES, CMAES_opts
from src.EA.NSGA import NSGAII, NSGA_opts
from src.world.World import World
from src.world.robot.controllers import MLP
from src.world.robot.morphology.AntCustomRobot import AntRobot
from src.utils.Filesys import get_project_root
from gymnasium.vector import AsyncVectorEnv

import xml.etree.ElementTree as xml
import gymnasium as gym
import numpy as np
import os

""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.

    Exercise3 body-brain: This is your first full body+brain evolution by adapting a custom Ant-v5 gym environment. 
    We adjust both leg lengths and controller weights for a locomotion task.   
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'


class AntWorld(World):
    def __init__(self, ):
        action_space = 8  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 27  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space

        self.n_repeats = 3
        self.n_steps = 1000
        self.controller = MLP.NNController(state_space, action_space)
        self.n_weights = self.controller.n_params

        self.n_params = self.n_weights + 8
        self.world_file = os.path.join(ROOT_DIR, "AntEnv.xml")

        self.joint_limits = [[-30, 30], [30, 70],
                             [-30, 30], [-70, -30],
                             [-30, 30], [-70, -30],
                             [-30, 30], [30, 70], ]
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           [0, 0, 1], [-1, 1, 0],
                           [0, 0, 1], [1, 1, 0],
                           ]

    def geno2pheno(self, genotype):
        """
        Converts genotype values to phenotype parameters (morphology and controller).
        
        The genotype is split into two parts:
        - Body parameters: 8 parameters for leg segment lengths
        - Control weights: Parameters for the neural network controller
        
        Each leg has two segments:
        - Upper segment (leg): Controls the length from hip to knee
        - Lower segment (ankle): Controls the length from knee to toe
        
        Genotype is mapped from [-1, 1] to produce leg lengths in range [0.1, 0.35].
        """
        # Split genotype into control weights and body parameters
        control_weights = genotype[-self.n_weights:]
        
        # Map body parameters from [-1, 1] to a reasonable range for leg lengths [0.1, 0.35]
        # Add flexibility to allow slightly different ranges for different leg parts
        leg_params_raw = genotype[:-self.n_weights]
        
        # Create different scaling factors for upper and lower leg segments
        # This allows more diversity in morphology
        upper_leg_scale = 0.12  # Range will be [0.1, 0.34]
        lower_leg_scale = 0.10  # Range will be [0.1, 0.30]
        
        # Apply different scaling to different leg parts
        body_params = np.zeros(8)
        # Upper leg segments (even indices)
        body_params[0::2] = (leg_params_raw[0::2] + 1) * upper_leg_scale + 0.1
        # Lower leg segments (odd indices)
        body_params[1::2] = (leg_params_raw[1::2] + 1) * lower_leg_scale + 0.1
        
        # Validate parameters
        assert len(body_params) == 8, "Expected 8 body parameters"
        assert len(control_weights) == self.n_weights, f"Expected {self.n_weights} control weights"
        assert not np.any(body_params <= 0), "All leg segments must have positive length"
        
        # Configure neural network weights
        self.controller.geno2pheno(control_weights)

        # Unpack body parameters
        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, \
        back_left_leg, back_left_ankle, back_right_leg, back_right_ankle = body_params
        
        # Calculate leg positions with more natural alignment
        # Hip positions are fixed at the corners of a square
        hip_offset = 0.2  # Distance from center to hip joint
        
        # Front Left Leg
        front_left_hip_xyz = np.array([hip_offset, hip_offset, 0])
        # Direction vector for front left leg (45 degrees)
        front_left_dir = np.array([1, 1, 0]) / np.sqrt(2)
        front_left_knee_xyz = front_left_hip_xyz + front_left_dir * front_left_leg
        front_left_toe_xyz = front_left_knee_xyz + front_left_dir * front_left_ankle
        
        # Front Right Leg
        front_right_hip_xyz = np.array([-hip_offset, hip_offset, 0])
        # Direction vector for front right leg (135 degrees)
        front_right_dir = np.array([-1, 1, 0]) / np.sqrt(2)
        front_right_knee_xyz = front_right_hip_xyz + front_right_dir * front_right_leg
        front_right_toe_xyz = front_right_knee_xyz + front_right_dir * front_right_ankle
        
        # Back Left Leg
        back_left_hip_xyz = np.array([-hip_offset, -hip_offset, 0])
        # Direction vector for back left leg (225 degrees)
        back_left_dir = np.array([-1, -1, 0]) / np.sqrt(2)
        back_left_knee_xyz = back_left_hip_xyz + back_left_dir * back_left_leg
        back_left_toe_xyz = back_left_knee_xyz + back_left_dir * back_left_ankle
        
        # Back Right Leg
        back_right_hip_xyz = np.array([hip_offset, -hip_offset, 0])
        # Direction vector for back right leg (315 degrees)
        back_right_dir = np.array([1, -1, 0]) / np.sqrt(2)
        back_right_knee_xyz = back_right_hip_xyz + back_right_dir * back_right_leg
        back_right_toe_xyz = back_right_knee_xyz + back_right_dir * back_right_ankle
        
        # Stack all points for the robot geometry
        points = np.vstack([
            front_left_hip_xyz, front_left_knee_xyz, front_left_toe_xyz,
            front_right_hip_xyz, front_right_knee_xyz, front_right_toe_xyz,
            back_left_hip_xyz, back_left_knee_xyz, back_left_toe_xyz,
            back_right_hip_xyz, back_right_knee_xyz, back_right_toe_xyz,
        ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat

    def evaluate_individual(self, genotype):
        points, connectivity_mat = self.geno2pheno(genotype)

        robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml()

        # % Defining the Robot environment in MuJoCo
        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding='unicode')

        with open(self.world_file, "w") as f:
            f.write(world_xml)

        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=0.1,
                    max_episode_steps=self.n_steps,
                )
                for i_env in range(self.n_repeats)
            ]
        )

        rewards_full = np.zeros((self.n_steps, self.n_repeats))
        multi_obj_rewards_full = np.zeros((self.n_steps, self.n_repeats, 2))  # TODO

        observations, info = envs.reset()
        done_mask = np.zeros(self.n_repeats, dtype=bool)
        for step in range(self.n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations.T).T)
            observations, rewards, dones, truncated, infos = envs.step(actions)

            # Store rewards for active environments only
            rewards_full[step, done_mask == False] = rewards[done_mask == False]

            # multi_obj_reward = np.array([infos[...], -infos[...]]).T  # TODO
            # multi_obj_rewards_full[step, done_mask == False] = multi_obj_reward[done_mask == False]

            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


def run_EA_single(ea_single, world):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    # Determine optimal number of workers based on CPU count
    max_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Starting single-objective optimization with {ea_single.n_gen} generations using {max_workers} parallel workers...")
    
    # Helper function for parallel evaluation
    def evaluate_single_individual(idx_genotype):
        idx, genotype = idx_genotype
        fit_ind, _ = world.evaluate_individual(genotype)
        return idx, fit_ind
    
    for gen in range(ea_single.n_gen):
        print(f"Generation {gen+1}/{ea_single.n_gen}...")
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        
        # Create index-genotype pairs for tracking results
        indexed_pop = list(enumerate(pop))
        
        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(evaluate_single_individual, indexed_pop))
            
            # Process results in the order they complete
            completed = 0
            for idx, fitness in results:
                fitnesses_gen[idx] = fitness
                completed += 1
                if completed % 10 == 0 or completed == len(pop):
                    print(f"  Evaluated {completed}/{len(pop)} individuals")
        
        ea_single.tell(pop, fitnesses_gen)
        print(f"  Best fitness in generation {gen+1}: {ea_single.f_best_so_far}")
    
    print(f"Single-objective optimization completed!")
    print(f"Best overall fitness: {ea_single.f_best_so_far}")


def run_EA_multi(ea_multi, world):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    # Determine optimal number of workers based on CPU count
    max_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Starting multi-objective optimization with {ea_multi.n_gen} generations using {max_workers} parallel workers...")
    
    # Helper function for parallel evaluation
    def evaluate_multi_individual(idx_genotype):
        idx, genotype = idx_genotype
        _, fit_ind = world.evaluate_individual(genotype)
        return idx, fit_ind
    
    for gen in range(ea_multi.n_gen):
        print(f"Generation {gen+1}/{ea_multi.n_gen}...")
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        
        # Create index-genotype pairs for tracking results
        indexed_pop = list(enumerate(pop))
        
        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(evaluate_multi_individual, indexed_pop))
            
            # Process results in the order they complete
            completed = 0
            for idx, fitness in results:
                fitnesses_gen[idx] = fitness
                completed += 1
                if completed % 10 == 0 or completed == len(pop):
                    print(f"  Evaluated {completed}/{len(pop)} individuals")
        
        ea_multi.tell(pop, fitnesses_gen)
        print(f"  Number of solutions in Pareto front: {len(ea_multi.pareto_front)}")
    
    print(f"Multi-objective optimization completed!")
    print(f"Number of solutions in final Pareto front: {len(ea_multi.pareto_front)}")


def generate_best_individual_video(world, video_name: str = 'EvoRob3_video.mp4'):
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")
    rewards_list = []

    observations, info = env.reset()
    frames = []
    for step in range(1000):
        frames.append(env.render())
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()


def visualise_individual(genotype):
    world = AntWorld()
    points, connectivity_mat = world.geno2pheno(genotype)
    robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="human")
    rewards_list = []

    observations, info = env.reset()
    for step in range(1000):
        action = world.controller.get_action(observations)
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    env.close()
    print(np.sum(rewards_list))


def main():
    # %% Understanding the world
    genotype = np.random.uniform(-1, 1, 953)  # 8 body parameters, 945 NN weights
    visualise_individual(genotype)

    # %% Optimise single-objective
    world = AntWorld()
    n_parameters = world.n_params

    population_size = 250
    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_parents"] = 100
    CMAES_opts["num_generations"] = 100
    CMAES_opts["mutation_sigma"] = 0.33

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single')
    ea_single = CMAES(population_size, n_parameters, CMAES_opts, results_dir)

    run_EA_single(ea_single, world)

    # %% Optimise multi-objective
    # TODO implement NSGAII
    world = AntWorld()
    n_parameters = world.n_params

    population_size = 250
    NSGA_opts["min"] = -1
    NSGA_opts["max"] = 1
    NSGA_opts["num_parents"] = population_size
    NSGA_opts["num_generations"] = 100
    NSGA_opts["mutation_prob"] = 0.3
    NSGA_opts["crossover_prob"] = 0.5

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'multi')
    ea_multi_obj = NSGAII(population_size, n_parameters, NSGA_opts, results_dir)

    run_EA_multi(ea_multi_obj, world)

    # %% visualise
    # TODO: Make a video of the best individual, and plot the fitness curve.
    best_individual = np.load(os.path.join(results_dir, f"{NSGA_opts["num_generations"]-1}", "x_best.npy"))

    points, connectivity_mat = world.geno2pheno(best_individual)
    robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    generate_best_individual_video(world)


if __name__ == '__main__':
    main()