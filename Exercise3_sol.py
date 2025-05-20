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
        # Create a directory for XML files if it doesn't exist
        self.xml_dir = os.path.join(ROOT_DIR, "xml_files")
        os.makedirs(self.xml_dir, exist_ok=True)
        # Base filename that will be made unique for each evaluation
        self.world_file_base = os.path.join(self.xml_dir, "AntEnv")

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
        control_weights = genotype[-self.n_weights:]
        body_params = (genotype[:-self.n_weights] + 1.5) / 5 * 0.5 + 0.1
        assert len(body_params) == 8
        assert len(control_weights) == self.n_weights
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle, = body_params

        # Define the 3D coordinates of the relative tree structure
        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array(
            [np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz = np.array(
            [np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array(
            [-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz = np.array(
            [-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array(
            [-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz = np.array(
            [-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array(
            [np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz = np.array(
            [np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat

    def evaluate_individual(self, genotype):
        # Generate a unique filename for this evaluation using timestamp and process ID
        import time
        import os
        unique_id = f"{time.time()}_{os.getpid()}"
        robot_filename = f"AntRobot_{unique_id}.xml"
        robot_xml_path = os.path.join(self.xml_dir, robot_filename)
        self.world_file = os.path.join(self.xml_dir, f"AntEnv_{unique_id}.xml")
        
        points, connectivity_mat = self.geno2pheno(genotype)

        robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False, name=f"AntRobot_{unique_id}")
        robot.xml = robot.define_robot()
        
        # Write the robot XML to the specific path we've generated
        # The AntRobot.write_xml() method doesn't accept a 'name' parameter, only 'directory'
        robot.write_xml(directory=self.xml_dir)
        
        # Rename the file if necessary (if it doesn't use the name we set)
        default_path = os.path.join(self.xml_dir, f"{robot.name}.xml")
        if os.path.exists(default_path) and default_path != robot_xml_path:
            os.rename(default_path, robot_xml_path)
        
        # % Defining the Robot environment in MuJoCo
        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
        robot_env = world.getroot()

        # Use the full path to the robot XML file
        robot_env.append(xml.Element("include", attrib={"file": robot_xml_path}))
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

            multi_obj_reward = np.array([infos['reward_forward'], -infos['ctrl_cost']]).T  # TODO
            multi_obj_rewards_full[step, done_mask == False] = multi_obj_reward[done_mask == False]

            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


# Move evaluation functions outside to make them picklable
def evaluate_single_individual(idx_genotype):
    idx, genotype, world = idx_genotype  # Correctly unpack all three elements
    fit_ind, _ = world.evaluate_individual(genotype)
    return idx, fit_ind

def evaluate_multi_individual(idx_genotype):
    idx, genotype, world = idx_genotype  # Correctly unpack all three elements
    _, fit_ind = world.evaluate_individual(genotype)
    return idx, fit_ind

def run_EA_single(ea_single, world):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    # Determine optimal number of workers based on CPU count
    max_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Starting single-objective optimization with {ea_single.n_gen} generations using {max_workers} parallel workers...")
    
    for gen in range(ea_single.n_gen):
        print(f"Generation {gen+1}/{ea_single.n_gen}...")
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        
        # Create index-genotype-world tuples for tracking results (include world in each tuple)
        indexed_pop = [(i, genotype, world) for i, genotype in enumerate(pop)]
        
        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(evaluate_single_individual, indexed_pop))
            
            # Process results
            for idx, fitness in results:
                fitnesses_gen[idx] = fitness
                
        ea_single.tell(pop, fitnesses_gen)
        print(f"Best fitness in generation {gen+1}: {ea_single.f_best_so_far}")


def run_EA_multi(ea_multi, world):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    # Determine optimal number of workers based on CPU count
    max_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Starting multi-objective optimization with {ea_multi.n_gen} generations using {max_workers} parallel workers...")
    
    for gen in range(ea_multi.n_gen):
        print(f"Generation {gen+1}/{ea_multi.n_gen}...")
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        
        # Create index-genotype-world tuples for tracking results (include world in each tuple)
        indexed_pop = [(i, genotype, world) for i, genotype in enumerate(pop)]
        
        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(evaluate_multi_individual, indexed_pop))
            
            # Process results
            for idx, fitness in results:
                fitnesses_gen[idx] = fitness
                
        ea_multi.tell(pop, fitnesses_gen)
        
        # Print generation info with fitness values
        print(f"Generation {gen+1} completed with best fitness: {ea_multi.f_best_so_far}")
        print(f"Mean fitness: {np.mean(fitnesses_gen, axis=0)}")
        print(f"Min fitness: {np.min(fitnesses_gen, axis=0)}")
        print(f"Max fitness: {np.max(fitnesses_gen, axis=0)}")


def generate_best_individual_video(world, video_name: str = 'EvoRob3_video.mp4'):
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   render_mode="rgb_array")
    rewards_list = []

    observations, info = env.reset()
    frames = []
    for step in range(3000):
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
    # visualise_individual(genotype)

    # %% Optimise single-objective
    world = AntWorld()
    n_parameters = world.n_params

    population_size = 50  # Reduced from 250
    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_parents"] = 10  # Reduced from 100
    CMAES_opts["num_generations"] = 1  # Reduced from 100
    CMAES_opts["mutation_sigma"] = 0.6

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'single')
    ea_single = CMAES(population_size, n_parameters, CMAES_opts, results_dir)

    run_EA_single(ea_single, world)

    # %% Optimise multi-objective
    # TODO implement the NSGAII
    world = AntWorld()
    n_parameters = world.n_params

    population_size = 50  # Reduced from 250
    NSGA_opts["min"] = -1
    NSGA_opts["max"] = 1
    NSGA_opts["num_parents"] = population_size
    NSGA_opts["num_generations"] = 1  # Reduced from 100
    NSGA_opts["mutation_prob"] = 0.3
    NSGA_opts["crossover_prob"] = 0.5

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'multi')
    ea_multi_obj = NSGAII(population_size, n_parameters, NSGA_opts, results_dir)

    run_EA_multi(ea_multi_obj, world)

    # %% visualise
    # Get the last generation number (0-indexed)
    last_gen = NSGA_opts["num_generations"] - 1
    
    # Try to load the best individual file with error handling
    try:
        best_individual_path = os.path.join(results_dir, f"{last_gen}", "x_best.npy")
        print(f"Attempting to load best individual from: {best_individual_path}")
        
        if os.path.exists(best_individual_path):
            best_individual = np.load(best_individual_path)
            
            # Ensure world_file is properly set before generating the video
            # Create a unique filename for this video
            import time
            unique_id = f"video_{time.time()}"
            robot_xml_path = os.path.join(world.xml_dir, f"AntRobot_{unique_id}.xml")
            world.world_file = os.path.join(world.xml_dir, f"AntEnv_{unique_id}.xml")
            
            points, connectivity_mat = world.geno2pheno(best_individual)
            robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, verbose=False, name=f"AntRobot_{unique_id}")
            robot.xml = robot.define_robot()
            
            # Write the robot XML file
            robot.write_xml(directory=world.xml_dir)
            
            # Rename if needed
            default_path = os.path.join(world.xml_dir, f"{robot.name}.xml")
            if os.path.exists(default_path) and default_path != robot_xml_path:
                os.rename(default_path, robot_xml_path)

            # Defining the Robot environment in MuJoCo
            world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
            robot_env = world_xml.getroot()

            robot_env.append(xml.Element("include", attrib={"file": robot_xml_path}))
            world_xml = xml.tostring(robot_env, encoding='unicode')
            with open(world.world_file, "w") as f:
                f.write(world_xml)

            generate_best_individual_video(world)
        else:
            print(f"Best individual file not found at {best_individual_path}")
            print("Skipping video generation")
    except Exception as e:
        print(f"Error loading or processing best individual: {str(e)}")
        print("Skipping video generation")

if __name__ == '__main__':
    main()