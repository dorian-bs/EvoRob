from src.EA.CMAES import CMAES, CMAES_opts
from src.world.World import World
from src.world.robot.morphology.PassiveWalkerRobot import PassiveWalkerRobot
from src.utils.Filesys import get_project_root
import xml.etree.ElementTree as xml

import gymnasium as gym
import numpy as np
import os

""" Large programming projects are often modularised in different components. 
    In the upcoming exercise(s) we will (re)build an evolutionary pipeline for robot evolution in MuJoCo.

    Exercise2 Morphology evolutions: In this exercise we will focus on morphology design of a Passive-Dynamic Walker.
    The evolutionary optimisation is focused on the optimisation of leg lengths of this non-actuated system.
"""

ROOT_DIR = get_project_root()
ENV_NAME = "PassiveWalker-v0"


class PassiveWalkerWorld(World):
    def __init__(self, ):
        self.n_params = 6
        self.world_file = os.path.join(ROOT_DIR, "PassiveWalkerEnv.xml")
        self.slope_height = np.sin(5 * np.pi / 180) * 5
        self.env = gym.make(
            ENV_NAME,
            robot_path=self.world_file,
            init_z_offset=self.slope_height, )

        self.joint_limits = [[-45, 45], [-150, 0], [-45, 45], [-150, 0], ]

    def geno2pheno(self, genotype):
        # TODO Improve the genotype to phenotype mapping
        assert len(genotype) == 6
        right_up_leg, right_low_leg, right_foot, left_up_leg, left_low_leg, left_foot = genotype

        # Define the 3D coordinates of the relative tree structure
        right_hip_xyz   = np.array([0         ,-0.05    , 0            ])
        right_knee_xyz  = np.array([0         , 0       ,-right_up_leg ]) + right_hip_xyz
        right_ankle_xyz = np.array([0         , 0       ,-right_low_leg]) + right_knee_xyz
        right_toe1_xyz  = np.array([right_foot,-0.025   , 0            ]) + right_ankle_xyz
        right_toe2_xyz  = np.array([0         , 0.06    , 0            ]) + right_toe1_xyz

        left_hip_xyz    = np.array([0         , 0.05    , 0            ])
        left_knee_xyz   = np.array([0         , 0       ,-left_up_leg  ]) + left_hip_xyz
        left_ankle_xyz  = np.array([0         , 0       ,-left_low_leg ]) + left_knee_xyz
        left_toe1_xyz   = np.array([left_foot , 0.025   , 0            ]) + left_ankle_xyz
        left_toe2_xyz   = np.array([0         ,-0.06    , 0            ]) + left_toe1_xyz

        points = np.vstack([right_hip_xyz, right_knee_xyz, right_ankle_xyz, right_toe1_xyz, right_toe2_xyz,
                            left_hip_xyz, left_knee_xyz, left_ankle_xyz, left_toe1_xyz, left_toe2_xyz, ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[1, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, np.inf, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, np.inf, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, np.inf],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        return points, connectivity_mat

    def evaluate_individual(self, genotype):
        n_sim_steps = 2000

        points, connectivity_mat = self.geno2pheno(genotype)
        robot = PassiveWalkerRobot(points, connectivity_mat, self.joint_limits, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml(ROOT_DIR)  # Save XML to the EvoRob folder instead of default locationOOT_DIR)

        # % Defining the Robot environment in MuJoCo #TODO
        world = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "walker_world.xml"))
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "PassiveWalkerRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding='unicode')
        with open(self.world_file, "w") as f:
            f.write(world_xml)

        self.env = gym.make(
            ENV_NAME,
            robot_path=self.world_file,
            init_z_offset=self.slope_height, )

        self.env.reset()
        actions = []
        rewards_list = []
        for step in range(n_sim_steps):
            observations, rewards, terminated, truncated, info = self.env.step(actions)
            rewards_list.append(rewards)
            if terminated:
                break
        self.env.close()
        return observations[0]


def run_EA(ea, world):
    for gen in range(ea.n_gen):
        pop = ea.ask()
        fitnesses_gen = np.empty(ea.n_pop)
        for index, genotype in enumerate(pop):
            fit_ind = world.evaluate_individual(genotype)
            fitnesses_gen[index] = fit_ind
        ea.tell(pop, fitnesses_gen)


def generate_best_individual_video(world, video_name: str = 'EvoRob2_video.mp4'):
    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   init_z_offset=world.slope_height,
                   render_mode="rgb_array")
    rewards_list = []
    observations, info = env.reset()
    frames = []
    actions = []
    for step in range(2000):
        frames.append(env.render())
        observations, rewards, terminated, truncated, info = env.step(actions)
        rewards_list.append(rewards)
        if terminated:
            break
    print(np.sum(rewards_list))

    import imageio
    imageio.mimsave(video_name, frames, fps=30)  # Set frames per second (fps)
    env.close()


def visualise_individual(genotype):
    world = PassiveWalkerWorld()
    points, connectivity_mat = world.geno2pheno(genotype)
    robot = PassiveWalkerRobot(points, connectivity_mat, world.joint_limits, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml(ROOT_DIR)  # Save XML to the EvoRob folder instead of default locationOOT_DIR)  # Save XML to the EvoRob folder instead of default location

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "walker_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "PassiveWalkerRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    env = gym.make(ENV_NAME,
                   robot_path=world.world_file,
                   init_z_offset=world.slope_height,
                   render_mode="human")
    rewards_list = []

    observations, info = env.reset()
    action = []
    for step in range(1000):
        observations, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        if terminated:
            break
    env.close()
    print(np.sum(rewards_list))


def main():
    # %% Understanding the world
    genotype = [0.3, 0.2, 0.1,
                0.3, 0.2, 0.1]
    visualise_individual(genotype)

    # %% Defining environment
    world = PassiveWalkerWorld()
    n_parameters = world.n_params

    results_dir = os.path.join(ROOT_DIR, 'results', ENV_NAME, 'CMAES')

    CMAES_opts["min"] = 0
    CMAES_opts["max"] = 3
    CMAES_opts["num_parents"] = 20
    CMAES_opts["num_generations"] = 40
    CMAES_opts["mutation_sigma"] = 10

    population_size = 100

    ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir)

    # %% Optimise
    run_EA(ea, world)

    # %% visualise
    # TODO: Make a video of the best individual, and plot the fitness curve.
    best_individual = np.load(os.path.join(results_dir, f"{CMAES_opts["num_generations"]-1}", "x_best.npy"))

    points, connectivity_mat = world.geno2pheno(best_individual)
    robot = PassiveWalkerRobot(points, connectivity_mat, world.joint_limits, verbose=False)
    robot.xml = robot.define_robot()
    robot.write_xml()

    # % Defining the Robot environment in MuJoCo
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "walker_world.xml"))
    robot_env = world_xml.getroot()

    robot_env.append(xml.Element("include", attrib={"file": "PassiveWalkerRobot.xml"}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    generate_best_individual_video(world)


if __name__ == '__main__':
    main()
# %%
