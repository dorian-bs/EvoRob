"""
Experiment: Effects of Genotype-to-Phenotype Mappings on Robot Evolution

Required packages:
- numpy
- matplotlib
- pandas
- seaborn (optional, for prettier plots)
- gymnasium
- imageio (for video generation)

To install missing packages:
pip install numpy matplotlib pandas seaborn gymnasium imageio
"""

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
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import seaborn as sns
import warnings  # Add warnings module


"""
Experiment: Effects of Genotype-to-Phenotype Mappings on Robot Evolution

This experiment investigates how different genotype-to-phenotype mappings affect
the evolution of both morphology and control for a quadruped robot.

We compare three different mapping approaches:
1. Direct linear mapping (baseline)
2. Nonlinear mapping (sigmoid-based)
3. Symmetry-constrained mapping (enforcing left-right symmetry)

For each mapping, we run evolutionary optimization and analyze:
- Final fitness achieved
- Morphological characteristics
- Behavioral strategies
"""

ROOT_DIR = get_project_root()
ENV_NAME = 'Ant_custom'

class AntWorldBase(World):
    """Base class with common functionality for all AntWorld variants"""
    
    def __init__(self):
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
        
        # For recording morphology statistics
        self.morphology_stats = []
        self.mapping_name = "base"  # Will be overridden in subclasses
    
    def evaluate_individual(self, genotype):
        # Generate a unique filename for this evaluation using timestamp and process ID
        import time
        import os
        unique_id = f"{time.time()}_{os.getpid()}"
        robot_xml_path = os.path.join(self.xml_dir, f"AntRobot_{unique_id}.xml")
        env_xml_path = os.path.join(self.xml_dir, f"AntEnv_{unique_id}.xml")
        self.world_file = env_xml_path  # Use the unique path for world file

        points, connectivity_mat = self.geno2pheno(genotype)

        # Record morphology statistics
        self.record_morphology_stats(points)

        try:
            # Create the robot XML - note we're setting the name here, not during write_xml
            robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, 
                           verbose=False, name=f"AntRobot_{unique_id}")
            robot.xml = robot.define_robot()
            
            # Only pass directory to write_xml - don't include filename parameter
            robot.write_xml(directory=self.xml_dir)
            
            # Verify robot XML file exists
            expected_path = os.path.join(self.xml_dir, f"{robot.name}.xml")
            if not os.path.exists(expected_path):
                raise FileNotFoundError(f"Robot XML file not created at {expected_path}")
            
            # Rename if needed to match our expected path
            if expected_path != robot_xml_path and os.path.exists(expected_path):
                os.rename(expected_path, robot_xml_path)
            
            # Create the world XML that references the robot
            world_tree = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
            robot_env = world_tree.getroot()
            
            # Use absolute path for the robot XML to avoid path issues
            abs_robot_path = os.path.abspath(robot_xml_path)
            include_element = xml.Element("include", attrib={"file": abs_robot_path})
            robot_env.append(include_element)
            
            # Convert to XML string and validate
            world_xml = xml.tostring(robot_env, encoding='unicode')
            
            # Write the environment XML file
            with open(env_xml_path, "w") as f:
                f.write(world_xml)
                
            # Verify environment XML file exists
            if not os.path.exists(env_xml_path):
                raise FileNotFoundError(f"Environment XML file not created at {env_xml_path}")
                
            # Create environments with proper error handling
            try:
                envs = AsyncVectorEnv(
                    [
                        lambda i_env=i_env: gym.make(
                            ENV_NAME,
                            robot_path=env_xml_path,
                            reset_noise_scale=0.1,
                            max_episode_steps=self.n_steps,
                        )
                        for i_env in range(self.n_repeats)
                    ]
                )

                rewards_full = np.zeros((self.n_steps, self.n_repeats))
                multi_obj_rewards_full = np.zeros((self.n_steps, self.n_repeats, 2))

                observations, info = envs.reset()
                done_mask = np.zeros(self.n_repeats, dtype=bool)
                for step in range(self.n_steps):
                    actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations.T).T)
                    observations, rewards, dones, truncated, infos = envs.step(actions)

                    # Store rewards for active environments only
                    rewards_full[step, done_mask == False] = rewards[done_mask == False]

                    multi_obj_reward = np.array([infos['reward_forward'], -infos['ctrl_cost']]).T
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
            
            except Exception as e:
                print(f"Error creating environment: {e}")
                # Return a default poor fitness value when evaluation fails
                return -10.0, np.array([-10.0, -10.0])
                
        except Exception as e:
            print(f"Error in individual evaluation: {e}")
            # Return a default poor fitness value when evaluation fails
            return -10.0, np.array([-10.0, -10.0])

    def record_morphology_stats(self, points):
        """Record statistics about the morphology for later analysis"""
        # Calculate leg lengths
        leg_lengths = []
        for i in range(0, 12, 3):  # Four legs, each with 3 points (hip, knee, toe)
            hip = points[i]
            knee = points[i+1]
            toe = points[i+2]
            
            upper_length = np.linalg.norm(knee - hip)
            lower_length = np.linalg.norm(toe - knee)
            total_length = upper_length + lower_length
            
            leg_lengths.append({
                'upper': upper_length,
                'lower': lower_length,
                'total': total_length,
                'ratio': lower_length / upper_length if upper_length != 0 else 0
            })
        
        # Calculate body size (distance between diagonal hips)
        front_left_hip = points[0]
        back_right_hip = points[9]
        body_diagonal = np.linalg.norm(back_right_hip - front_left_hip)
        
        self.morphology_stats.append({
            'mapping': self.mapping_name,
            'legs': leg_lengths,
            'body_diagonal': body_diagonal,
        })

class AntWorldLinear(AntWorldBase):
    """AntWorld with linear genotype-to-phenotype mapping (baseline)"""
    
    def __init__(self):
        super().__init__()
        self.mapping_name = "linear"
    
    def geno2pheno(self, genotype):
        """
        Linear mapping from genotype to phenotype.
        Maps values directly from [-1,1] to [0.1,0.35] range for morphology.
        """
        # Split genotype into control weights and body parameters
        control_weights = genotype[-self.n_weights:]
        leg_params_raw = genotype[:-self.n_weights]
        
        # Simple linear mapping
        body_params = (leg_params_raw + 1) * 0.125 + 0.1  # Maps [-1,1] to [0.1,0.35]
        
        # Validate parameters
        assert len(body_params) == 8, "Expected 8 body parameters"
        assert len(control_weights) == self.n_weights, f"Expected {self.n_weights} control weights"
        assert not np.any(body_params <= 0), "All leg segments must have positive length"
        
        # Configure neural network weights
        self.controller.geno2pheno(control_weights)

        # Unpack body parameters
        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, \
        back_left_leg, back_left_ankle, back_right_leg, back_right_ankle = body_params
        
        # Calculate leg positions with aligned directions
        hip_offset = 0.2  # Distance from center to hip joint
        
        # Front Left Leg
        front_left_hip_xyz = np.array([hip_offset, hip_offset, 0])
        front_left_dir = np.array([1, 1, 0]) / np.sqrt(2)
        front_left_knee_xyz = front_left_hip_xyz + front_left_dir * front_left_leg
        front_left_toe_xyz = front_left_knee_xyz + front_left_dir * front_left_ankle
        
        # Front Right Leg
        front_right_hip_xyz = np.array([-hip_offset, hip_offset, 0])
        front_right_dir = np.array([-1, 1, 0]) / np.sqrt(2)
        front_right_knee_xyz = front_right_hip_xyz + front_right_dir * front_right_leg
        front_right_toe_xyz = front_right_knee_xyz + front_right_dir * front_right_ankle
        
        # Back Left Leg
        back_left_hip_xyz = np.array([-hip_offset, -hip_offset, 0])
        back_left_dir = np.array([-1, -1, 0]) / np.sqrt(2)
        back_left_knee_xyz = back_left_hip_xyz + back_left_dir * back_left_leg
        back_left_toe_xyz = back_left_knee_xyz + back_left_dir * back_left_ankle
        
        # Back Right Leg
        back_right_hip_xyz = np.array([hip_offset, -hip_offset, 0])
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

        # Define the type of connections
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

class AntWorldNonlinear(AntWorldBase):
    """AntWorld with nonlinear genotype-to-phenotype mapping using sigmoid function"""
    
    def __init__(self):
        super().__init__()
        self.mapping_name = "nonlinear"
    
    def geno2pheno(self, genotype):
        """
        Nonlinear sigmoid-based mapping from genotype to phenotype.
        Maps values using a sigmoid function, giving more precision in the middle of the range.
        """
        # Split genotype into control weights and body parameters
        control_weights = genotype[-self.n_weights:]
        leg_params_raw = genotype[:-self.n_weights]
        
        # Nonlinear mapping using sigmoid function
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x * 2))  # Steeper sigmoid
        body_params = sigmoid(leg_params_raw) * 0.25 + 0.1  # Maps to [0.1,0.35]
        
        # Validate parameters
        assert len(body_params) == 8, "Expected 8 body parameters"
        assert len(control_weights) == self.n_weights, f"Expected {self.n_weights} control weights"
        assert not np.any(body_params <= 0), "All leg segments must have positive length"
        
        # Configure neural network weights
        self.controller.geno2pheno(control_weights)

        # Unpack body parameters
        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, \
        back_left_leg, back_left_ankle, back_right_leg, back_right_ankle = body_params
        
        # Calculate leg positions
        hip_offset = 0.2
        
        # Front Left Leg
        front_left_hip_xyz = np.array([hip_offset, hip_offset, 0])
        front_left_dir = np.array([1, 1, 0]) / np.sqrt(2)
        front_left_knee_xyz = front_left_hip_xyz + front_left_dir * front_left_leg
        front_left_toe_xyz = front_left_knee_xyz + front_left_dir * front_left_ankle
        
        # Front Right Leg
        front_right_hip_xyz = np.array([-hip_offset, hip_offset, 0])
        front_right_dir = np.array([-1, 1, 0]) / np.sqrt(2)
        front_right_knee_xyz = front_right_hip_xyz + front_right_dir * front_right_leg
        front_right_toe_xyz = front_right_knee_xyz + front_right_dir * front_right_ankle
        
        # Back Left Leg
        back_left_hip_xyz = np.array([-hip_offset, -hip_offset, 0])
        back_left_dir = np.array([-1, -1, 0]) / np.sqrt(2)
        back_left_knee_xyz = back_left_hip_xyz + back_left_dir * back_left_leg
        back_left_toe_xyz = back_left_knee_xyz + back_left_dir * back_left_ankle
        
        # Back Right Leg
        back_right_hip_xyz = np.array([hip_offset, -hip_offset, 0])
        back_right_dir = np.array([1, -1, 0]) / np.sqrt(2)
        back_right_knee_xyz = back_right_hip_xyz + back_right_dir * back_right_leg
        back_right_toe_xyz = back_right_knee_xyz + back_right_dir * back_right_ankle
        
        # Stack all points
        points = np.vstack([
            front_left_hip_xyz, front_left_knee_xyz, front_left_toe_xyz,
            front_right_hip_xyz, front_right_knee_xyz, front_right_toe_xyz,
            back_left_hip_xyz, back_left_knee_xyz, back_left_toe_xyz,
            back_right_hip_xyz, back_right_knee_xyz, back_right_toe_xyz,
        ])
        
        # Define connectivity matrix
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

class AntWorldSymmetric(AntWorldBase):
    """AntWorld with symmetry-enforcing mapping - left and right legs are mirrored"""
    
    def __init__(self):
        super().__init__()
        self.mapping_name = "symmetric"
    
    def geno2pheno(self, genotype):
        """
        Symmetric mapping from genotype to phenotype.
        Uses only 4 parameters for the legs (2 pairs), enforcing bilateral symmetry.
        """
        # Split genotype 
        control_weights = genotype[-self.n_weights:]
        
        # We only need 4 parameters for legs due to symmetry
        # But we'll still use all 8 and just extract/repeat values
        leg_params_raw = genotype[:-self.n_weights]
        
        # Map to parameter space
        leg_params_mapped = (leg_params_raw + 1) * 0.125 + 0.1  # Maps [-1,1] to [0.1,0.35]
        
        # Extract only the parameters we need (first 4)
        # and create symmetric body by repeating for left/right pairs
        front_legs = leg_params_mapped[0:2]  # Upper and lower for front legs
        back_legs = leg_params_mapped[4:6]   # Upper and lower for back legs
        
        # Create the full parameter set with symmetry
        body_params = np.array([
            front_legs[0], front_legs[1],  # Front left leg (upper, lower)
            front_legs[0], front_legs[1],  # Front right leg (upper, lower) - SYMMETRIC
            back_legs[0], back_legs[1],    # Back left leg (upper, lower)
            back_legs[0], back_legs[1]     # Back right leg (upper, lower) - SYMMETRIC
        ])
        
        # Validate parameters
        assert len(body_params) == 8, "Expected 8 body parameters"
        assert len(control_weights) == self.n_weights, f"Expected {self.n_weights} control weights"
        assert not np.any(body_params <= 0), "All leg segments must have positive length"
        
        # Configure neural network weights
        self.controller.geno2pheno(control_weights)

        # Unpack body parameters
        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, \
        back_left_leg, back_left_ankle, back_right_leg, back_right_ankle = body_params
        
        # Calculate leg positions
        hip_offset = 0.2
        
        # Front Left Leg
        front_left_hip_xyz = np.array([hip_offset, hip_offset, 0])
        front_left_dir = np.array([1, 1, 0]) / np.sqrt(2)
        front_left_knee_xyz = front_left_hip_xyz + front_left_dir * front_left_leg
        front_left_toe_xyz = front_left_knee_xyz + front_left_dir * front_left_ankle
        
        # Front Right Leg
        front_right_hip_xyz = np.array([-hip_offset, hip_offset, 0])
        front_right_dir = np.array([-1, 1, 0]) / np.sqrt(2)
        front_right_knee_xyz = front_right_hip_xyz + front_right_dir * front_right_leg
        front_right_toe_xyz = front_right_knee_xyz + front_right_dir * front_right_ankle
        
        # Back Left Leg
        back_left_hip_xyz = np.array([-hip_offset, -hip_offset, 0])
        back_left_dir = np.array([-1, -1, 0]) / np.sqrt(2)
        back_left_knee_xyz = back_left_hip_xyz + back_left_dir * back_left_leg
        back_left_toe_xyz = back_left_knee_xyz + back_left_dir * back_left_ankle
        
        # Back Right Leg
        back_right_hip_xyz = np.array([hip_offset, -hip_offset, 0])
        back_right_dir = np.array([1, -1, 0]) / np.sqrt(2)
        back_right_knee_xyz = back_right_hip_xyz + back_right_dir * back_right_leg
        back_right_toe_xyz = back_right_knee_xyz + back_right_dir * back_right_ankle
        
        # Stack all points
        points = np.vstack([
            front_left_hip_xyz, front_left_knee_xyz, front_left_toe_xyz,
            front_right_hip_xyz, front_right_knee_xyz, front_right_toe_xyz,
            back_left_hip_xyz, back_left_knee_xyz, back_left_toe_xyz,
            back_right_hip_xyz, back_right_knee_xyz, back_right_toe_xyz,
        ])
        
        # Define connectivity matrix
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

# Evaluation helper functions that can be pickled
def evaluate_single_individual(idx_genotype):
    idx, genotype, world = idx_genotype  # Correctly unpack all three elements
    fit_ind, _ = world.evaluate_individual(genotype)
    return idx, fit_ind

def evaluate_multi_individual(idx_genotype):
    idx, genotype, world = idx_genotype  # Correctly unpack all three elements
    _, fit_ind = world.evaluate_individual(genotype)
    return idx, fit_ind

def run_EA_single(ea_single, world, experiment_name):
    """Run single-objective optimization with the specified EA and world"""
    # Determine optimal number of workers based on CPU count
    max_workers = min(multiprocessing.cpu_count(), 3)
    print(f"Starting {experiment_name} optimization with {ea_single.n_gen} generations using {max_workers} parallel workers...")
    
    # For tracking progress
    all_best_fitness = []
    all_mean_fitness = []
    gen_times = []
    
    for gen in range(ea_single.n_gen):
        start_time = time.time()
        print(f"Generation {gen+1}/{ea_single.n_gen}...")
        pop = ea_single.ask()
        fitnesses_gen = np.empty(len(pop))
        
        # Create index-genotype-world tuples for tracking results
        indexed_pop = [(i, genotype, world) for i, genotype in enumerate(pop)]
        
        # Parallel evaluation
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(evaluate_single_individual, indexed_pop))
                
                # Process results
                for idx, fitness in results:
                    fitnesses_gen[idx] = fitness
        except Exception as e:
            print(f"Error during parallel evaluation: {e}")
            # Fall back to sequential evaluation if parallel fails
            print("Falling back to sequential evaluation...")
            for i, (idx, genotype, w) in enumerate(indexed_pop):
                try:
                    fit_ind, _ = w.evaluate_individual(genotype)
                    fitnesses_gen[idx] = fit_ind
                except Exception as indiv_e:
                    print(f"Error evaluating individual {i}: {indiv_e}")
                    fitnesses_gen[idx] = -10.0  # Default poor fitness
        
        ea_single.tell(pop, fitnesses_gen)
        
        # Record statistics
        all_best_fitness.append(ea_single.f_best_so_far)
        all_mean_fitness.append(np.mean(fitnesses_gen))
        
        elapsed = time.time() - start_time
        gen_times.append(elapsed)
        
        print(f"Generation {gen+1} completed in {elapsed:.2f}s")
        print(f"Best fitness: {ea_single.f_best_so_far:.2f}, Mean fitness: {np.mean(fitnesses_gen):.2f}")
    
    return {
        'experiment': experiment_name,
        'best_fitness': all_best_fitness,
        'mean_fitness': all_mean_fitness,
        'gen_times': gen_times,
        'best_genome': ea_single.x_best_so_far,
        'morphology_stats': world.morphology_stats,
    }

def run_EA_multi(ea_multi, world, experiment_name):
    """Run multi-objective optimization with NSGAII"""
    # Determine optimal number of workers based on CPU count
    max_workers = min(multiprocessing.cpu_count(), 3)
    print(f"Starting {experiment_name} multi-objective optimization with {ea_multi.n_gen} generations using {max_workers} parallel workers...")
    
    # For tracking progress
    all_pareto_fronts = []
    gen_times = []
    
    for gen in range(ea_multi.n_gen):
        start_time = time.time()
        print(f"Generation {gen+1}/{ea_multi.n_gen}...")
        pop = ea_multi.ask()
        fitnesses_gen = np.empty((len(pop), 2))
        
        # Create index-genotype-world tuples for tracking results
        indexed_pop = [(i, genotype, world) for i, genotype in enumerate(pop)]
        
        # Parallel evaluation
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(evaluate_multi_individual, indexed_pop))
                
                # Process results
                for idx, fitness in results:
                    fitnesses_gen[idx] = fitness
        except Exception as e:
            print(f"Error during parallel evaluation: {e}")
            # Fall back to sequential evaluation if parallel fails
            print("Falling back to sequential evaluation...")
            for i, (idx, genotype, w) in enumerate(indexed_pop):
                try:
                    _, fit_ind = w.evaluate_individual(genotype)
                    fitnesses_gen[idx] = fit_ind
                except Exception as indiv_e:
                    print(f"Error evaluating individual {i}: {indiv_e}")
                    fitnesses_gen[idx] = np.array([-10.0, -10.0])  # Default poor fitness
        
        ea_multi.tell(pop, fitnesses_gen)
        
        # Record statistics - we'll extract the Pareto front later
        # Save current population and fitness for later analysis
        all_pareto_fronts.append({
            'generation': gen,
            'population': pop.copy(),
            'fitness': fitnesses_gen.copy()
        })
        
        elapsed = time.time() - start_time
        gen_times.append(elapsed)
        
        print(f"Generation {gen+1} completed in {elapsed:.2f}s")
        print(f"Mean fitness: {np.mean(fitnesses_gen, axis=0)}")
        print(f"Min fitness: {np.min(fitnesses_gen, axis=0)}")
        print(f"Max fitness: {np.max(fitnesses_gen, axis=0)}")
    
    return {
        'experiment': experiment_name,
        'pareto_fronts': all_pareto_fronts,
        'gen_times': gen_times,
        'morphology_stats': world.morphology_stats,
    }

def generate_video(world, genotype, filename):
    """Generate a video of the robot with the specified genotype"""
    # Generate a unique filename for this video
    import time
    unique_id = f"video_{time.time()}"
    robot_xml_path = os.path.join(world.xml_dir, f"AntRobot_{unique_id}.xml")
    world.world_file = os.path.join(world.xml_dir, f"AntEnv_{unique_id}.xml")
    
    points, connectivity_mat = world.geno2pheno(genotype)
    # Set the name when creating the robot
    robot = AntRobot(points, connectivity_mat, world.joint_limits, world.joint_axis, 
                   verbose=False, name=f"AntRobot_{unique_id}")
    robot.xml = robot.define_robot()
    
    # Only pass directory to write_xml
    robot.write_xml(directory=world.xml_dir)
    
    # Check if we need to rename
    expected_path = os.path.join(world.xml_dir, f"{robot.name}.xml")
    if expected_path != robot_xml_path and os.path.exists(expected_path):
        os.rename(expected_path, robot_xml_path)
    
    # Set up the environment
    world_xml = xml.parse(os.path.join(ROOT_DIR, 'src', 'world', 'robot', 'assets', "ant_world.xml"))
    robot_env = world_xml.getroot()
    robot_env.append(xml.Element("include", attrib={"file": robot_xml_path}))
    world_xml = xml.tostring(robot_env, encoding='unicode')
    
    with open(world.world_file, "w") as f:
        f.write(world_xml)

    # Create video
    env = gym.make(ENV_NAME, robot_path=world.world_file, render_mode="rgb_array")
    observations, info = env.reset()
    frames = []
    rewards = 0
    
    for step in range(500):  # Shorter for faster video generation
        frames.append(env.render())
        action = world.controller.get_action(observations)
        observations, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        if terminated:
            break
    
    env.close()
    print(f"Total reward: {rewards:.2f}")
    
    # Save video
    try:
        import imageio
        imageio.mimsave(filename, frames, fps=30)
        print(f"Video saved to {filename}")
    except Exception as e:
        print(f"Error saving video: {e}")

def analyze_and_visualize_results(results, worlds):
    """Analyze results and create visualizations"""
    os.makedirs('experiment_results', exist_ok=True)
    
    # 1. Plot fitness over generations for each experiment
    plt.figure(figsize=(12, 6))
    for result in results:
        plt.plot(result['best_fitness'], label=f"{result['experiment']} - Best")
        plt.plot(result['mean_fitness'], linestyle='--', 
                 label=f"{result['experiment']} - Mean")
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiment_results/fitness_comparison.png', dpi=300)
    
    # 2. Generate videos of best solutions
    for i, result in enumerate(results):
        world = worlds[i]
        generate_video(world, result['best_genome'], 
                      f"experiment_results/{result['experiment']}_best.mp4")
    
    # 3. Analyze morphology statistics
    morphology_data = []
    for world in worlds:
        for stat in world.morphology_stats:
            # Extract relevant metrics for each leg
            if 'legs' not in stat or not stat['legs']:
                continue  # Skip if no leg data
                
            for i, leg in enumerate(stat['legs']):
                # Make sure we don't go out of bounds with leg positions
                leg_position = ['front_left', 'front_right', 'back_left', 'back_right'][min(i//3, 3)]
                record = {
                    'mapping': stat.get('mapping', 'unknown'),
                    'leg_position': leg_position,
                    'upper_length': leg.get('upper', 0),
                    'lower_length': leg.get('lower', 0),
                    'total_length': leg.get('total', 0),
                    'ratio': leg.get('ratio', 0),
                    'body_diagonal': stat.get('body_diagonal', 0)
                }
                morphology_data.append(record)
    
    # Check if we have enough data to create plots
    if not morphology_data:
        print("Not enough morphology data for analysis. Skipping visualization.")
        return
        
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(morphology_data)
    
    # Plot morphology statistics
    try:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot total leg lengths
        try:
            import seaborn as sns
            sns.boxplot(x='mapping', y='total_length', hue='leg_position', data=df, ax=axs[0, 0])
        except (ValueError, KeyError) as e:
            print(f"Error creating boxplot: {e}")
            axs[0, 0].text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        axs[0, 0].set_title('Total Leg Length by Mapping')
        
        # Plot upper/lower ratio
        try:
            sns.boxplot(x='mapping', y='ratio', hue='leg_position', data=df, ax=axs[0, 1])
        except (ValueError, KeyError) as e:
            print(f"Error creating ratio plot: {e}")
            axs[0, 1].text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        axs[0, 1].set_title('Upper/Lower Length Ratio by Mapping')
        
        # Plot body size
        try:
            sns.boxplot(x='mapping', y='body_diagonal', data=df, ax=axs[1, 0])
        except (ValueError, KeyError) as e:
            print(f"Error creating body size plot: {e}")
            axs[1, 0].text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        axs[1, 0].set_title('Body Size by Mapping')
        
        # Plot computation time
        times_data = []
        for result in results:
            for i, t in enumerate(result['gen_times']):
                times_data.append({
                    'mapping': result['experiment'],
                    'generation': i,
                    'time': t
                })
        
        times_df = pd.DataFrame(times_data)
        try:
            sns.lineplot(x='generation', y='time', hue='mapping', data=times_df, ax=axs[1, 1])
        except (ValueError, KeyError) as e:
            print(f"Error creating time plot: {e}")
            axs[1, 1].text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        axs[1, 1].set_title('Computation Time per Generation')
        
        plt.tight_layout()
        plt.savefig('experiment_results/morphology_analysis.png', dpi=300)
        
        # Save all results to CSV for further analysis
        df.to_csv('experiment_results/morphology_data.csv', index=False)
        times_df.to_csv('experiment_results/computation_times.csv', index=False)
    except Exception as e:
        print(f"Error during visualization: {e}")

def visualize_pareto_front(results, worlds):
    """Visualize the Pareto fronts from multi-objective optimization"""
    os.makedirs('experiment_results', exist_ok=True)
    
    # Create figure for Pareto front visualization
    plt.figure(figsize=(12, 10))
    
    # Create a colormap for generations
    cmap = plt.cm.viridis
    
    for exp_idx, result in enumerate(results):
        # Get the final Pareto front
        if 'pareto_fronts' not in result or not result['pareto_fronts']:
            print(f"No Pareto front data found for {result['experiment']}")
            continue
            
        final_front = result['pareto_fronts'][-1]
        fitness = final_front['fitness']
        
        # Plot the points
        plt.scatter(
            fitness[:, 0], fitness[:, 1], 
            label=f"{result['experiment']} final front",
            s=100, alpha=0.8
        )
        
        # Show progression across generations (if we have enough generations)
        if len(result['pareto_fronts']) > 1:
            # Plot every other generation or fewer if too many
            step = max(1, len(result['pareto_fronts']) // 5)
            for i, front in enumerate(result['pareto_fronts'][::step]):
                gen = front['generation']
                gen_fitness = front['fitness']
                
                # Use different color based on generation
                color = cmap(i / (len(result['pareto_fronts']) // step))
                plt.scatter(
                    gen_fitness[:, 0], gen_fitness[:, 1],
                    alpha=0.3, s=30, color=color
                )
    
    plt.xlabel('Forward Speed Reward')
    plt.ylabel('Energy Efficiency (Neg. Control Cost)')
    plt.title('Pareto Fronts from Multi-Objective Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('experiment_results/pareto_fronts.png', dpi=300)
    
    # Create video of selected solutions from the Pareto front
    for exp_idx, result in enumerate(results):
        if 'pareto_fronts' not in result or not result['pareto_fronts']:
            continue
            
        world = worlds[exp_idx]
        mapping_name = result['experiment']
        
        # Get the final population and fitness
        final_front = result['pareto_fronts'][-1]
        pop = final_front['population']
        fitness = final_front['fitness']
        
        if len(pop) == 0:
            print(f"No solutions in Pareto front for {mapping_name}")
            continue
        
        # Select solutions: best forward speed, best energy efficiency, and a balanced one
        best_forward_idx = np.argmax(fitness[:, 0])
        best_efficiency_idx = np.argmax(fitness[:, 1])
        
        # For balanced solution, normalize both objectives and find max of sum
        if len(fitness) > 2:  # Need at least 3 points to make this meaningful
            normalized_fitness = (fitness - np.min(fitness, axis=0)) / (np.max(fitness, axis=0) - np.min(fitness, axis=0) + 1e-10)
            balanced_idx = np.argmax(np.sum(normalized_fitness, axis=1))
        else:
            # If only 1-2 solutions, just pick the first one as "balanced"
            balanced_idx = 0
        
        # Generate videos for these solutions
        solution_names = ["forward", "efficiency", "balanced"]
        solution_indices = [best_forward_idx, best_efficiency_idx, balanced_idx]
        
        # Make sure indices are unique to avoid generating duplicate videos
        unique_indices = []
        for idx in solution_indices:
            if idx not in unique_indices:
                unique_indices.append(idx)
        
        for sol_name, sol_idx in zip(solution_names[:len(unique_indices)], unique_indices):
            try:
                generate_video(world, pop[sol_idx], 
                               f"experiment_results/{mapping_name}_pareto_{sol_name}.mp4")
                
                # Print the fitness values
                print(f"{mapping_name} - {sol_name}: {fitness[sol_idx]}")
            except Exception as e:
                print(f"Error generating video for {mapping_name} {sol_name}: {e}")

def compare_algorithms(single_results, multi_results, worlds):
    """Create comparative visualizations between CMAES and NSGAII"""
    os.makedirs('experiment_results', exist_ok=True)
    
    # 1. Plot fitness comparison between algorithms
    plt.figure(figsize=(14, 8))
    
    # Plot CMAES results
    for result in single_results:
        plt.plot(result['best_fitness'], 
                 label=f"CMAES - {result['experiment']}", 
                 linewidth=2)
    
    # Plot NSGAII results (using forward speed as main fitness)
    for result in multi_results:
        if 'pareto_fronts' not in result or not result['pareto_fronts']:
            continue
            
        # Extract best forward speed fitness from each generation
        best_forward = []
        for front in result['pareto_fronts']:
            if len(front['fitness']) > 0:
                best_forward.append(np.max(front['fitness'][:, 0]))
            else:
                best_forward.append(0)
        
        plt.plot(best_forward, 
                 label=f"NSGAII - {result['experiment']}", 
                 linewidth=2, 
                 linestyle='--')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness (Speed for NSGAII)', fontsize=12)
    plt.title('Evolutionary Performance: CMAES vs NSGAII', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('experiment_results/algorithm_comparison.png', dpi=300)
    
    # 2. Computation time comparison
    plt.figure(figsize=(12, 6))
    
    # Collect time data for both algorithms
    times_data = []
    
    for result in single_results:
        for i, t in enumerate(result['gen_times']):
            times_data.append({
                'Algorithm': 'CMAES',
                'Mapping': result['experiment'],
                'Generation': i,
                'Time': t
            })
    
    for result in multi_results:
        for i, t in enumerate(result['gen_times']):
            times_data.append({
                'Algorithm': 'NSGAII',
                'Mapping': result['experiment'],
                'Generation': i,
                'Time': t
            })
    
    times_df = pd.DataFrame(times_data)
    
    # Plot with seaborn
    sns.lineplot(x='Generation', y='Time', hue='Algorithm', style='Mapping', data=times_df)
    plt.title('Computation Time Comparison', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('experiment_results/computation_time_comparison.png', dpi=300)
    
    # Save the combined data for further analysis
    times_df.to_csv('experiment_results/algorithm_times_comparison.csv', index=False)

def main():
    # Suppress CMA-ES warnings about sampling standard deviation changes
    warnings.filterwarnings("ignore", message="Sampling standard deviation")
    
    # Set experiment parameters
    population_size = 10
    n_generations = 30
    
    # Configure CMAES parameters
    CMAES_opts["min"] = -1
    CMAES_opts["max"] = 1
    CMAES_opts["num_parents"] = 2
    CMAES_opts["num_generations"] = n_generations
    CMAES_opts["mutation_sigma"] = 5
    
    # Set up parameters for multi-objective optimization
    NSGA_opts["min"] = -1
    NSGA_opts["max"] = 1
    NSGA_opts["num_generations"] = n_generations
    NSGA_opts["mutation_prob"] = 0.3
    NSGA_opts["crossover_prob"] = 0.5
    
    sns.set_theme(style="whitegrid")

    # Create worlds with different mappings
    world_linear = AntWorldLinear()
    world_nonlinear = AntWorldNonlinear()
    world_symmetric = AntWorldSymmetric()
    
    worlds = [world_linear, world_nonlinear, world_symmetric]
    
    # Run single-objective experiments
    single_objective_results = []
    
    for world in worlds:
        # Set up output directory
        mapping_name = world.mapping_name
        results_dir = os.path.join(ROOT_DIR, 'results', f"{ENV_NAME}_{mapping_name}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize EA
        n_parameters = world.n_params
        ea = CMAES(population_size, n_parameters, CMAES_opts, results_dir)
        
        # Run optimization
        result = run_EA_single(ea, world, mapping_name)
        single_objective_results.append(result)
    
    # Analyze single-objective results
    analyze_and_visualize_results(single_objective_results, worlds)
    
    # Now run multi-objective optimization with NSGAII
    print("\n\nStarting multi-objective optimization experiments...")
    multi_objective_results = []
    
    for world in worlds:
        mapping_name = world.mapping_name
        results_dir = os.path.join(ROOT_DIR, 'results', f"{ENV_NAME}_{mapping_name}_multi")
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize NSGAII
        n_parameters = world.n_params
        ea_multi = NSGAII(population_size, n_parameters, NSGA_opts, results_dir)
        
        # Run optimization
        result = run_EA_multi(ea_multi, world, mapping_name)
        multi_objective_results.append(result)
    
    # Analyze multi-objective results
    visualize_pareto_front(multi_objective_results, worlds)
    
    # Compare both algorithms
    compare_algorithms(single_objective_results, multi_objective_results, worlds)
    
    print("\nExperiment completed! Results are in the experiment_results directory.")

if __name__ == '__main__':
    main()