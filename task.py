import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 3
        self.time_treshold = 10

        # Goal
        # self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.target_pos = np.array([0., 0., 60.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1. - np.tanh(np.linalg.norm(self.sim.pose[:3] - self.target_pos) * 0.3)
        if np.linalg.norm(self.sim.pose[:3] - self.target_pos) < 3:
            reward = 1.
        else:
            reward = 0.
        return reward

    def get_vector(self, target_location, drone_location):
        vector = target_location - drone_location
        distance = np.linalg.norm(drone_location - target_location)
        max_distance = 100.
        scale_ = 1.0
        if distance > max_distance:
            scale_ = max_distance / distance
        scaled_vector = vector * scale_ * 0.01
        return scaled_vector

    def step(self, rotor_speeds_):
        """Uses action to obtain next state, reward, done."""
        rotor_speeds = np.array([.0, .0, .0, .0])
        rotor_speeds[0] = rotor_speeds_[0] + ((rotor_speeds_[1] + rotor_speeds_[2])) * 0.2
        rotor_speeds[1] = rotor_speeds_[0] + ((rotor_speeds_[1] - rotor_speeds_[2])) * 0.2
        rotor_speeds[2] = rotor_speeds_[0] - ((rotor_speeds_[1] - rotor_speeds_[2])) * 0.2
        rotor_speeds[3] = rotor_speeds_[0] - ((rotor_speeds_[1] + rotor_speeds_[2])) * 0.2
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            vector = self.get_vector(self.target_pos, self.sim.pose[:3])  # Vector drone to target
            scaled_pose = self.sim.pose[3:] * np.array([.1, .1, .1])  # Scaled drone rotation
            pose = np.concatenate([vector, scaled_pose])  # Concat vector + rotation
            pose_all.append(pose)
        next_state = np.concatenate(pose_all)
        self.state = next_state
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        vector = self.get_vector(self.target_pos, self.sim.pose[:3])  # Vector drone to target
        scaled_pose = self.sim.pose[3:] * np.array([.1, .1, .1])  # Scaled drone rotation
        pose = np.concatenate([vector, scaled_pose])  # Concat vector + rotation
        self.state = np.concatenate([pose] * self.action_repeat)  # Add repetitions
        return self.state
