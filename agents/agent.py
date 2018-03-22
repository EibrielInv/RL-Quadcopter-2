import json
import time
import tensorflow as tf

import agents.training as training
from agents.models import Actor
from agents.models import Critic
from agents.memory import Memory
from agents.noise import *

from agents.common import logger

from task import Task

from mpi4py import MPI


class action_space:
    def __init__(self):
        self.shape = (3,)
        self.range = 600
        # self.low = np.array([-self.range, -self.range, -self.range, -self.range])
        # self.high = np.array([self.range, self.range, self.range, self.range])
        self.low = np.array([-self.range, -1., -1.])
        self.high = np.array([self.range, 1., 1.])


class observation_space:
    def __init__(self):
        self.shape = (19,)


class dummy_environment:
    def __init__(self, eval_=False):
        self.action_space = action_space()
        self.observation_space = observation_space()
        self.task = Task(runtime=20.)
        self.global_time = 0
        self.labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
                       'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
                       'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'reward']
        self.results = {x: [] for x in self.labels}
        self.to_write = None
        self.eval = eval_

    def seed(self, seed):
        self.seed = seed

    def step(self, action):
        # action += self.action_space.range
        action += np.array([self.action_space.range, 0., 0.])
        new_obs, r, done = self.task.step(action)
        self.to_write = [self.global_time] + list(self.task.sim.pose) + list(self.task.sim.v) + list(self.task.sim.angular_v) + list(action) + [float(r)]
        for ii in range(len(self.labels)):
            self.results[self.labels[ii]].append(self.to_write[ii])
        self.global_time += 1
        if self.eval:
            file_output = 'data.json'
            if self.global_time % 1000 == 0:
                with open(file_output, 'w') as data_file:
                    json.dump(self.results,
                              data_file,
                              sort_keys=True,
                              indent=4,
                              separators=(',', ': '))
        return new_obs, r, done, {}

    def reset(self):
        state = self.task.reset().reshape((19,))
        return state

    def close(self):
        return


class DDPG_Agent:
    def __init__(self):
        rank = MPI.COMM_WORLD.Get_rank()
        env = dummy_environment()
        eval_env = dummy_environment(True)
        # logger = Logger()
        layer_norm = True
        seed = 0
        stddev = 0.2
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=stddev, desired_action_stddev=stddev)
        action_noise = None
        kwargs = {
            'nb_epochs': 50,  # 500
            'nb_epoch_cycles': 20,
            'render_eval': False,
            'reward_scale': 1.,
            'render': False,
            'normalize_returns': False,
            'normalize_observations': False,
            'critic_l2_reg': 1e-2,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'popart': False,
            'gamma': 0.99,
            'clip_norm': None,
            'nb_train_steps': 50,
            'nb_rollout_steps': 100,
            'nb_eval_steps': 100,
            'batch_size': 64
        }

        nb_actions = env.action_space.shape[-1]

        # Configure components.
        memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
        critic = Critic(layer_norm=layer_norm)
        actor = Actor(nb_actions, layer_norm=layer_norm)

        # Seed everything to make things reproducible.
        seed = seed + 1000000 * rank
        logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
        tf.reset_default_graph()
        # set_global_seeds(seed)
        env.seed(seed)
        if eval_env is not None:
            eval_env.seed(seed)

        # Disable logging for rank != 0 to avoid noise.
        if rank == 0:
            start_time = time.time()
        training.train(env=env, eval_env=eval_env, param_noise=param_noise,
                       action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
        env.close()
        if eval_env is not None:
            eval_env.close()
        if rank == 0:
            logger.info('total runtime: {}s'.format(time.time() - start_time))
        self.env = env

    def train(self):
        print("Training")
        return self.env.results
