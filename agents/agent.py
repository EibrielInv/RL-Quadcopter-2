import random
import numpy as np
import tensorflow as tf

from collections import deque
from task import Task


class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10,
                 name='QNetwork', optimize=True):
        alpha = 0.1
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            with tf.variable_scope('actions'):
                self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
                one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

            with tf.variable_scope('relu_hidden_layers'):
                # ReLU hidden layers
                self.fc1 = tf.layers.dense(self.inputs_,
                                           hidden_size,
                                           activation=None,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.fc1 = tf.maximum(alpha * self.fc1, self.fc1)

                self.fc2 = tf.layers.dense(self.fc1, hidden_size,
                                           activation=None,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.fc2 = tf.maximum(alpha * self.fc2, self.fc2)

                out_layer = self.fc2

            with tf.variable_scope('linear_output_layer'):
                # Linear output layer
                self.output = tf.layers.dense(out_layer, action_size,
                                              activation=None,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())

            tf.summary.histogram('output', self.output)
            # Train with loss (targetQ - Q)^2
            # output has length of possible actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded action.
            # Example: [1.2, 3.4, 9.3] x [1, 0, 0] = [1.2, 0, 0] = 1.2
            with tf.variable_scope('selected_Q'):
                self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            tf.summary.histogram('Q', self.Q)
            tf.summary.histogram('target_Q', self.targetQs_)

            if optimize:
                with tf.variable_scope('optimize_loss'):
                    # According to doi:10.1038/nature14236 clipping
                    # Because the absolute value loss function x
                    # has a derivative of -1 for all negative values of x
                    # and a derivative of 1 for all positive values of x
                    # , clipping the squared error to be between -1 and 1 cor-
                    # responds to using an absolute value loss function for
                    # errors outside of the (-1,1) interval.
                    # This form of error clipping further improved the stability of the algorithm.

                    # self.loss = tf.reduce_mean(self.huber_loss(self.targetQs_, self.Q))
                    self.loss = tf.reduce_mean(self.targetQs_ - self.Q)
                    tf.summary.scalar('loss', self.loss)
                    # self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                    self.opt = tf.train.RMSPropOptimizer(learning_rate, momentum=0.95).minimize(self.loss)

            self.merged = tf.summary.merge_all()

    def clipped_error(self, x):
        # Huber Loss
        # return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)  # condition, true, false
        tf.contrib
        tf.contrib.graph_editor
        tf.contrib.graph_editor.select()
        return tf.contrib.graph_editor.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)  # condition, true, false

    def huber_loss(y_true, y_pred, max_grad=1.):
        """Calculates the huber loss. - https://stackoverflow.com/a/42985363

        Parameters
        ----------
        y_true: np.array, tf.Tensor
          Target value.
        y_pred: np.array, tf.Tensor
          Predicted value.
        max_grad: float, optional
          Positive floating point value. Represents the maximum possible
          gradient magnitude.

        Returns
        -------
        tf.Tensor
          The huber loss.
        """
        err = tf.abs(y_true - y_pred, name='abs')
        mg = tf.constant(max_grad, name='max_grad')

        lin = mg * (err - .5 * mg)
        quad = .5 * err * err

        return tf.where(err < mg, quad, lin)


# create memory class for storing previous experiences
class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


class DQL_Agent():
    def __init__(self, task):
        hidden_size = 64
        learning_rate = 0.001
        state_size = 18
        self.action_size = 4
        memory_size = 1000
        pretrain_length = 500
        self.batch_size = 32
        self.gamma = 0.99                   # future reward discount
        tf.reset_default_graph()
        self.mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate, state_size=state_size, action_size=self.action_size, optimize=False)
        self.copyQN = QNetwork(name='copy', hidden_size=hidden_size, learning_rate=learning_rate, state_size=state_size, action_size=self.action_size)
        self.task = task

        # Copy Op
        def get_var(varname):
            ret = [v for v in tf.global_variables() if v.name == varname]
            if len(ret) == 0:
                print("\"{}\" not found".format(varname))
                return None
            return ret[0]
        vars2copy = []
        vars2save = {}
        for vvar in tf.global_variables():
            if vvar.name.startswith('main/'):
                # Copy the following vars
                vars2copy.append(vvar.name[5:])
                # Save the following vars
                if get_var(vvar.name) is not None:
                    vars2save[vvar.name] = get_var(vvar.name)

        copying_cm = []
        with tf.variable_scope('copy_parameters_cm'):
            for vvar in vars2copy:
                fromvar = get_var('copy/{}'.format(vvar))
                tovar = get_var('main/{}'.format(vvar))
                if fromvar is not None and tovar is not None:
                        copying_cm.append(tovar.assign(fromvar))

        copying_mc = []
        with tf.variable_scope('copy_parameters_mc'):
            for vvar in vars2copy:
                fromvar = get_var('main/{}'.format(vvar))
                tovar = get_var('copy/{}'.format(vvar))
                if fromvar is not None and tovar is not None:
                        copying_mc.append(tovar.assign(fromvar))

        # Initialize the simulation
        self.reset_episode()
        # Take one random step to generate an initial state
        # state, reward, done, _ = self.step([0.2, 0.1, 0.5, 0.1])

        self.memory = Memory(max_size=memory_size)
        action = self.discrete_to_continuous(self.random_discrete_action())
        state, reward, done = self.task.step(action)
        # Make a bunch of random actions and store the experiences
        print("Initial experiences")
        for ii in range(pretrain_length):
            action = self.discrete_to_continuous(self.random_discrete_action())
            print(action)
            next_state, reward, done = self.task.step(action)
            if done:
                # The simulation fails so no next state
                next_state = np.zeros(state.shape)
                # Add experience to memory
                self.memory.add((state, action, reward, next_state))

                # Start new episode
                self.task.reset()
                # Take one random step to generate an initial state
                random_action = self.discrete_to_continuous(self.random_discrete_action())
                state, reward, done = self.task.step(random_action)
            else:
                # Add experience to memory
                self.memory.add((state, action, reward, next_state))
                state = next_state

        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(copying_cm)  # Perform copy parameters from copy to main

            # file_writer = tf.summary.FileWriter('./logs/1', sess.graph)

    def random_discrete_action(self):
        return random.randint(0, 12)  # np.array([0.2, 0.1, 0.5, 0.1])

    def discrete_to_continuous(self, discrete):
        values = [-300, 0.5, 300]
        combinations = []
        for v_1 in values:
            for v_2 in values:
                for v_3 in values:
                    for v_4 in values:
                        combinations.append([v_1, v_2, v_3, v_4])
        return np.array(combinations[discrete])

    def act(self, state):
        # Choose action based on given state and policy
        # action = np.dot(state, self.w)  # simple linear policy
        # action = np.array([.1, .1, .1, .1])
        if 0.5 > np.random.rand():
            action = self.random_discrete_action()  # TODO take a random action
        else:
            # Get action from Q copy
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                feed = {
                    self.copyQN.inputs_: state.reshape((1, *state.shape)),
                    self.copyQN.batch_size: 1,
                    self.copyQN.keep_prob: 1.0,
                    # Dummy values, not used, only to satisfy the Graph
                    self.mainQN.inputs_: state.reshape((1, *state.shape)),
                    self.mainQN.batch_size: 1,
                    self.mainQN.keep_prob: 1.0
                }
                Qs = sess.run(self.copyQN.output, feed_dict=feed)
                action = np.argmax(Qs)
        return self.discrete_to_continuous(action)

    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        if done:
            # the episode ends so no next state
            next_state = np.zeros(state.shape)

            t = max_steps
            rewards_list.append((ep, total_reward))

            # Add experience to memory
            self.memory.add((state, action, reward, next_state))

            # Start new episode
            env.reset()
            # Take one random step, get new state and reward
            state, reward, done, _ = env.step(env.action_space.sample())

        else:
            # Add experience to memory
            self.memory.add((state, action, reward, next_state))
            state = next_state
            # t += 1
        # Learn, if at end of episode
        # if done:
        #     self.learn()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # Sample mini-batch from memory
            batch = self.memory.sample(self.batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            # Get main Q^ #
            # Executes batch_size actions, and caches the output
            # of the Neural Network. output = Q
            feed_dict = {
                self.mainQN.inputs_: next_states,
                self.mainQN.batch_size: self.batch_size,
                self.mainQN.keep_prob: 1.0
            }
            target_Qs = sess.run(self.mainQN.output, feed_dict=feed_dict)

            # data = sess.run(self.mainQN.inputs2d, feed_dict=feed_dict)

            # Set target_Qs to 0 for states where episode ends
            # Ending episode + 1 should have zero Q (all the reward
            # is "stored" on the current reward)
            episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
            target_Qs[episode_ends] = [0 for _ in range(0, self.action_size)]

            # Updates the already generated Q according to the reward
            # like if the generated Q where real (?)
            targets = rewards + self.gamma * np.max(target_Qs, axis=1)

            # Train copy Network #
            # Force the network to output the new Q
            # given the same state and action as before
            feed_dict = {
                self.copyQN.inputs_: states,
                self.copyQN.targetQs_: targets,
                self.copyQN.actions_: actions,
                self.copyQN.batch_size: self.batch_size,
                self.copyQN.keep_prob: 7.0,
                # Dummy values, not used, only to satisfy the Graph
                self.mainQN.inputs_: states,
                self.mainQN.actions_: actions,
                self.mainQN.targetQs_: targets,
                self.mainQN.batch_size: self.batch_size,
                self.mainQN.keep_prob: 7.0,
            }
            summary, loss, _ = sess.run([self.copyQN.merged, self.copyQN.loss, self.copyQN.opt],
                                        feed_dict=feed_dict)

    def learn(self):
        self.score = .0
        self.best_score = .1
        self.noise_scale = 0.2

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state
