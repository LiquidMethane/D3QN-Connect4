import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random


class DQN(keras.Model):

    def __init__(self, n_actions, d1_dims, d2_dims):
        """D3QN class constructor

        Args:
            n_actions (int): number of possible actions
            d1_dims (int): number of neurons in the first dense layer
            d2_dims (int): number of neurons in the second dense layer

        Returns:
            None

        """

        # super constructor
        super(DQN, self).__init__()

        # define dueling double deep Q network (D3QN)
        # because observation is a matrix of size 6 x 7, there is really
        # no need for convolution layers, hence two dense layers are used 
        # here
        self.dense1 = keras.layers.Dense(d1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(d2_dims, activation='relu')

        # Value output
        self.V = keras.layers.Dense(1, activation=None)

        # Advantage output
        self.A = keras.layers.Dense(n_actions, activation=None)

        pass

    def call(self, state, training=None, mask=None):
        """override call function

        Args:
            state: the current state of the environment
            training: does noting
            mask: does nothing

        Returns:
            Q
        """

        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        # Q is calculated as defined in the paper "Dueling Network 
        # Architectures for Deep Reinforcement Learning"
        Q = (V + A - tf.math.reduce_mean(A, axis=1, keepdims=True))

        return Q

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 42), dtype=tf.float32)],
        experimental_relax_shapes=True
    )
    def advantage(self, state):
        """calculates and returns Advantage

        Args:
            state: the current state of the environment

        Returns:
            A: Advantage

        """
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A

    pass


class ReplayBuffer:

    def __init__(self, size):
        """replay buffer constructor

        Args:
            size (int): maximum size of replay buffer

        Returns:
            None
        """

        # deque overrides the oldest entry with the newest entry, which
        # makes it ideal for buffer storage
        self.replay_buffer = deque(maxlen=size)

        pass

    def update_buffer(self, transition):
        """update replay buffer with new transition

        Args:
            transition (tuple): (current_state, action, reward, new_state, done)

        Returns:
            None
        """
        self.replay_buffer.append(transition)

        pass

    def size(self):
        """returns the length of replay buffer

        Args:

        Returns:
            length of replay buffer
        """

        return len(self.replay_buffer)

    pass

    def random_sample(self, size):
        """returns a random sample of replay buffer with given size

        Args:
            size: size of random sample

        Returns:
            random sample of replay buffer

        """
        return random.sample(self.replay_buffer, size)


class Agent:

    def __init__(self, config, lr=1e-3, gamma=0.99, batch_size=64, epsilon=0,
                 eps_dec=0.99, eps_min=1e-2, buff_size=1_000_000,
                 d1_dims=128, d2_dims=128, replace_target_weight=10):
        """agent constructor

        Args:
            config: environment configuration
            lr: learning rate
            gamma: discount
            epsilon: epsilon that controls epsilon greedy policy
            eps_dec: epsilon decay
            eps_min: minimum epsilon
            batch_size: batch size for minibatch traning
            buff_size: size of replay buffer
            d1_dims: dimension of dense1 layer
            d2_dims: dimension of dense2 layer
            replace_target_weight: number of episodes before target
                                   weights are updated

        Returns:
            None
        """

        # extract number of actions from environment configuration
        self.n_actions = config.columns

        # store parameters as member variables
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size

        # define model and target
        self.model = DQN(self.n_actions, d1_dims, d2_dims)
        self.target = DQN(self.n_actions, d1_dims, d2_dims)

        # compile models
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
        # won't actually optimize
        self.target.compile(loss='mse', optimizer=Adam(learning_rate=lr))

        # copy model weights to target weights
        self.target.set_weights(self.model.get_weights())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(buff_size)

        # define replace_target_weight_threshold and counter
        self.target_replace_counter = 0
        self.replace_target_weight = replace_target_weight

        pass

    def update_replay_buffer(self, transition):
        """method to update replay buffer with new transition

        Args:
            transition: new transition of type tuple \
                (current_state, action, reward, new_state, done)

        Returns:
            None
        """

        self.replay_buffer.update_buffer(transition)

        pass

    def save_model(self, path):
        """save the target model weights to a file

        Args:
            path: filepath

        Returns:
            None
        """

        self.target.save(path)

    def load_model(self, path):
        """load the model weights from a file

        Args:
            path: filepath

        Returns:
            None

        """

        self.model = keras.models.load_model(path, custom_objects={"DQN": DQN})

    def choose_action(self, observation):
        """method to choose an action with a given observation with
        epsilon greedy policy

        Args:
            observation: the current observation from the environment

        Returns:
            action: the action chosen

        """

        # define a mask for valid actions only
        # the first n_actions elements in observation corresponds to the top row of connect 4 board
        # actions are only valid when the element equates to 0
        mask = [True if observation[idx] == 0 else False for idx in range(self.n_actions)]

        # exploration
        if np.random.random() < self.epsilon:
            # pick random action from valid action space
            valid_action_space = np.where(mask)[0]
            action = np.random.choice(valid_action_space)

        # exploitation
        else:
            # pass the current state through DQN
            state = np.array([observation])
            advantages = self.model.advantage(state).numpy().flatten()
            # mask off invalid actions
            valid_advantages = [advantages[idx] if mask[idx] else np.NaN for idx in range(self.n_actions)]
            action = np.nanargmax(valid_advantages)
        return action.item()

    def learn(self):
        """method to random sample from replay buffer and refit DQN

        Args:


        Returns:
            None
        """

        # if replay buffer has not been fully populated, do not learn
        if self.replay_buffer.size() < self.batch_size:
            return

        # sample a minibatch of transitions from replay buffer
        batch = self.replay_buffer.random_sample(self.batch_size)

        # extract current states and predict current Q's
        curr_states_list = np.array([transition[0] for transition in batch])
        curr_Qs_list = self.model.predict(curr_states_list)

        # extract next states and predict next Q's
        next_states_list = np.array([transition[3] for transition in batch])
        next_Qs_list = self.target.predict(next_states_list)

        X = []  # training X, states
        y = []  # training y, Q values

        # for each transition in batch, if it is not a terminal state,
        # then get the max Q value from the future Q values and follow 
        # formula to calculate new Q value
        # if the state is a terminal state, set the new Q value to reward
        for idx, (state, action, reward, state_, done) in enumerate(batch):
            if not done:
                max_next_Q = np.max(next_Qs_list[idx])
                new_Q = reward + self.gamma * max_next_Q

            else:
                new_Q = reward

            # update the Q value corresponding to the action
            curr_Qs = curr_Qs_list[idx]
            curr_Qs[action] = new_Q

            # append this traning sample for DQN model refit
            X.append(state)
            y.append(curr_Qs)

            pass

        # refit the model with minibatch
        self.model.fit(np.array(X), np.array(y),
                       batch_size=self.batch_size, verbose=0,
                       shuffle=False)

    def evolve(self):
        """method to copy model weights to target and decay epsilon
        """
        # update target replace counter at the end of each episode
        self.target_replace_counter += 1

        # replace target weights with model weights 
        if self.target_replace_counter > self.replace_target_weight:
            self.target.set_weights(self.model.get_weights())
            self.target_replace_counter = 0
            # print('Target model weights updated.')

        # Decay epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_dec
            self.epsilon = max(self.epsilon, self.eps_min)


    def get_agent_reward(self, reward, done):
        """agent reward function

        Args:
            reward: environment reward
            done: simulation termination status

        Returns:
            agent_rewad: agent_reward

        """

        agent_reward = 0

        if not done:
            agent_reward += -0.5

        else:
            if reward == 0:
                agent_reward += 10
            elif reward == 1:
                agent_reward += 50
            elif reward == -1:
                agent_reward += -25

        return agent_reward
        