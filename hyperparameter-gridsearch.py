# Hyperparameter selection
# - learning rate (0.001, 0.0005)
# - batch size (64, 128)
# - gamma (0.95, 0.99)
# - replace target weight (10, 20)

from tqdm import tqdm
from D3QNAgent import Agent
from kaggle_environments import evaluate, make, utils
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Process

nsl_agent = './fast_Nstep_lookahead_agent.py'
random_agent = 'random'
negamax_agent = 'negamax'



lr = [0.0005, 0.001]
batch_size = [64, 128]
gamma = [0.95, 0.99]
replace_weight = [10, 20]



def train_agent(idx, lr, batch_size, gamma, replace_weight):

    print('{} Started!'.format(idx))

    def preprocess_board_state(observation):
        board = np.array(observation['board'], dtype=np.float32).reshape(6, 7)
        marker = observation.mark

        state = np.zeros((6, 7, 2), dtype=np.float32)
        
        if marker == 1:
            state[:, :, 0][board == 1] = 1
            state[:, :, 1][board == 2] = 1
            
        else:
            state[:, :, 0][board == 2] = 1
            state[:, :, 1][board == 1] = 1

        return state

    env = make("connectx")
    env.render()
    
    # define agent with current hyperparameter values
    agent = Agent(
        env.configuration, 
        lr=lr, 
        gamma=gamma, 
        batch_size=batch_size,
        epsilon=1,        
        eps_dec=0.999,
        eps_min=0.02,
        buff_size=50_000,
        conv1_filts=16,
        d1_dims=128,
        d2_dims=128,
        replace_target_weight=replace_weight,
        input_shape=(None, 6, 7, 2),
    )
    
    # define rewards
    rewards = []

    # define trainer
    trainer = env.train([nsl_agent, None])

    # define training episodes
    num_episodes = 5000

    # training procedure
    for i in tqdm(range(num_episodes)):
        done = False
        tot_reward = 0

        observation = trainer.reset()
        obs = preprocess_board_state(observation)
        
        # print(obs.reshape(6, 7))

        while not done:

            # choose the best action
            action = agent.choose_action(obs)
            
            # step the environment with action
            # store all returns
            observation_, reward, done, info = trainer.step(action)

            # reprocess the new raw state from the environment
            obs_ = preprocess_board_state(observation_)

            # calculate agent reward from environment response
            agent_reward = agent.get_agent_reward(reward, done)
            
            tot_reward += agent_reward
            
            # store this transition
            agent.update_replay_buffer((obs, action, agent_reward, obs_, done))
            
            # update the current obs with new obs
            obs = obs_
            
            # perform gradient descent step
            agent.learn()
        
        # decay epsilon and update target model weights
        agent.evolve()

        # store total rewards
        rewards.append(tot_reward)
        
        
        # if i % 20 == 0:
        #     clear_output(wait=True)
        #     print('Searching #{}\tlr: {}\tbatch size: {}\tgamma: {}\treplace weight: {}\tCurrent Episode: {}'.format(idx, lr, batch_size, gamma, replace_weight, i))

    # calculate moving average rewards
    avg_reward_100 = np.convolve(rewards, np.ones(100), mode='valid') / 100

    # plot average rewards
    fig, ax = plt.subplots(dpi=120, figsize=(12.8, 7.2))
    ax.plot(avg_reward_100, color='orange', linestyle='-')
    ax.set_ylabel('Average reward')
    ax.set_xlabel('# Episode')
    ax.title.set_text('lr={}, batch size={}, gamma={}, replace_weight={}'.format(lr, batch_size, gamma, replace_weight))
    fig.savefig('./figures/cnn-nsl-{}-{}-{}-{}-{}.png'.format(idx, lr, batch_size, gamma, replace_weight))
    
    # save agent model weights 
    agent.save_DQN_weights('./models/cnn-nsl-{}-{}-{}-{}-{}.h5'.format(idx, lr, batch_size, gamma, replace_weight))


    print('{} Done!'.format(idx))

if __name__ == '__main__':

    process_list = []

    # loop through all combination of the hyperparamters to perform grid search
    for idx, (lr, batch_size, gamma, replace_weight) in enumerate(list(itertools.product(lr, batch_size, gamma, replace_weight))):
        p = Process(target=train_agent, args=(idx, lr, batch_size, gamma, replace_weight,))
        p.start()
        p.join()
        process_list.append(p)

