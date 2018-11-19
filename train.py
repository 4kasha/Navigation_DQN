
"""
Navigation : Banana env in Unity ML-Agents Environments

Includes the following algorithms:
  * DNQ 
  * Double-DQN (DDQN)
  * Dueling DQN
  * Dueling DDQN

First project for Udacity's Deep Reinforcement Learning (DRL) program.
Modified the code provided by Udacity DRL Team, 2018.
"""

import numpy as np
from collections import deque
import torch
import pickle
from Agents import Agent
from unityagents import UnityEnvironment

"""
Params
======
    n_episodes (int): maximum number of training episodes
    eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    goal_score (float): average score to be required
    checkpoint (str): filename for saving weight
    scores_file (str): filename for saving scores
    env_file_name (str): your path to Banana.app
    hidden_layers (list): size of hidden_layers
    drop_p (float): probability of an element to be zeroed
    method (str): choose "DQN" or "DDQN"
    Dueling (bool): use Dueling network or not (default)
        
"""
n_episodes=600
eps_start=1.0
eps_end=0.01
eps_decay=0.995
goal_score=13.0
checkpoint="weights_DQN.pth"
scores_file="scores_DQN.txt"

env_file_name="/data/Banana_Linux_NoVis/Banana.x86_64"
hidden_layers=[512,196,64]
drop_p=0.3
method="DQN"
Dueling=False

########  Environment Setting  ########
env = UnityEnvironment(file_name=env_file_name)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
print('Number of agents:', len(env_info.agents))

# the action space 
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# the state space 
state = env_info.vector_observations[0]
state_size = len(state)
print('States have length:', state_size)
#######################################


###########  Agent Setting  ###########
agent = Agent(state_size, action_size, seed=0, hidden_layers=hidden_layers, drop_p=drop_p, method=method, Dueling=Dueling)
print('-------- Model structure --------')
print('method :', method)
print(agent.qnetwork_local)
print('---------------------------------')   
#######################################

scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores

eps = eps_start                    # initialize epsilon
isFirst = True

print('Interacting with env ...')   
for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                              
    while True:
        action = agent.act(state, eps)                 # get an action with eps-greedy
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window)>=goal_score and isFirst:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), checkpoint)
        print("Saved weights.")
        isFirst = False

f = open(scores_file, 'wb')
pickle.dump(scores, f)

# End