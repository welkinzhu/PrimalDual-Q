import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os
import datetime
from datetime import date

#import filter_env
from primaldualq import *
import gc
import gym
# import gym_vehicle
from copy import deepcopy

import random

gc.enable()

ENV_NAME = 'vehicle_to_grid-v0'
EPISODES = 100000
# TEST = 10
TIMESTEPLIMIT = 500


if __name__ == '__main__':
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    env = gym.make(ENV_NAME)
    agent = primaldualq(env)
    # env.monitor.start('experiments/' + ENV_NAME, force=True)

    episodes = []
    rewards = []
    rewards_low = []
    rewards_high = []
    remains = []
    e_step = []
    sum_reward = []
    output = open('./log/log.txt', 'w')
    outReward = open('./reward/reward.txt', 'w')
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        remains = []
        states = []
        actions = []
        num = []
        budget = 1000
        TOTAL_STEP = TIMESTEPLIMIT  
        temp_reward = []
        for step in range(TIMESTEPLIMIT): 
            output.write(str(episode) + '\t')
            output.write(str(step) + '\t')
            output.write(str(state[0]) + '\t') 
            output.write(str(state[1]) + '\t') 
            output.write(str(state[2]) + '\t')
            output.write(str(state[3]) + '\t')  
            temp_state = deepcopy(state)
            discretize_state = discretize(temp_state)
            action = agent.choose_action(discretize_state)
            next_state, reward, done, remain_action, assign, decision = env.step(state, action)
            budget = budget - action + remain_action
            temp_next = deepcopy(next_state)
            state = temp_next
            dis_next_state = discretize(next_state)
            agent.update(discretize_state, action, reward, str(dis_next_state)) 
            output.write(str(action) + '\t')
            output.write(str(reward) + '\t')
            output.write(str(assign) + '\t')
            output.write(str(remain_action) + '\t')
            output.write('\n')

            if budget <= 0:
                done = True          
            if done:
                TOTAL_STEP = step
                break
            total_reward += reward
            temp_reward.append(reward)

        sum_reward.append(total_reward)
        e_step.append(TOTAL_STEP)

        ave_reward = total_reward / TOTAL_STEP
        rewards.append(ave_reward)
        outReward.write(str(episode) + '\t')
        outReward.write(str(ave_reward) + '\t')
        outReward.write(str(TOTAL_STEP) + '\t')
        outReward.write(str(total_reward) + '\t')
        outReward.write('\n')
        episodes.append(episode)

    # draw_reward(episodes, sum_reward)
    # draw_reward(episodes, rewards)
    # draw_step(episodes, e_step)
    # draw_reward(episodes, e_num)
    output.close()
    outReward.close()
    agent.final()

    # env.monitor.close()
