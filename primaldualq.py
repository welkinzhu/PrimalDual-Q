
import gym
# import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import datetime
import random

# from replay_buffer import ReplayBuffer

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 1
H = 500
B = 500
actions = 7
action_l = 1
action_h = 26 
action_step = 4  
x_dual = 1
epsilon = math.sqrt(math.log(10, 3)/B)

iota = 10  # log(SAT/p)


class primaldualq:

    def __init__(self, env):
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        #self.state_dim = env.observation_space.shape[0]
        self.action_dim = actions + 1
            # env.action_space.shape[0]
        self.actions = list(range(action_l, action_h, 4))
        self.q_table = pd.DataFrame(columns=self.action_0, dtype=np.int)
        self.n_table = pd.DataFrame(columns=self.action_0, dtype=np.int)
        self.v_table = pd.DataFrame(columns=[1], dtype=np.int)

        self.gamma = 0.9  # decay
        self.lr = 0.01  # learning rate

    def choose_action(self, observation):
        # Checking if the state exists in the table
        self.check_state_exist(str(observation))
        load_gap, need_energy = observation
        empty = 0
        if len(need_energy) == 0 or load_gap <= 0:  # and len(park_time) == 0
            empty = 1
            action = 0
        # state_action = self.q_table.loc[[observation], :]
        temp = []
        i = 0
        if empty == 0:
            observation = str(observation)
            for act in self.actions:
                t = self.q_table.loc[observation, act]
                t = t / (x_dual * (act + 0.005))
                temp.append(t)
            a = temp.index(max(temp))
            action = self.actions[a]
        return action
 
    def update(self, state, action, reward, next_state):
        # Checking if the next step exists in the Q-table
        self.check_state_exist(next_state)
        load_gap, need_energy = state
        empty = 0
        if len(need_energy) == 0 or action <= 0:  # and len(park_time) == 0
            empty = 1
        if empty == 0:
            state = str(state)
            self.n_table.loc[state, action] += 1
            n = int(self.n_table.loc[state, action])
            alpha = (H + 1) / (H + n)
            bonus = math.sqrt(H ** 3 * iota / n) / 10000
            q_value = int(self.q_table.loc[state, action])
            v_value = int(self.v_table.loc[next_state, 0])
            q_value = int((1 - alpha) * q_value + alpha * (reward + v_value + bonus))
            self.q_table.loc[state, action] = int(q_value)
            self.v_table.loc[state] = min(H, int(self.q_table.loc[state, action]))


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            newq = pd.Series([H] * self.action_dim, index=self.q_table.columns, name=state,)
            self.q_table = self.q_table.append(newq)        
            newn = pd.Series([0] * self.action_dim, index=self.q_table.columns, name=state,)
            self.n_table = self.n_table.append(newn)
            newv = pd.Series([0], name=state,)
            self.v_table = self.v_table.append(newv)
     

    def final(self):
        outqtablepath = './table/qtable.xlsx'
        self.q_table.to_excel(outqtablepath, index=True, header=True)
        outvtablepath = './table/vtable.xlsx'
        self.v_table.to_excel(outvtablepath, index=True, header=True)
        outntablepath = './table/ntable.xlsx'
        self.n_table.to_excel(outntablepath, index=True, header=True)
