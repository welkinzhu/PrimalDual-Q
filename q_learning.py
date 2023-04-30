
import gym
import tensorflow as tf
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
action_h = 26  # 实际为1-25
action_step = 4  # 每隔4个取一个
x_dual = 1
epsilon = math.sqrt(math.log(10, 3)/B)

iota = 10  # log(SAT/p)


class QLearning:

    def __init__(self, env):
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        #self.state_dim = env.observation_space.shape[0]
        self.action_dim = actions + 1
            # env.action_space.shape[0]
        self.actions = list(range(action_l, action_h, 4))
        self.action_0 = list([0, 1, 5, 9, 13, 17, 21, 25])
        #q-table，行列分别为s,a
        self.q_table = pd.DataFrame(columns=self.action_0, dtype=np.int)

        #表示(s,a)pair访问次数
        self.n_table = pd.DataFrame(columns=self.action_0, dtype=np.int)
        #表示state s的value值
        self.v_table = pd.DataFrame(columns=[1], dtype=np.int)

        self.gamma = 0.9  # decay
        self.lr = 0.01  # learning rate

        # initialize replay buffer
        # self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        # self.exploration_noise = OUNoise(self.action_dim)

    def choose_action(self, observation):
        # Checking if the state exists in the table
        self.check_state_exist(str(observation))
        # tuple_obs = list2tuple(observation)
        load_gap, need_energy = observation  ## , park_time
        empty = 0
        if len(need_energy) == 0 or load_gap <= 0:  # and len(park_time) == 0
            empty = 1
            action = 0
        # state_action = self.q_table.loc[[observation], :]  # 获取对应state一整行
        temp = []
        i = 0
        if empty == 0:
            observation = str(observation)
            for act in self.actions:
                t = self.q_table.loc[observation, act]
                # print(t)
                t = t / (x_dual * (act + 0.005))
                temp.append(t)
                # print(self.q_table.loc[observation, act])
            # print(temp)
            # print(temp)
            a = temp.index(max(temp))
            action = self.actions[a]
            # print(action)
        # Selection of the action - 90 % according to the epsilon == 0.9
        # Choosing the best action
        # state_action = state_action.reindex(np.random.permutation(state_action.index))
        # action = state_action.idxmax()
        return action

    # 每次随机选择action
    def choose_random_action(self, observation):
        self.check_state_exist(str(observation))
        load_gap, need_energy = observation
        empty = 0
        if len(need_energy) == 0 or load_gap <= 0:  # and len(park_time) == 0
            empty = 1
            action = 0
        # state_action = self.q_table.loc[[observation], :]  # 获取对应state一整行
        # temp = []
        # i = 0
        if empty == 0:
            action = random.randrange(action_l, action_h, action_step)

            # print(action)
        return action

    # 每次都选择最大的action
    def choose_greedy_action(self, observation):
        self.check_state_exist(str(observation))
        load_gap, need_energy = observation
        empty = 0
        if len(need_energy) == 0 or load_gap <= 0:  # and len(park_time) == 0
            empty = 1
            action = 0
        if empty == 0:
            action = 25
        return action

    # 传统q-learning的action
    def q_learning_action(self, observation):
        self.check_state_exist(str(observation))
        # Selection of the action - 90 % according to the epsilon == 0.9
        # Choosing the best action
        load_gap, need_energy = observation  ## , park_time
        empty = 0
        # if len(need_energy) == 0 or load_gap <= 0:  # and len(park_time) == 0
        #     empty = 1
        #     action = 0
        # state_action = self.q_table.loc[[observation], :]  # 获取对应state一整行
        temp = []
        i = 0
        if empty == 0:
            if np.random.uniform() < 0.9:  # epsilon-greedy
                state_action = self.q_table.loc[str(observation), :]  # 获取对应state一整行
                state_action = state_action.reindex(np.random.permutation(state_action.index))
                action = state_action.idxmax()
            else:
                # Choosing random action - left 10 % for choosing randomly
                action = np.random.choice(self.action_0)

        return action

    def bandit_action(self, observation):
        self.check_state_exist(str(observation))
        load_gap, need_energy = observation  ## , park_time
        empty = 0
        # if len(need_energy) == 0 or load_gap <= 0:  # and len(park_time) == 0
        #     empty = 1
        #     action = 0
        # state_action = self.q_table.loc[[observation], :]  # 获取对应state一整行
        temp = []
        i = 0
        if empty == 0:
            if np.random.uniform() < 0.9:  # epsilon-greedy
                state_action = self.q_table.loc[str(observation), :]  # 获取对应state一整行
                state_action = state_action.reindex(np.random.permutation(state_action.index))
                action = state_action.idxmax()
            else:
                # Choosing random action - left 10 % for choosing randomly
                action = np.random.choice(self.action_0)
        return action

    # Function for learning and updating Q-table with new knowledge
    def update_q(self, state, action, reward, next_state):
        # Checking if the next step exists in the Q-table
        self.check_state_exist(next_state)
        load_gap, need_energy = state
        empty = 0
        if len(need_energy) == 0 or action <= 0:  # and len(park_time) == 0
            empty = 1
        if empty == 0:
            state = str(state)
            self.n_table.loc[state, action] += 1
            # Current state in the current position
            q_predict = self.q_table.loc[state, action]
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
            # Updating Q-table with new knowledge
            self.q_table.loc[state, action] += self.lr * (q_target - q_predict)
        # return self.q_table.loc[state, action]

    def update_bandit(self, state, action, reward, next_state):
        # q(s,a) = (q(s,a) * n(s,a) + r(s,a)) / (n(s,a) + 1)
        load_gap, need_energy = state
        empty = 0
        if len(need_energy) == 0 or action <= 0:  # and len(park_time) == 0
            empty = 1
        if empty == 0:
            state = str(state)

            self.n_table.loc[state, action] += 1
            n = int(self.n_table.loc[state, action])
            q_value = int(self.q_table.loc[state, action])
            # print('previous q: %d, alpha: %f, bonus: %f' %(q_value, alpha, bonus))
            q_value = (n * q_value + reward) / (n + 1)
            # print('updated q is %d' %(q_value))
            self.q_table.loc[state, action] = q_value



    # Function for learning and updating Q-table with new knowledge
    def update(self, state, action, reward, next_state):
        # Checking if the next step exists in the Q-table
        self.check_state_exist(next_state)

        load_gap, need_energy = state
        empty = 0
        if len(need_energy) == 0 or action <= 0:  # and len(park_time) == 0
            empty = 1
        if empty == 0:
            state = str(state)
            # print(state)
            # print(self.n_table)
            # print(action)
            # print(self.n_table.loc[state, action])
            # print(type(self.n_table.loc[state, action]))
            self.n_table.loc[state, action] += 1
            n = int(self.n_table.loc[state, action])
            alpha = (H + 1) / (H + n)
            bonus = math.sqrt(H ** 3 * iota / n) / 10000
            q_value = int(self.q_table.loc[state, action])
            # print('previous q: %d, alpha: %f, bonus: %f' %(q_value, alpha, bonus))
            v_value = int(self.v_table.loc[next_state, 0])
            q_value = int((1 - alpha) * q_value + alpha * (reward + v_value + bonus))
            # print('updated q is %d' %(q_value))
            self.q_table.loc[state, action] = int(q_value)
            self.v_table.loc[state] = min(H, int(self.q_table.loc[state, action]))
        # Current state in the current position
        # return self.q_table.loc[state, action]

    # 如果没有visit到，就加上
    def check_state_exist(self, state):
        #state = list2tuple(state)
        if state not in self.q_table.index:
            newq = pd.Series([H] * self.action_dim, index=self.q_table.columns, name=state,)
            # print(newq)
            #print(self.q_table)
            self.q_table = self.q_table.append(newq)
            # self.q_table = pd.concat([self.q_table, newq], ignore_index=False)
            newn = pd.Series([0] * self.action_dim, index=self.q_table.columns, name=state,)
            self.n_table = self.n_table.append(newn)
            # self.n_table = pd.concat([self.n_table, newn], ignore_index=False)
            newv = pd.Series([0], name=state,)
            self.v_table = self.v_table.append(newv)
            # self.v_table = pd.concat([self.v_table, newv], ignore_index=False)


    def final(self):
        outqtablepath = './table/qtable.xlsx'
        self.q_table.to_excel(outqtablepath, index=True, header=True)
        outvtablepath = './table/vtable.xlsx'
        self.v_table.to_excel(outvtablepath, index=True, header=True)
        outntablepath = './table/ntable.xlsx'
        self.n_table.to_excel(outntablepath, index=True, header=True)
        # Deleting the agent at the end
        # self.canvas_widget.delete(self.agent)

        # Showing the number of steps
        # print('The shortest route:', self.shortest)
        # print('The longest route:', self.longest)

        # Creating initial point
        # origin = np.array([20, 20])
        # self.initial_point = self.canvas_widget.create_oval(
        #     origin[0] - 5, origin[1] - 5,
        #     origin[0] + 5, origin[1] + 5,
        #     fill='blue', outline='blue')
        #
        # # Filling the route
        # for j in range(len(self.f)):
        #     # Showing the coordinates of the final route
        #     print(self.f[j])
        #     self.track = self.canvas_widget.create_oval(
        #         self.f[j][0] + origin[0] - 5, self.f[j][1] + origin[0] - 5,
        #         self.f[j][0] + origin[0] + 5, self.f[j][1] + origin[0] + 5,
        #         fill='blue', outline='blue')
        #     # Writing the final route in the global variable a
        #     a[j] = self.f[j]


def draw_reward(episodes, rewards):
    # title = "reward"
    # color = cm.virdis(0.5)
    f, ax = plt.subplots(1, 1)
    ax.plot(episodes, rewards)
    ax.legend
    ax.set_xlabel("episodes")
    ax.set_ylabel("rewards")
    exp_dir = 'plot/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    else:
        os.makedirs(exp_dir, exist_ok=True)
    f.savefig(os.path.join('plot', 'reward' + str(str(datetime.datetime.now().strftime('%m%d%H%M')))
                           + '.png'), dpi=1000)
    f.show()


def draw_step(episodes, steps):
    # title = "reward"
    # color = cm.virdis(0.5)
    f, ax = plt.subplots(1, 1)
    ax.plot(episodes, steps)
    ax.legend
    ax.set_xlabel("episodes")
    ax.set_ylabel("steps")
    exp_dir = 'plot/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    else:
        os.makedirs(exp_dir, exist_ok=True)
    f.savefig(os.path.join('plot', 'step' + str(str(datetime.datetime.now().strftime('%m%d%H%M')))
                           + '.png'), dpi=1000)
    f.show()


def nozero(need_energy, park_time):
    n = []
    p = []
    for i in range(len(need_energy)):
        if need_energy[i] > 0 or park_time[i] > 0:
            n.append(need_energy[i])
            p.append(park_time[i])
    return n, p



#去掉时间，把每一个都离散化
def discretize(temp_state):
    time, load_gap, need_energy, park_time = temp_state
    dis_load_gap = int(load_gap/15) * 15
    if dis_load_gap < 0:
        dis_load_gap = 0
    # dis_renewable = int(renewable/10) * 10
    # dis_ref = int(ref_load/10) * 10
    dis_energy = need_energy
    dis_time = park_time
    #print(dis_time)
    #print(type(dis_time))
    if len(need_energy) >= 1:
        for i in range(0, len(need_energy)):
            dis_energy[i] = need_energy[i] // 7 * 7
            dis_time[i] = park_time[i] // 6 * 6
    else:
        dis_energy = need_energy
        dis_time = park_time

    dis_energy, dis_time = nozero(dis_energy, dis_time)
    dis_state = (dis_load_gap, dis_energy)
    return dis_state

# def list2tuple(state):
#     load, renewable, ref_load, need_energy, park_time = state
#     need_energy = tuple(need_energy)
#     park_time = tuple(park_time)
#     tuple_state = (load, renewable, ref_load, need_energy, park_time)
#     tuple_state = tuple(tuple_state)
#     return tuple_state
