import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os
import datetime
from datetime import date

#import filter_env
from q_learning import *
import gc
import gym
import gym_vehicle
from vehicle_env import VehicleEnv
from copy import deepcopy

import random

gc.enable()

# ENV_NAME = 'vehicle-v0'
EPISODES = 100000
# TEST = 10
TIMESTEPLIMIT = 2000


if __name__ == '__main__':
    # env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    # env = gym.make(ENV_NAME)
    env = VehicleEnv()
    agent = QLearning(env)
    # env.monitor.start('experiments/' + ENV_NAME, force=True)

    episodes = []
    rewards = []
    rewards_low = []
    rewards_high = []
    a5rewards = []
    remains = []
    e_step = []
    sum_reward = []
    # e_num = []
    output = open('./log/log.txt', 'w')
    outReward = open('./reward/reward.txt', 'w')
    # outQtable = open('./table/qtable.xls', 'w')
    output.write('episode\tstep\ttime\tload_gap\tneed_energy\tpark_time\tdis_action\t'
              'action\treward\tassign\tremain_action\tdecision\n')
    outReward.write('episode\tave_reward\te_step\n')
    for episode in range(EPISODES):
        state = env.reset()
        # print("episode:",episode)
        # Train
        total_reward = 0
        remains = []
        states = []
        actions = []
        num = []
        budget = 1000

        TOTAL_STEP = TIMESTEPLIMIT  # 一个episode内 总STEP数量
        # TOTAL_NUM = 0  # 一个episode内 抽烟总量
        temp_reward = []
        i = 1

        for step in range(TIMESTEPLIMIT):
            # if episode == int(EPISODES/5) * i:
            output.write(str(episode) + '\t')
            output.write(str(step) + '\t')
            output.write(str(state[0]) + '\t')  # time
            output.write(str(state[1]) + '\t')  # load_gap
            # output.write(str(state[2]) + '\t')  # renewable
            # output.write(str(state[2]) + '\t')  # ref_load
            output.write(str(state[2]) + '\t')  # need_energy list
            output.write(str(state[3]) + '\t')  # park_time list
            # print('former episodes: %d %d' % (episode, i))
            temp_state = deepcopy(state)
            discretize_state = discretize(temp_state)
            # output.write(str(discretize_state) + '\t')

            action = agent.choose_action(discretize_state)
            # action = agent.bandit_action(discretize_state)
            # action = agent.q_learning_action(discretize_state)
            # action = agent.choose_random_action(discretize_state)
            # action = agent.choose_greedy_action(discretize_state)

            next_state, reward, done, remain_action, assign, decision = env.step(state, action)
            # print('run_after:')
            # print(next_state[2],  next_state[3])
            budget = budget - action + remain_action

            temp_next = deepcopy(next_state)
            state = temp_next

            dis_next_state = discretize(next_state)

            # agent.update_bandit(discretize_state, action, reward, str(dis_next_state))
            agent.update(discretize_state, action, reward, str(dis_next_state))  # 更新q值
            # agent.update_q(discretize_state, action, reward, str(dis_next_state))
            #action = np.clip(action, env.action_space.low, env.action_space.high)
                # random.randint(0,30)
            # agent.perceive(state, action, reward, next_state, done)

            # (time, load, renewable, ref_load, need_energy, park_time)
            # actions.append(action)
            # states.append(state[3])
            # remains.append(state[4])
            # num.append(state[5])


            # output.write(str(state[7]) + '\t')
            output.write(str(action) + '\t')
            output.write(str(reward) + '\t')
            output.write(str(assign) + '\t')
            output.write(str(remain_action) + '\t')

            output.write('\n')


            # output.write(str(decision) + '\t')
            # output.write(str(state[5]/state[6]) + '\t')

            if budget <= 0:
                done = True
            if episode == int(EPISODES / 5) * i and done:
                i = i + 1
            if done:
                TOTAL_STEP = step
                #TOTAL_NUM = state[5]

                break

            # print('step finish:')
            # print(state)
            total_reward += reward
            temp_reward.append(reward)

        #ave_reward = TOTAL_NUM / TOTAL_STEP
        sum_reward.append(total_reward)
        e_step.append(TOTAL_STEP)

        ave_reward = total_reward / TOTAL_STEP

        # a5rewards.append(ave_reward)
        rewards.append(ave_reward)
        outReward.write(str(episode) + '\t')
        outReward.write(str(ave_reward) + '\t')

        # outReward.write(str(ave_reward) + '\t')

        # rewards_low.append(min(temp_reward))
        # rewards_high.append(max(temp_reward))

        # rewards.append(a5rewards[-1])
        print('episodes: %d' %(episode))
        # outReward.write(str(episode) + '\t')
        # outReward.write(str(total_reward) + '\t')

        # e_num.append(TOTAL_NUM)
        outReward.write(str(TOTAL_STEP) + '\t')
        outReward.write(str(total_reward) + '\t')
        # outReward.write(str(min(temp_reward)) + '\t')
        # outReward.write(str(max(temp_reward)) + '\t')
        # outReward.write(str(TOTAL_NUM) + '\t')
        outReward.write('\n')
        # ave_remain_rewards = remain_reward / TIMESTEPLIMIT
        episodes.append(episode)
        # rewards.append(total_reward)
        # plt.show()
        # print('episode: ', episode, 'Eva_Ave_Reward:', ave_reward, 'remain:', remains[0],
        #      'actions:', actions[0], 'imp:', states[0], 'total num:', num[len(num)-1])
             # 'Remained monetary reward:', ave_remain_rewards)

    # draw_reward(episodes, a5rewards)
    draw_reward(episodes, sum_reward)
    draw_reward(episodes, rewards)
    draw_step(episodes, e_step)
    #draw_reward(episodes, e_num)
    output.close()
    outReward.close()
    agent.final()

    '''
        # Testing:
        if episode % 10 == 0 and episode >= 10:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(TIMESTEPLIMIT):
                    # env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
    '''

    # env.monitor.close()
