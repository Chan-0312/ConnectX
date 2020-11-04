import sys
sys.path.append('..')
from ConnectXEnv.ConnectX_Game import ConnectX_Gym
from QTable.Q_Table import QLearning, Sarsa
from tqdm import tqdm
import numpy as np
import pickle


rl_flag = 'QLearning'    # rl_flag {'Sarsa', 'QLearning'}
board_size = (6, 7)      # 棋盘尺寸
qtable_path = './q_table67.pkl' # 模型保存路径
display = False          # 是否显示棋盘
episodes = 5000000        # 迭代次数


env = ConnectX_Gym(rows=board_size[0], columns=board_size[1], inarow=4)
print('env_configuration: ',env.configuration)

if rl_flag == 'Sarsa':
    RL = Sarsa(action_scope=env.configuration['columns'],
               learning_rate=0.1,
               reward_decay=0.95,
               e_greedy_max=0.99,
               e_greedy_increment=5/episodes)
else:
    RL = QLearning(action_scope=env.configuration['columns'],
                   learning_rate=0.1,
                   reward_decay=0.95,
                   e_greedy_max=0.99,
                   e_greedy_increment=5/episodes)

# 加载之前的模型
# RL.load_qtable(qtable_path)

state_count = [0, 0, 0, 0]  # 胜利，失败，错误，平局
rewards_100mean = 0
all_avg_rewards = []
all_won_lost_rate = []
all_qtable_rows = []

# 迭代序列
for i in tqdm(range(1, episodes+1)):
    observation = env.reset()

    if rl_flag == 'Sarsa':
        action = RL.choose_action(str(observation.flatten().tolist()))

    while True:

        # 绘制
        if display:
            env.render()

        if rl_flag == 'QLearning':
            action = RL.choose_action(str(observation.flatten().tolist()))

        # 执行一次
        observation_, reward, done, _ = env.step(action)

        # 存放最近100回报指之和
        rewards_100mean += reward

        # 学习
        if rl_flag == 'Sarsa':
            action_ = RL.choose_action(str(observation_.flatten().tolist()))
            RL.learn(str(observation.flatten().tolist()), action, reward, str(observation_.flatten().tolist()), action_)
        else:
            RL.learn(str(observation.flatten().tolist()), action, reward, str(observation_.flatten().tolist()))

        # 下一次迭代
        observation = observation_
        if rl_flag == 'Sarsa':
            action = action_

        # 游戏结束
        if done:
            if reward == 1:     # 胜利
                state_count[0] += 1
            elif reward == -1:  # 失败
                state_count[1] += 1
            elif reward == -10: # 无效放置
                state_count[2] += 1
            else:               # 平局
                state_count[3] += 1

            if i % 10000 == 0:
                RL.save_qtable(path=qtable_path)
                pickle.dump(all_avg_rewards, open('./all_avg_rewards_q67.pkl', 'wb'))
                pickle.dump(all_qtable_rows, open('./all_qtable_rows_q67.pkl', 'wb'))
                pickle.dump(all_won_lost_rate, open('./all_won_lost_rate_q67.pkl', 'wb'))

            if i % 100 == 0:
                # won_lost_rate = [round(j/i, 3) for j in state_count]
                won_lost_rate = [round(j/100, 2) for j in state_count]
                state_count = [0, 0, 0, 0]
                all_avg_rewards.append(rewards_100mean/100)
                rewards_100mean = 0
                all_won_lost_rate.append(won_lost_rate)
                all_qtable_rows.append(len(RL.q_table))

                print('|---------|', i, all_qtable_rows[-1], '%.3f'%RL.epsilon,
                      won_lost_rate,
                      '%.1f' % all_avg_rewards[-1],
                      end='|---------|\n')
            break
