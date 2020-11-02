import sys
sys.path.append('..')
from ConnectXEnv.ConnectX_Game import ConnectX_Gym
from QTable.Q_Table import QLearning, Sarsa
from tqdm import tqdm
import numpy as np
import pickle


rl_flag = 'QLearning'    # rl_flag {'Sarsa', 'QLearning'}
board_size = (4, 4)      # 棋盘尺寸
qtable_path = './q_table44.pkl' # 模型保存路径
display = False          # 是否显示棋盘
episodes = 2000000        # 迭代次数


env = ConnectX_Gym(rows=board_size[0], columns=board_size[1], inarow=4)
print('env_configuration: ',env.configuration)

if rl_flag == 'Sarsa':
    RL = Sarsa(action_scope=env.configuration['columns'],
               learning_rate=0.1,
               reward_decay=0.95,
               e_greedy=0.95)
else:
    RL = QLearning(action_scope=env.configuration['columns'],
                   learning_rate=0.1,
                   reward_decay=0.95,
                   e_greedy=0.95)

# 加载之前的模型
# RL.load_qtable(qtable_path)

state_count = [0, 0, 0, 0]  # 胜利，失败，错误，平局
rewards_50 = [0]
all_avg_rewards = []
all_won_lost_rate = []
all_qtable_rows = []

# 迭代序列
for i in tqdm(range(1, episodes+1)):
    observation = env.reset()

    if rl_flag == 'Sarsa':
        action = RL.choose_action(str(observation['board']))

    while True:

        # 绘制
        if display:
            env.render()

        if rl_flag == 'QLearning':
            action = RL.choose_action(str(observation['board']))

        # 执行一次
        observation_, reward, done, _ = env.step(action)

        # 存放最近50个回报值
        if len(rewards_50) < 50:
            rewards_50.append(reward)
        else:
            rewards_50.pop(0)
            rewards_50.append(reward)

        # 学习
        if rl_flag == 'Sarsa':
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation['board']), action, reward, str(observation_['board']), action_)
        else:
            RL.learn(str(observation['board']), action, reward, str(observation_['board']))

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
                pickle.dump(all_avg_rewards, open('./all_avg_rewards.pkl', 'wb'))
                pickle.dump(all_qtable_rows, open('./all_qtable_rows.pkl', 'wb'))
                pickle.dump(all_won_lost_rate, open('./all_won_lost_rate.pkl', 'wb'))

            if i % 50 == 0:
                won_lost_rate = [round(j/i, 3) for j in state_count]
                all_avg_rewards.append(np.mean(rewards_50))
                all_won_lost_rate.append(won_lost_rate)
                all_qtable_rows.append(len(RL.q_table))

                print('|---------|', i, all_qtable_rows[-1],
                      won_lost_rate,
                      '%.1f' % all_avg_rewards[-1],
                      end='|---------|\n')
            break
