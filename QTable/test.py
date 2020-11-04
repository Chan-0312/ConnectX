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
episodes = 100           # 迭代次数


env = ConnectX_Gym(rows=board_size[0], columns=board_size[1], inarow=4, switch=2, agent_level=1)
print('env_configuration: ',env.configuration)

if rl_flag == 'Sarsa':
    RL = Sarsa(action_scope=env.configuration['columns'],
               learning_rate=0.1,
               reward_decay=1,
               e_greedy=1)
else:
    RL = QLearning(action_scope=env.configuration['columns'],
                   learning_rate=0.1,
                   reward_decay=1,
                   e_greedy=1)

# 加载之前的模型
RL.load_qtable(qtable_path)

state_count = [0, 0, 0, 0]  # 胜利，失败，错误，平局


# 迭代序列
for i in tqdm(range(1, episodes+1)):
    observation = env.reset()

    while True:

        # 绘制
        if display:
            env.render()


        action = RL.choose_action(str(observation.flatten().tolist()))

        # 执行一次
        observation_, reward, done, _ = env.step(action)

        # 下一次迭代
        observation = observation_

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

            break

    print('|---------|', i, state_count, end='|---------|\n')
