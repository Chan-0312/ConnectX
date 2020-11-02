import sys
sys.path.append('..')
from DQN.DQN_Net import DQN_Dense, DQN_Conv
from ConnectXEnv.ConnectX_Game import ConnectX_Gym
from tqdm import tqdm
import numpy as np
import pickle

rl_flag = 'Dense'    # rl_flag {'Dense', 'Conv'}
board_size = (6, 7)      # 棋盘尺寸
model_path = './Dense_model' # 模型保存路径
display = False          # 是否显示棋盘
episodes = 1000000        # 迭代次数


env = ConnectX_Gym(rows=board_size[0], columns=board_size[1], inarow=4)
print('env_configuration: ',env.configuration)

if rl_flag == 'Dense':
    RL = DQN_Dense(
        n_actions=board_size[1],
        n_features=board_size[0]*board_size[1],
        learning_rate=0.001,
        reward_decay=0.95,
        e_greedy=0.99,
        replace_target_iter=200,
        memory_size=50000,
        batch_size=64,
        hidden_units=[128, 512, 512, 128],
        e_greedy_increment=0.01
        # output_graph = False
    )
else:
    RL = DQN_Conv(
            n_actions=board_size[1],
            n_features=board_size[0]*board_size[1],
            learning_rate=0.001,
            reward_decay=0.95,
            e_greedy=0.99,
            replace_target_iter=200,
            memory_size=50000,
            batch_size=64,
            hidden_units=[32,128,256],
            e_greedy_increment=0.01
            # output_graph = False
        )

# 加载之前的模型
# RL.load_model(model_path)

state_count = [0, 0, 0, 0]  # 胜利，失败，错误，平局
rewards_50 = [0]
all_avg_rewards = []
all_won_lost_rate = []

# 迭代序列
for i in tqdm(range(1, episodes+1)):
    observation = env.reset()

    observation = observation.reshape(-1)

    while True:

        # 绘制
        if display:
            env.render()

        action = RL.choose_action(observation)

        # 执行一次
        observation_, reward, done, _ = env.step(action)
        observation_ = observation_.reshape(-1)

        # 存放最近50个回报值
        if len(rewards_50) < 50:
            rewards_50.append(reward)
        else:
            rewards_50.pop(0)
            rewards_50.append(reward)

        # 学习
        RL.store_transition(observation, action, reward, observation_)

        if i > 50:
            RL.learn()

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

            if i % 10000 == 0:
                RL.save_model(model_path)
                pickle.dump(all_avg_rewards, open('./all_avg_rewards.pkl', 'wb'))
                pickle.dump(all_won_lost_rate, open('./all_won_lost_rate.pkl', 'wb'))

            if i % 50 == 0:
                won_lost_rate = [round(j/i, 3) for j in state_count]
                all_avg_rewards.append(np.mean(rewards_50))
                all_won_lost_rate.append(won_lost_rate)

                print('|---------|', i, won_lost_rate,
                      '%.1f' % all_avg_rewards[-1],
                      end='|---------|\n')
            break
