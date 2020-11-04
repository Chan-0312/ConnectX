import sys
sys.path.append('..')
from stable_baselines.bench import Monitor
from ConnectXEnv.ConnectX_Game import ConnectX_Gym

# 监控环境，读取每次训练的reward值保存到monitor.csv
env = Monitor(ConnectX_Gym(), './')

from stable_baselines import DQN
from stable_baselines.deepq.policies import LnMlpPolicy

# 训练模型
model = DQN(LnMlpPolicy, env, verbose=1).learn(total_timesteps=200*10)

# 保存环境
model.save('./DQN_model')
print('save model')
