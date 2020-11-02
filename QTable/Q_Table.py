import numpy as np
import pickle

# QTable类
class QTable(object):

    # 初始化
    def __init__(self,
                 action_scope=7,        # 行为范围
                 learning_rate=0.1,     # 学习率
                 reward_decay=0.9,      # 回报衰退因子
                 e_greedy=0.9,          # 贪婪因子
                 ):

        self.action_scope = action_scope
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = dict()

    # 选择行为, observation要为字符串形式
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = np.array(self.q_table[str(observation)])
            # print(state_action)
            action = np.random.choice(np.arange(self.action_scope)[state_action == state_action.max()])
        else:

            # temp = [i for i in range(self.action_scope) if observation[i] == 0]
            # if temp == []:
            #     action = -1
            # else:
            #     action = np.random.choice(temp)

            # 不能让其知道规则
            action = np.random.choice(range(self.action_scope))

        return int(action)

    # 学习训练
    def learn(self):
        pass

    # 检查表是否存在
    def check_state_exist(self, observation):
        if observation not in self.q_table:
            self.q_table[observation] = list(np.zeros(self.action_scope))

    # 保存Q表
    def save_qtable(self, path):
        if self.q_table != None:
            pickle.dump(self.q_table, open(path, 'wb'))
        print('save qtable')

    # 加载Q表
    def load_qtable(self, path):
        self.q_table = pickle.load(open(path, 'rb'))
        print('load qtable')

# Q学习
class QLearning(QTable):

    # 初始化
    def __init__(self,
                 action_scope=7,        # 行为范围
                 learning_rate=0.1,     # 学习率
                 reward_decay=0.9,      # 回报衰退因子
                 e_greedy=0.9,          # 贪婪因子
                 ):
        super(QLearning, self).__init__(action_scope, learning_rate, reward_decay, e_greedy)
        print('QLearning - lr:%f,reward_decay:%f,e_greedy:%f'%(learning_rate,reward_decay,e_greedy))

    # 学习训练
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        q_target = r + self.gamma * max(self.q_table[s_])

        self.q_table[s][a] += self.lr * (q_target - q_predict)  # update


# Sarsa类
class Sarsa(QTable):

    def __init__(self,
                 action_scope=7,        # 行为范围
                 learning_rate=0.1,     # 学习率
                 reward_decay=0.9,      # 回报衰退因子
                 e_greedy=0.9,          # 贪婪因子
                 ):
        super(Sarsa, self).__init__(action_scope, learning_rate, reward_decay, e_greedy)
        print('Sarsa - lr:%f,reward_decay:%f,e_greedy:%f'%(learning_rate,reward_decay,e_greedy))

    # 学习训练
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        q_target = r + self.gamma * self.q_table[s_][a_]
        self.q_table[s][a] += self.lr * (q_target - q_predict)
