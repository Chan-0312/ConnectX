import tensorflow as tf
import numpy as np

# DQN基类
class DQN_Net(object):

    # 初始化
    def __init__(
            self,
            n_actions = 7,
            n_features = 42,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=32,
            batch_size=32,
            hidden_units=[128, 256, 256, 128],
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions    # 可选择的行为数量
        self.n_features = n_features    # 环境变量尺寸
        self.lr = learning_rate       # 学习速率
        self.gamma = reward_decay     # 回报衰减值
        self.epsilon_max = e_greedy   # 最大贪婪值
        self.replace_target_iter = replace_target_iter  # 更换target_net的步数
        self.memory_size = memory_size  # 记忆库上限
        self.batch_size = batch_size    # 批量训练的数量
        self.epsilon_increment = e_greedy_increment  # epsilon的增量
        self.hidden_units = hidden_units  # 隐藏层数


        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 是否开启探索模式, 并逐步减少探索次数

        # 记录学习次数(用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0

    # 存储训练数据
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size

        self.memory[index, :] = transition

        self.memory_counter += 1

    # 选择行为
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            # temp = [i for i in range(self.n_actions) if observation[0,i] == 0]
            # if temp == []:
            #     action = -1
            # else:
            #     action = np.random.choice(temp)

            # 不能让其知道规则
            action = np.random.choice(range(self.n_actions))

        return int(action)

    # 学习参数
    def learn(self):
        # 交换target与eval网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # 选出批量训练的数据
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        # 训练网络
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            }
        )

        # 增量epslion
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # 学习次数加1
        self.learn_step_counter += 1

    # 保存模型
    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    # 加载模型
    def load_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


# 全连接
class DQN_Dense(DQN_Net):

    def __init__(
            self,
            n_actions = 7,
            n_features = 42,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=32,
            batch_size=32,
            hidden_units=[128, 256, 256, 128],
            e_greedy_increment=None,
            output_graph=False,
    ):
        super(DQN_Dense, self).__init__(n_actions, n_features, learning_rate, reward_decay, e_greedy,
                                        replace_target_iter, memory_size, batch_size, hidden_units,
                                        e_greedy_increment, output_graph)
        print(
            'DQN_Dense - lr:%f,reward_decay:%f,e_greedy:%f,replace_target_iter:%d,memory_size:%d,batch_size:%d,e_greedy_increment:%f'
            % (learning_rate, reward_decay, e_greedy, replace_target_iter, memory_size, batch_size, e_greedy_increment))

        # 初始化记忆库 尺寸（memory_size，[s,a,r,s_]）
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # 创建神经网络 target_net,evaluate_net
        self.__build_net()

        # 定义替换 target net 的参数的操作
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_parms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacemtn'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_parms)]

        self.sess = tf.Session()

        # 绘制tensorboard图
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("./logs/", self.sess.graph)

        # 变量初始化
        self.sess.run(tf.global_variables_initializer())


    # 构建网络
    def __build_net(self):
        tf.reset_default_graph()

        # 所有的输入
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s')  # 当前环境
        self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s_')  # 下一个环境
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, ], name='r')  # 奖励
        self.a = tf.placeholder(dtype=tf.float32, shape=[None, ], name='a')  # 选择行为

        # 构建evaluate网络
        with tf.variable_scope('eval_net'):
            self.eval_hidden_layers = []
            for i, units in enumerate(self.hidden_units):
                if i == 0:
                    e = tf.layers.dense(inputs=self.s,
                                        units=units,
                                        activation=tf.nn.relu)
                else:
                    e = tf.layers.dense(inputs=self.eval_hidden_layers[-1],
                                        units=units,
                                        activation=tf.nn.relu)

                bn_e = tf.layers.batch_normalization(inputs=e)

                self.eval_hidden_layers.append(bn_e)


            self.q_eval = tf.layers.dense(inputs=self.eval_hidden_layers[-1] ,
                                          units=self.n_actions,
                                          activation=None,
                                          name='q_eval')

        # 构建target网络(与evaluate网络结构一模一样)
        with tf.variable_scope('target_net'):
            self.target_hidden_layers = []
            for i, units in enumerate(self.hidden_units):
                if i == 0:
                    t = tf.layers.dense(inputs=self.s_,
                                        units=units,
                                        activation=tf.nn.relu)
                else:
                    t = tf.layers.dense(inputs=self.target_hidden_layers[-1],
                                        units=units,
                                        activation=tf.nn.relu)
                # 批量归一化
                bn_t = tf.layers.batch_normalization(inputs=t)

                self.target_hidden_layers.append(bn_t)


            self.q_next = tf.layers.dense(inputs=self.target_hidden_layers[-1],
                                          units=self.n_actions,
                                          activation=None,
                                          name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')

            """
            通过self.q_target = tf.stop_gradient(q_target)，
            将原本为TensorFlow计算图中的一个op（节点）转为一个常量self.q_target，
            这时候对于loss的求导反传就不会传到target net去了。
            """
            self.q_target = tf.stop_gradient(q_target)

        # 不懂
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), tf.cast(self.a, dtype=tf.int32)],
                                 axis=1)

            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        # 损失函数
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(
                    self.q_target,
                    self.q_eval_wrt_a,
                    name='TD_error')
            )
        # 优化方法
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


# 卷积
class DQN_Conv(DQN_Net):

    def __init__(
            self,
            n_actions = 7,
            n_features = 42,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=32,
            batch_size=32,
            hidden_units=[128, 256, 256, 128],
            e_greedy_increment=None,
            output_graph=False,
    ):
        super(DQN_Conv, self).__init__(n_actions, n_features, learning_rate, reward_decay, e_greedy,
                                        replace_target_iter, memory_size, batch_size, hidden_units,
                                        e_greedy_increment, output_graph)
        print(
            'DQN_Conv - lr:%f,reward_decay:%f,e_greedy:%f,replace_target_iter:%d,memory_size:%d,batch_size:%d,e_greedy_increment:%f'
            % (learning_rate, reward_decay, e_greedy, replace_target_iter, memory_size, batch_size, e_greedy_increment))

        # 初始化记忆库 尺寸（memory_size，[s,a,r,s_]）
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # 创建神经网络 target_net,evaluate_net
        self.__build_net()

        # 定义替换 target net 的参数的操作
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_parms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacemtn'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_parms)]

        self.sess = tf.Session()

        # 绘制tensorboard图
        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("./logs/", self.sess.graph)

        # 变量初始化
        self.sess.run(tf.global_variables_initializer())

    # 构建网络
    def __build_net(self):
        tf.reset_default_graph()

        # 所有的输入
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s')  # 当前环境
        self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s_')  # 下一个环境
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, ], name='r')  # 奖励
        self.a = tf.placeholder(dtype=tf.float32, shape=[None, ], name='a')  # 选择行为

        # 修改输入维度
        image_s = tf.reshape(self.s, shape=[-1, int(self.n_features/self.n_actions), self.n_actions, 1],  name='image_s')
        image_s_ = tf.reshape(self.s_, shape=[-1, int(self.n_features/self.n_actions), self.n_actions, 1],  name='image_s_')

        # 构建evaluate网络
        with tf.variable_scope('eval_net'):
            self.eval_hidden_layers = []
            for i, units in enumerate(self.hidden_units):
                if i == 0:
                    e = tf.layers.conv2d(inputs=image_s,
                                         kernel_size=(3, 3),
                                         filters=units,
                                         padding='valid',
                                         activation=tf.nn.tanh)
                else:
                    e = tf.layers.conv2d(inputs=self.eval_hidden_layers[-1],
                                         kernel_size=(3, 3),
                                         filters=units,
                                         padding='valid',
                                         activation=tf.nn.tanh)

                bn_e = tf.layers.batch_normalization(inputs=e)

                self.eval_hidden_layers.append(bn_e)

            e_input_shape = self.eval_hidden_layers[-1].get_shape().as_list()
            e_n_input_units = np.prod(e_input_shape[1:])
            e_flat = tf.reshape(self.eval_hidden_layers[-1], shape=[-1, e_n_input_units])

            e_h = tf.layers.dense(inputs=e_flat,
                                  units=512,
                                  activation=tf.nn.relu)

            self.q_eval = tf.layers.dense(inputs=e_h,
                                          units=self.n_actions,
                                          activation=None,
                                          name='q_eval')

        # 构建target网络(与evaluate网络结构一模一样)
        with tf.variable_scope('target_net'):
            self.target_hidden_layers = []
            for i, units in enumerate(self.hidden_units):
                if i == 0:
                    t = tf.layers.conv2d(inputs=image_s_,
                                         kernel_size=(3, 3),
                                         filters=units,
                                         padding='valid',
                                         activation=tf.nn.tanh)
                else:
                    t = tf.layers.conv2d(inputs=self.target_hidden_layers[-1],
                                         kernel_size=(3, 3),
                                         filters=units,
                                         padding='valid',
                                         activation=tf.nn.tanh)

                bn_t = tf.layers.batch_normalization(inputs=t)

                self.target_hidden_layers.append(bn_t)

            t_input_shape = self.target_hidden_layers[-1].get_shape().as_list()
            t_n_input_units = np.prod(t_input_shape[1:])
            t_flat = tf.reshape(self.target_hidden_layers[-1], shape=[-1, t_n_input_units])

            t_h = tf.layers.dense(inputs=t_flat,
                                  units=512,
                                  activation=tf.nn.relu)

            self.q_next = tf.layers.dense(inputs=t_h,
                                          units=self.n_actions,
                                          activation=None,
                                          name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')

            """
            通过self.q_target = tf.stop_gradient(q_target)，
            将原本为TensorFlow计算图中的一个op（节点）转为一个常量self.q_target，
            这时候对于loss的求导反传就不会传到target net去了。
            """
            self.q_target = tf.stop_gradient(q_target)

        # 不懂
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), tf.cast(self.a, dtype=tf.int32)],
                                 axis=1)

            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        # 损失函数
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(
                    self.q_target,
                    self.q_eval_wrt_a,
                    name='TD_error')
            )
        # 优化方法
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
