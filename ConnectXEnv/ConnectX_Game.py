import numpy as np

# 默认参数
DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4


# 棋盘及规则
class Board(object):

    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH
        self.win_length = win_length or DEFAULT_WIN_LENGTH

        # 残局
        if np_pieces is None:
            self.np_pieces = np.zeros([self.height, self.width], dtype=np.int)
        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.height, self.width)

    # 落子
    def add_stone(self, column, player):
        "Create copy of board containing new stone."
        available_idx, = np.where(self.np_pieces[:, column] == 0)
        if len(available_idx) == 0:
            raise ValueError("Can't play column %s on board %s" % (column, self))

        self.np_pieces[available_idx[-1]][column] = player

    # 获取有效移动
    def get_valid_moves(self):
        "Any zero value in top row in a valid move"
        return self.np_pieces[0] == 0

    # 获取胜利状态
    def get_win_state(self):
        # 这里是分开成两张棋盘，1棋盘，-1棋盘
        for player in [-1, 1]:
            player_pieces = self.np_pieces == -player
            if (self._is_straight_winner(player_pieces) or
                self._is_straight_winner(player_pieces.transpose()) or
                self._is_diagonal_winner(player_pieces)):
                return True, -player

        # 判断平局
        if not self.get_valid_moves().any():
            return True, None

        # 没有结束
        return False, None

    # 创建残局副本
    def with_np_pieces(self, np_pieces):
        """Create copy of board with specified pieces."""
        if np_pieces is None:
            np_pieces = self.np_pieces
        return Board(self.height, self.width, self.win_length, np_pieces)

    # 判断对角胜利
    def _is_diagonal_winner(self, player_pieces):
        """Checks if player_pieces contains a diagonal win."""
        win_length = self.win_length
        for i in range(len(player_pieces) - win_length + 1):
            for j in range(len(player_pieces[0]) - win_length + 1):
                if all(player_pieces[i + x][j + x] for x in range(win_length)):
                    return True
            for j in range(win_length - 1, len(player_pieces[0])):
                if all(player_pieces[i + x][j - x] for x in range(win_length)):
                    return True
        return False

    # 判断列胜利
    def _is_straight_winner(self, player_pieces):
        """Checks if player_pieces contains a vertical or horizontal win."""
        run_lengths = [player_pieces[:, i:i + self.win_length].sum(axis=1)
                       for i in range(len(player_pieces) - self.win_length + 2)]
        return max([x.max() for x in run_lengths]) >= self.win_length

    # 返回一个对象的描述信息
    def __str__(self):
        return str(self.np_pieces)



# AlphaZero的游戏类
class ConnectX_AlphaZero(object):

    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        self._base_board = Board(height, width, win_length, np_pieces)

    def getInitBoard(self):
        return self._base_board.np_pieces

    def getBoardSize(self):
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self):
        return self._base_board.width

    def getNextState(self, board, player, action):
        # 返回更新后的棋盘副本，原始棋盘未修改。
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)
        # 返回新棋盘和交换先后手
        return b.np_pieces, -player

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    # 输赢判断
    def getGameEnded(self, board, player):
        b = self._base_board.with_np_pieces(np_pieces=board)
        is_ended,winner = b.get_win_state()
        if is_ended:
            if winner is None:
                # draw has very little value.
                return 1e-4
            elif winner == player:
                return +1
            elif winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ')
        else:
            # 0 used to represent unfinished game.
            return 0

    # 交换棋盘
    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * player

    # 返回两个对称的棋盘与对应行为选择的先验概率
    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]


    # 字符串表示
    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        row = board.shape[0]
        column = board.shape[1]
        print()
        print('+' * (column * 3 + 3))
        for i in range(row):
            print(end='+ ')
            for j in range(column):
                print(2 if board[i,j]== -1 else board[i,j], end='  ')
            print(end='+\n')
        print(end='++')
        for i in range(column):
            print(i, end='++')
        print(end='+\n')
        print()


# mcts专用的游戏类
class ConnectX_MCTS(object):

    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        self._base_board = Board(height, width, win_length, np_pieces)

    def getInitBoard(self):
        return self._base_board.np_pieces

    def getActionSize(self):
        return self._base_board.width

    # 根据当前棋盘判断下棋手
    def get_player_to_play(self, board):
        player_1 = np.sum(board == 1)
        player_2 = np.sum(board == -1)
        if player_1 <= player_2:
            return 1
        else:
            return -1


    # 获取下一个状态,与胜利
    def getNextState(self, board, action, player=None):
        # 返回更新后的棋盘副本，原始棋盘未修改。
        if player == None:
            player = self.get_player_to_play(board)

        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)

        is_ended, winner = b.get_win_state()

        # 返回新棋盘
        return b.np_pieces, winner if winner != None else 0

    # 返回有效移动的列表
    def getValidMoves(self, board):
        mask = self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()
        return [i for i in range(self.getActionSize()) if mask[i]]

    # 显示
    @staticmethod
    def display(board):
        row = board.shape[0]
        column = board.shape[1]
        print()
        print('+' * (column * 3 + 3))
        for i in range(row):
            print(end='+ ')
            for j in range(column):
                print(2 if board[i,j]== -1 else board[i,j], end='  ')
            print(end='+\n')
        print(end='++')
        for i in range(column):
            print(i, end='++')
        print(end='+\n')
        print()



from ConnectXEnv.myAgent import agents
import gym
import random
import copy

# Gym环境专用类
class ConnectX_Gym(gym.Env):

    def __init__(self,
                 rows=6,          # 棋盘行数
                 columns=7,       # 棋盘列数
                 inarow=4,        # 棋子胜利连续数
                 switch=-1,       # 先后手 1 表示先手  2 表示后手 -1表示随机
                 agent_level=-2,  # 游戏代理难度[0-n]表示对应强度 0最小,-1表示等概率随机选择，-2表示非等概率选择(强度越高的被选概率越大)
                 ):

        self.board = Board(height=rows, width=columns, win_length=inarow)

        self.set_switch = switch
        self.set_agent_level = agent_level

        # 环境配置
        self.configuration = {'rows': rows,
                              'columns': columns,
                              'inarow': inarow,
                              'agent_level': agent_level,
                              'mark': switch
                              }

        # 行为空间
        self.action_space = gym.spaces.Discrete(columns)

        # 环境空间
        self.observation_space = gym.spaces.Box(low=-1,
                                                high=1,
                                                shape=(rows, columns, 1),
                                                dtype=np.int)

        # 奖励范围
        '''
        错误行为 -10
        胜利 1
        失败 -1
        其他 1/(rows*columns)
        '''
        self.reward_range = (-10, 1)

        # 机器人移动状态
        self._robot_move = -1

    # 执行一步
    def step(self, action):

        # 无效的移动
        valid_moves = self.board.get_valid_moves()
        if valid_moves[action] == False:
            return self.board.np_pieces.reshape(self.observation_space.shape).copy(), -10, True, {}

        self.board.add_stone(column=action, player=self.configuration['mark'] if self.configuration['mark'] != 2 else -1)

        done, winner = self.board.get_win_state()

        # 玩家胜利
        if done:
            if winner != None:
                return self.board.np_pieces.reshape(self.observation_space.shape).copy(), 1, done, {}
            else:
                # 平局
                return self.board.np_pieces.reshape(self.observation_space.shape).copy(), 1/(100*self.configuration['rows']*self.configuration['columns']), done, {}

        # 电脑下棋
        obs = self.board.np_pieces.reshape(self.observation_space.shape).copy()
        conf = copy.deepcopy(self.configuration)
        conf['mark'] = 2 if self.configuration['mark'] == 1 else 1

        action = agents[conf['agent_level']](obs, conf)
        self._robot_move = action
        self.board.add_stone(action, player=conf['mark'] if conf['mark'] != 2 else -1)

        done, winner = self.board.get_win_state()

        # 电脑胜利
        if done:
            if winner != None:
                return self.board.np_pieces.reshape(self.observation_space.shape).copy(), -1, done, {}
            else:
                # 平局
                return self.board.np_pieces.reshape(self.observation_space.shape).copy(), 1 / (
                            100*self.configuration['rows'] * self.configuration['columns']), done, {}

        return self.board.np_pieces.reshape(self.observation_space.shape).copy(), 1 / (
                            100*self.configuration['rows'] * self.configuration['columns']), done, {}

    # 重置环境
    def reset(self):
        switch = self.set_switch
        level = self.set_agent_level
        if switch == 1 or switch == 2:
            pass
        else:
            switch = random.choice([1, 2])

        if level >= 0 and level < len(agents):
            # 指定代理
            pass
        elif level == -2:
            # 非等概率选择代理
            level_pro = [i + 1 for i in range(len(agents))]
            level_pro = [i / sum(level_pro) for i in level_pro]
            for i in range(len(level_pro) - 1):
                level_pro[i + 1] += level_pro[i]

            pro = random.random()
            for i in range(len(level_pro)):
                if pro < level_pro[i]:
                    level = i
                    break
            pass
        else:
            # 等概率选择代理
            level = random.choice(range(len(agents)))
            pass

        self.configuration['mark'] = switch
        self.configuration['agent_level'] = level
        # 重置棋盘
        self.board = self.board.with_np_pieces(np.zeros(shape=(self.configuration['rows'], self.configuration['columns']), dtype=np.int))

        # 重置机器人移动位置
        self._robot_move = -1
        # 机器人先手
        if switch == 2:

            obs = self.board.np_pieces.reshape(self.observation_space.shape).copy()
            conf = copy.deepcopy(self.configuration)
            conf['mark'] = 2 if self.configuration['mark'] == 1 else 1

            action = agents[self.configuration['agent_level']](obs, conf)

            self._robot_move = action

            self.board.add_stone(action, player=conf['mark'] if conf['mark'] != 2 else -1)

        return self.board.np_pieces.reshape(self.observation_space.shape).copy()

    # 绘制环境
    def render(self, mode='xsc', **kwargs):
        board = self.board.np_pieces
        row = board.shape[0]
        column = board.shape[1]
        print()
        print('+' * (column * 3 + 3))
        for i in range(row):
            print(end='+ ')
            for j in range(column):
                print(2 if board[i, j] == -1 else board[i, j], end='  ')
            print(end='+\n')
        print(end='++')
        for i in range(column):
            print(i, end='++')
        print(end='+\n')
        print()
        return

    # 获取机器人运动位置
    def get_robot_move(self):
        return self._robot_move


