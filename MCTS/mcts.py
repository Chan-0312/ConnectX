import numpy as np
from tqdm import tqdm
import pickle
import random



# 定义结点数据结构
class Node:
    '''
    # MCTS结点结构
    '''
    def __init__(self, state, winning, move, parent):
        self.parent = parent   # 父结点
        self.move = move       # 父结点怎么移动而来
        self.win = 0           # 胜利次数
        self.games = 0         # 当前节点访问次数
        self.children = None   # 孩子节点
        self.state = state     # 当前状态
        self.winner = winning  # 不为0则表示游戏结束结点

    # 设置孩子节点
    def set_children(self, children):
        self.children = children

    # 计算uct值
    def get_uct(self):
        if self.games == 0:
            # 未访问过
            return None
        # 根据UCT公式返回
        return (self.win/self.games) + np.sqrt(2*np.log(self.parent.games)/self.games)

    # 选择最佳移动行为
    def select_move(self):
        # 其没有孩子结点
        if self.children is None:
            return None, None

        # 胜利的孩子结点
        winners = [child for child in self.children if child.winner]
        if len(winners) > 0:
            return winners[0], winners[0].move

        # 否则按概率选择最佳的结点和移动
        games = [child.win/child.games if child.games > 0 else 0 for child in self.children]
        # print(games)
        # for child in self.children:
        #     print(child.games, child.win, child.winner)
        best_child = self.children[np.argmax(games)]
        return best_child, best_child.move

    # 获取孩子结点
    def get_children_with_move(self, move):
        if self.children is None:
            return None
        for child in self.children:
            if child.move == move:
                return child

        raise Exception('Not existing child')


class mcts(object):
    """
    MCTS类
    - 需要传入game类：需要包含方法
        - getInitBoard()             # 获取初始环境状态
        - getValidMoves(board)       # 获取有效行为
        - get_player_to_play(board)  # 根据棋盘自动获取游戏手(1,-1)
        - getNextState(board, move)  # 获取移动的下一个状态

    """
    def __init__(self,
                 game,          # 游戏类
                 ):

        self.game = game
        self.root = Node(self.game.getInitBoard(), 0, None,  None)

    # 随机策略仿真
    def random_play(self, board):
        while True:
            moves = self.game.getValidMoves(board)
            if len(moves) == 0:
                return 0
            selected_move = random.choice(moves)
            player_to_play = self.game.get_player_to_play(board)
            board, winner = self.game.getNextState(board, selected_move)
            if np.abs(winner) > 0:
                return player_to_play

    # 随机策略仿真，加入了规则，能够更加容易学会规则
    def random_play_improved(self, board):

        # 查找胜利的移动
        def get_winning_moves(board, moves, player):
            return [move for move in moves if self.game.getNextState(board, move, player=player)[1]]

        # If can win, win
        while True:
            # 获取有效的移动
            moves = self.game.getValidMoves(board)
            if len(moves) == 0:
                return 0

            player_to_play = self.game.get_player_to_play(board)

            winning_moves = get_winning_moves(board, moves, player_to_play)
            loosing_moves = get_winning_moves(board, moves, -player_to_play)

            if len(winning_moves) > 0:
                # selected_move = winning_moves[0]
                selected_move = random.choice(winning_moves)
            elif len(loosing_moves) == 1:
                # selected_move = loosing_moves[0]
                selected_move = random.choice(loosing_moves)
            else:
                selected_move = random.choice(moves)

            board, winner = self.game.getNextState(board, selected_move)
            if np.abs(winner) > 0:
                # print(board)
                return player_to_play



    # 每一个结点进行自我学习
    def learn(self, mcts=None, numMCTSSims=200, policy='random_play_improved'):

        if mcts == None:
            mcts = self.root
        for i in range(numMCTSSims):
            node = mcts
            # 选择部分，走到叶子节点
            while node.children is not None:  # 结束不为None
                # 返回每个行为的uct值
                ucts = [child.get_uct() for child in node.children]
                if None in ucts:
                    node = random.choice(node.children)
                else:
                    node = node.children[np.argmax(ucts)]

            # 获取当前状态的有效移动列表
            moves = self.game.getValidMoves(node.state)
            if len(moves) > 0:
                # 该结点以及被胜利了？
                if node.winner == 0:
                    # 所有新的状态[(),((棋盘，胜利状态)，行为)]
                    states = [(self.game.getNextState(node.state, move), move) for move in moves]
                    # 设置其孩子节点
                    node.set_children(
                        [Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in
                         states])

                    # 仿真
                    # 如果孩子结点有胜利结点这选择胜利结点
                    winner_nodes = [n for n in node.children if n.winner]
                    if len(winner_nodes) > 0:
                        # 随机选择一个胜利结点
                        node = winner_nodes[0]
                        victorious = node.winner
                    else:
                        # 否则随机选择一个孩子结点
                        node = random.choice(node.children)
                        # print(node.state)
                        # 随机模拟直到游戏结束
                        if policy == 'random_play':
                            victorious = self.random_play(node.state)
                        else:
                            victorious = self.random_play_improved(node.state)

                else:
                    # 已经是胜利结点
                    victorious = node.winner

                # 反向传播
                parent = node
                while parent is not None:
                    # 所有父结点访问次数+1
                    parent.games += 1
                    if victorious != 0 and self.game.get_player_to_play(parent.state) != victorious:
                        # 胜利的结点才加+1
                        parent.win += 1
                    parent = parent.parent
            else:
                print('no valid moves, expended all')

        return mcts


    # 自行对战训练
    def self_play(self,numEps=100,numMCTSSims=200,display=False):
        for i in tqdm(range(numEps), desc='self_play'):
            node = self.root
            while True:
                if i < numEps/4:
                    policy = 'random_play'
                else:
                    policy = 'random_play_improved'
                self.learn(node, numMCTSSims, policy)
                new_node, move = node.select_move()
                board, winner = self.game.getNextState(node.state, move)
                if display:
                    self.game.display(board)
                node = new_node
                assert np.sum(node.state - board) == 0, node.state
                if winner != 0:
                    break

            self.save_model('./temp_root.pkl')
        pass

    # 保存模型
    def save_model(self, path):
        pickle.dump(self.root, open(path, 'wb'))
        print('save model', path)

    # 保存模型
    def load_model(self, path):
        self.root = pickle.load(open(path, 'rb'))
        print('load model', path)