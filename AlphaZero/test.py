import sys
sys.path.append('..')

from AlphaZero.MCTS import MCTS
from ConnectXEnv.ConnectX_Game import ConnectX_AlphaZero, ConnectX_Gym
from AlphaZero.Net.NNet import NNetWrapper as NNet
from tqdm import tqdm
import numpy as np
from AlphaZero.utils import *


# 和人类玩
'''
player 1 表示人类先手， 2 表示人类后手
round 表示游戏回合
'''
def play_with_human(alpha_zero_action, player=1, round=1):
    game = ConnectX_AlphaZero()
    player -= 1
    for i in range(round):
        board = game.getInitBoard()
        step = 0
        game.display(board)
        while True:
            # 人类下
            if (step % 2) == player:
                print('Please input:', [i for i in range(game.getActionSize()) if game.getValidMoves(board, 0)[i]])
                move = int(input())

                board, _ = game.getNextState(board, 1 if player == 0 else -1, move)
            else:
                board, _ = game.getNextState(board, 1 if player == 1 else -1, alpha_zero_action(board))

            winner = game.getGameEnded(board, 1 if player == 0 else 1)
            game.display(board)

            if winner == 1:
                print('You won !')
                break
            elif winner == -1:
                print('You lose!')
                break

            step += 1

# 和gym代理玩
def play_with_coumpter(alpha_zero_action, player=-1, agent_level=-1, round=5, display = False):
    env = ConnectX_Gym(switch=player, agent_level=agent_level)
    won_count = 0
    lose_count = 0
    for i in tqdm(range(round), desc='play'):
        obs = env.reset()
        while True:
            if display:
                env.render()

            action = alpha_zero_action(obs.reshape(6,7))

            obs_, reward, done, _ = env.step(action)

            obs = obs_
            if done:
                if display:
                    env.render()

                if reward == -1:
                    print('You lose!')
                    lose_count += 1
                elif reward == 1:
                    print('You won!')
                    won_count += 1
                elif reward == -10:
                    exit()
                break
    print('%d-%d-%d'%(won_count,round-(won_count+lose_count), lose_count))
    pass


g = ConnectX_AlphaZero()
# 神经网络
n = NNet(g)
n.load_checkpoint('./temp','best.pth.tar')
mcts = MCTS(g, n, dotdict({'numMCTSSims': 50, 'cpuct':1.0}))
# 执行选择
alpha_zero_action = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

# play_with_coumpter(alpha_zero_action, player=1, agent_level=2, round=100)
# play_with_coumpter(alpha_zero_action, player=2, agent_level=2, round=100)
# play_with_coumpter(alpha_zero_action, player=1, agent_level=1, round=100)
# play_with_coumpter(alpha_zero_action, player=2, agent_level=1, round=100)

play_with_human(alpha_zero_action,1)