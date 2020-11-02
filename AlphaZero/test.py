import AlphaZero.Arena as Arena
from AlphaZero.MCTS import MCTS
from ConnectXEnv.ConnectX_Game import ConnectX_AlphaZero
from ConnectXEnv.ConnectX_play import *
from AlphaZero.Net.NNet import NNetWrapper as NNet

import numpy as np
from AlphaZero.utils import *



g = ConnectX_AlphaZero()

# all players
rp = RandomPlayer(g).play
gp = OneStepLookaheadConnect4Player(g).play
hp = HumanConnect4Player(g).play


# nnet players
n = NNet(g)
n.load_checkpoint('./temp','best.pth.tar')

args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n, args)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


arena = Arena.Arena(n1p, gp, g, display=ConnectX_AlphaZero.display)

print(arena.playGames(50, verbose=False))


