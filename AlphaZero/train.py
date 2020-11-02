import logging

import coloredlogs

from AlphaZero.Coach import Coach
from ConnectXEnv.ConnectX_Game import ConnectX_AlphaZero as Game
from AlphaZero.Net.NNet import NNetWrapper as nn
from AlphaZero.utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 200,
    'numEps': 20,              # Number of complete self-play games to simulate during a new iteration. # 500
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 10,          # Number of games moves for MCTS to simulate. # 800
    'arenaCompare': 50,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 4,                 # CPUCT parameter
    'numItersForTrainExamplesHistory': 20,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),

})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
