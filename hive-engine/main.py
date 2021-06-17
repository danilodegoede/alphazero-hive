import logging

import coloredlogs

from Coach import Coach
from hive.HiveGame import HiveGame as Game
from hive.pytorch.NNet import NNetWrapper as nn
from hive.pytorch.NNet import args as nn_args
from utils import *
import json

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 100,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        # TOOD: Was 15 (For every self-play game, after 15 moves only the best actions are taken)
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 0.8,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,
    'results': 'xperiment_results' # Do not end this with '/'.
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

    # Write the training parameters and neural network parameters to a JSON file.
    with open(f"{args.results}/training-parameters.json", 'w+') as outfile:
        json.dump(args, outfile, indent=4)

    with open(f"{args.results}/nn-parameters.json", 'w+') as outfile:
        json.dump(nn_args, outfile, indent=4)

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
