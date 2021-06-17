import logging

from tqdm import tqdm

import numpy as np
from Arena import Arena
from MCTS import MCTS
from hive.HiveGame import HiveGame as Game
from hive.pytorch.NNet import NNetWrapper as nn
from Coach import Coach
import csv
from utils import *

log = logging.getLogger(__name__)

args = dotdict({
    'numIters': 20,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        # TOOD: Was 15 (For every self-play game, after 15 moves only the best actions are taken)
    'updateThreshold': 0.5,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 50,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 0.8,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'results': 'xperiment_results' # Do not end this with '/'.
})

g = Game()

def random_move_baseline(board):
    """"When given a board, return a random action given that board."""
    valids = g.getValidMoves(board, 1)
    num_valid_moves = np.sum(valids)

    # normalise valids to make it a probability distribution.
    valids = valids.astype('float64') / num_valid_moves

    return np.random.choice(g.getActionSize(), p=valids)

def minimax_baseline(board):
    """
    Given a board, return the encoding of the best action
    according to Beekeeper's minimax implementation.
    """
    board_struct, board_array = board
    best_action_encoding = board_struct.minimax()
    return best_action_encoding

def mcts_baseline(board):
    """
    Given a board, return the encoding of the best action
    according to Beekeeper's MCTS implementation.
    """
    board_struct, board_array = board
    best_action_encoding = board_struct.mcts()
    return best_action_encoding

def usability_real_life():
    """
    Retrieve the thinking time per move while varying the number of MCTS
    simulations.
    """
    nnet1 = nn(g)
    nnet2 = nn(g)
    num_games = 10

    for num_mcts_sims in range(100, 2501, 100):
        args.numMCTSSims = num_mcts_sims
        mcts1 = MCTS(g, nnet1, args)
        mcts2 = MCTS(g, nnet2, args)

        arena = Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp=0)),
                      lambda x: np.argmax(mcts2.getActionProb(x, temp=0)) , g, time_moves=True, args=args)

        wins_random, wins_zero, draws = arena.playGames(num_games)

        print(wins_zero, wins_random, draws)

def traditional_comparison():
    """
    Used to pit different engines, such as minimax, MCTS, self-play and random,
    against each other.
    """
    nnet1 = nn(g)

    # Uncomment if to load the weights of a trained network.
    # nnet1.load_checkpoint("./temp/", "best.pth.tar")

    mcts1 = MCTS(g, nnet1, args)
    arena = Arena(lambda x: random_move_baseline(x),
                  lambda x: np.argmax(mcts1.getActionProb(x, temp=0)) , g)

    wins_random, wins_zero, draws = arena.playGames(args.arenaCompare)

    print(wins_zero, wins_random, draws)

    with open(f"{args.results}/traditional_comparison.csv", 'a') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow([wins_zero, wins_random, draws])


def main():
    log.info('Loading %s...', Game.__name__)
    log.info('Loading %s...', nn.__name__)

    traditional_comparison()


if __name__ == '__main__':
    main()
