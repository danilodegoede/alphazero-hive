import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import csv

from .HiveNNet import HiveNNet as onnet

import torchvision

# DATA REPRESENTATION METHODS:
ORIGINAL = 0 # 0 to 10
SYMMETRIC = 1 # -5 to 5
SIMPLE = 2 # 1, -1, 10, -10
SPATIAL_PLANES = 3
SPATIAL_PLANES_4 = 4

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,
    'data_representation_method': SPATIAL_PLANES_4,
    'num_layers': 4, # Only 4, 6 and 8 are supported.
    'results': 'xperiment_results'
})

STATE_TRANSFORMATIONS_SYMMETRIC = {
    1: 1, # L_ANT
    2: 2, # L_GRASSHOPPER
    3: 3, # L_BEETLE
    4: 4, # L_SPIDER
    5: 5, # L_QUEEN
    0: -1, # D_ANT
    9: -2, # D_GRASSHOPPER
    8: -3, # D_BEETLE
    7: -4, # D_SPIDER
    6: -5 # D_QUEEN
}

# Board representation where every dark piece is a negative number, and every
# light piece is positive number. To detect queens, they are made 10/-10.
STATE_TRANSFORMATIONS_SIMPLE = {
    1: 1, # L_ANT
    2: 1, # L_GRASSHOPPER
    3: 1, # L_BEETLE
    4: 1, # L_SPIDER
    5: 10, # L_QUEEN
    0: -1, # D_ANT
    9: -1, # D_GRASSHOPPER
    8: -1, # D_BEETLE
    7: -1, # D_SPIDER
    6: -10 # D_QUEEN
}

PLANE_LOCATION = {
    1: 0, # L_ANT
    2: 0, # L_GRASSHOPPER
    3: 0, # L_BEETLE
    4: 0, # L_SPIDER
    5: 1, # L_QUEEN
    0: 3, # D_ANT
    9: 3, # D_GRASSHOPPER
    8: 3, # D_BEETLE
    7: 3, # D_SPIDER
    6: 2 # D_QUEEN
}

STATE_TRANSFORMATIONS_PLANE = {
    1: 1, # L_ANT
    2: 2, # L_GRASSHOPPER
    3: 3, # L_BEETLE
    4: 4, # L_SPIDER
    5: 1, # L_QUEEN
    0: 1, # D_ANT
    9: 2, # D_GRASSHOPPER
    8: 3, # D_BEETLE
    7: 4, # D_SPIDER
    6: 1 # D_QUEEN
}

class NNetWrapper(NeuralNet):
    def __init__(self, game, plot_nn_architecture=False):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.plot_nn_architecture = plot_nn_architecture

        if args.cuda:
            self.nnet.cuda()

    def to_spatial_plane(self, board):
        """Convert a 2D board to a 3D representation of binary spatial planes,
        where each plane encodes the positions of one specific tile type."""
        new_board = np.zeros((10, 14, 14))

        for i, row in enumerate(board):
            for j, tile in enumerate(row):
                if tile >= 1:
                    tile = tile % 10
                    if tile == 0:
                        tile = 10

                    new_board[tile - 1, i, j] = 1

        return new_board

    def to_spatial_plane_4(self, board):
        """Convert a 2D board to a 3D representation of four binary spatial
        planes, where each plane encodes the positions of one specific tile
        type (light queen, dark queen, light piece, dark piece)."""
        new_board = np.zeros((4, 14, 14))

        for i, row in enumerate(board):
            for j, tile in enumerate(row):
                if tile >= 1:
                    tile = tile % 10

                    plane_index = PLANE_LOCATION[tile]
                    tile_encoding = STATE_TRANSFORMATIONS_PLANE[tile]

                    new_board[plane_index, i, j] = tile_encoding

        return new_board

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = [board[6:-6, 6:-6] for board in boards]

                if args.data_representation_method == SYMMETRIC:
                    boards = [np.array([[0 if tile == 0 else STATE_TRANSFORMATIONS_SYMMETRIC[tile % 10] for tile in row] for row in board]) for board in boards]
                elif args.data_representation_method == SIMPLE:
                    boards = [np.array([[0 if tile == 0 else STATE_TRANSFORMATIONS_SIMPLE[tile % 10] for tile in row] for row in board]) for board in boards]
                elif args.data_representation_method == SPATIAL_PLANES:
                    boards = [self.to_spatial_plane(board) for board in boards]
                elif args.data_representation_method == SPATIAL_PLANES_4:
                    boards = [self.to_spatial_plane_4(board) for board in boards]
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))

                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Write loss to CSV file.
            with open(f"{args.results}/stage2.csv", 'a') as outfile:
                csvwriter = csv.writer(outfile)
                csvwriter.writerow([pi_losses, v_losses])

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()
        board = board[6:-6, 6:-6]

        # Convert the board according to the data representation method.
        if args.data_representation_method == SYMMETRIC:
            board = np.array([[0 if tile == 0 else STATE_TRANSFORMATIONS_SYMMETRIC[tile % 10] for tile in row] for row in board])
        elif args.data_representation_method == SIMPLE:
            board = np.array([[0 if tile == 0 else STATE_TRANSFORMATIONS_SIMPLE[tile % 10] for tile in row] for row in board])
        elif args.data_representation_method == SPATIAL_PLANES:
            board = self.to_spatial_plane(board)
        elif args.data_representation_method == SPATIAL_PLANES_4:
            board = self.to_spatial_plane_4(board)

        # Convert board to float and move it to the GPU if possible.
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()

        # Reshpae the board depending on the data representation method.
        if args.data_representation_method == SPATIAL_PLANES:
            board = board.view(10, self.board_x, self.board_y)
        elif args.data_representation_method == SPATIAL_PLANES_4:
            board = board.view(4, self.board_x, self.board_y)
        else:
            board = board.view(1, self.board_x, self.board_y)

        # Feed the prepared board to the NN and retrieve the output,
        # which is the policy vector pi and the value v.
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
