import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# DATA REPRESENTATION METHODS:
ORIGINAL = 0 # 0 to 10
SYMMETRIC = 1 # -5 to 5
SIMPLE = 2 # 1, -1, 10, -10
SPATIAL_PLANES = 3
SPATIAL_PLANES_4 = 4

class HiveNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(HiveNNet, self).__init__()

        if self.args.data_representation_method == SPATIAL_PLANES:
            self.conv1 = nn.Conv2d(10, args.num_channels, 3, stride=1, padding=1)
        elif self.args.data_representation_method == SPATIAL_PLANES_4:
            self.conv1 = nn.Conv2d(4, args.num_channels, 3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv5 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv6 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv7 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv8 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)
        self.bn5 = nn.BatchNorm2d(args.num_channels)
        self.bn6 = nn.BatchNorm2d(args.num_channels)
        self.bn7 = nn.BatchNorm2d(args.num_channels)
        self.bn8 = nn.BatchNorm2d(args.num_channels)

        if self.args.num_layers == 4:
            self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 4096)
        elif self.args.num_layers == 6:
            self.fc1 = nn.Linear(args.num_channels*(self.board_x-8)*(self.board_y-8), 4096)
        elif self.args.num_layers == 8:
            self.fc1 = nn.Linear(args.num_channels*(self.board_x-12)*(self.board_y-12), 4096)

        self.fc_bn1 = nn.BatchNorm1d(4096)

        self.fc2 = nn.Linear(4096, 2048)
        self.fc_bn2 = nn.BatchNorm1d(2048)

        self.fc3 = nn.Linear(2048, self.action_size)

        self.fc4 = nn.Linear(2048, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        if self.args.data_representation_method == SPATIAL_PLANES:
            s = s.view(-1, 10, self.board_x, self.board_y)               # batch_size x 1 x board_x x board_y
        elif self.args.data_representation_method == SPATIAL_PLANES_4:
            s = s.view(-1, 4, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        else:
            s = s.view(-1, 1, self.board_x, self.board_y)                   # batch_size x 1 x board_x x board_y

        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        total_stride = 4

        if self.args.num_layers > 4:
            s = F.relu(self.bn5(self.conv5(s)))                      # batch_size x num_channels x (board_x-6) x (board_y-6)
            s = F.relu(self.bn6(self.conv6(s)))                      # batch_size x num_channels x (board_x-8) x (board_y-8)
            total_stride += 4
        if self.args.num_layers > 6:
            s = F.relu(self.bn7(self.conv7(s)))                      # batch_size x num_channels x (board_x-10) x (board_y-10)
            s = F.relu(self.bn8(self.conv8(s)))                      # batch_size x num_channels x (board_x-12) x (board_y-12)
            total_stride += 4

        s = s.view(-1, self.args.num_channels*(self.board_x-total_stride)*(self.board_y-total_stride))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
