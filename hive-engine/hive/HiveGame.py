from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
import numpy as np

import ctypes
import datetime
import random
from ctypes import *
import numpy as np
from typing import Tuple
import re


f = "../beekeeper/c/build/libhive.so"
lib = CDLL(f)
board_size = c_uint.in_dll(lib, "pboardsize").value
tile_stack_size = c_uint.in_dll(lib, "ptilestacksize").value
max_turns = c_uint.in_dll(lib, "pmaxturns").value - 1


# GLOBAL VARIABLES
TILE_RELATIVE = 0
ABSOLUTE_COORDINATE = 1
SIMPLE = False
WIN = 1
NOT_ENDED = 0
LOSS = 2
BOARD_WIDTH = 26

ACTION_ENCODING = TILE_RELATIVE
USE_SYMMETRIES = True

# Directions in Beekeeper (6 is on top of another tile):
#    0   1
#  2   6   3
#    4   5
# When the board is rotated 60 degrees, direction 0 goes to 2,
# direction 1 goes to 0, direction 2 goes to 4, etc.
ROT60_DIRECTION_PERMUTATION = [2, 0, 4, 1, 5, 3, 6]
ROT60_DIRECTION_PERMUTATION_INVERSE = [1, 3, 0, 5, 2, 4, 6]

# When the board is reflected vertically (up <-> down swapped),
# direction 0-4 get swapped, 1-5 get swapped, and 2, 3 and 6 remain the same.
# This permutation is self-inverse (reflecting twice yields the same board).
FLIP_DIRECTION_PERMUTATION = [4, 5, 2, 3, 0, 1, 6]


# Calculate the action space size according to the used action encoding.
if ACTION_ENCODING == TILE_RELATIVE:
    if SIMPLE:
        # (row, col) = (next_to, tile)
        ACTION_SPACE_SHAPE = (22, 11)
    else:
        # (plane_nr, row, col) = (next_to, direction, tile)
        ACTION_SPACE_SHAPE = (22, 7, 11)
elif ACTION_ENCODING == ABSOLUTE_COORDINATE:
    if SIMPLE:
        # (row, col) = (row, col)
        ACTION_SPACE_SHAPE = (26, 26)
    else:
        # (plane_nr, row, col) = (tile, row, col)
        ACTION_SPACE_SHAPE = (11, 26, 26)

ACTION_SPACE_SIZE = np.prod(ACTION_SPACE_SHAPE)


# To convert the integer according to the tile-format provided by Beekeeper
# to an integer that represents the tile for the numpy-array representation of
# the board to be fed into the neural network.
TILE_DICT = {
    0: 0,   # EMPTY
    1: 1,   # L_ANT
    2: 2,   # L_GRASSHOPPER
    3: 3,   # L_BEETLE
    4: 4,   # L_SPIDER
    5: 5,   # L_QUEEN
    33: 10, # D_ANT
    34: 9,  # D_GRASSHOPPER
    35: 8,  # D_BEETLE
    36: 7,  # D_SPIDER
    37: 6   # D_QUEEN
}


# Interpret the tile encoding from Beekeeper such that they can be used in
# the action encodings for the tile that is moved (integer between 0 and 22).
# These are sorted in such a way that dark pieces with a high value
# have a lower integer encoding as less valuable dark pieces.
# For light pieces it is the other way around.
TILE_DICT_FULL = {
    0: {
        101:    0,  # q1 - DARK PIECES
        97:     1,  # a1
        -95:    2,  # a2
        -31:    3,  # a3
        98:     4,  # g1
        -94:    5,  # g2
        -30:    6,  # g3
        100:    7,  # s1
        -92:    8,  # s2
        99:     9,  # b1
        -93:    10, # b2
        67:     11, # B1 - LIGHT PIECES
        -125:   12, # B2
        68:     13, # S1
        -124:   14, # S2
        66:     15, # G1
        -126:   16, # G2
        -62:    17, # G3
        65:     18, # A1
        -127:   19, # A2
        -63:    20, # A3
        69:     21, # Q1
        0:      22  # EMPTY
    },
    1: {
        69:     0,  # Q1 - LIGHT PIECES
        65:     1,  # A1
        -127:   2,  # A2
        -63:    3,  # A3
        66:     4,  # G1
        -126:   5,  # G2
        -62:    6,  # G3
        68:     7,  # S1
        -124:   8,  # S2
        67:     9,  # B1
        -125:   10, # B2
        99:     11, # b1 - DARK PIECES
        -93:    12, # b2
        100:    13, # s1
        -92:    14, # s2
        98:     15, # g1
        -94:    16, # g2
        -30:    17, # g3
        97:     18, # a1
        -95:    19, # a2
        -31:    20, # a3
        101:    21, # q1
        0:      22, # EMPTY
    },
}


# Same as TILE_DICT_FULL, but light and dark are treated the same,
# and empty does not exist. This is used for the tile that is being placed
# in the tile-relative encoding.
TILE_DICT_CANONICAL = {
    101:    0,  # q1 - DARK PIECES
    97:     1,  # a1
    -95:    2,  # a2
    -31:    3,  # a3
    98:     4,  # g1
    -94:    5,  # g2
    -30:    6,  # g3
    100:    7,  # s1
    -92:    8,  # s2
    99:     9,  # b1
    -93:    10, # b2
    67:     9,  # B1 - LIGHT PIECES
    -125:   10, # B2
    68:     7,  # S1
    -124:   8,  # S2
    66:     4,  # G1
    -126:   5,  # G2
    -62:    6,  # G3
    65:     1,  # A1
    -127:   2,  # A2
    -63:    3,  # A3
    69:     0   # Q1
}


def print_board_pretty(board):
    """
    Print the numpy array board in a pretty manner (similar to
    print_board() from Beekeeper).
    """
    for x in range(26):
        print("  ", end="")
    for x in range(26):
        print("---", end="")
    print()

    for y in range(26):
        for x in range(26 - y - 1):
            print("  ", end="")
        print("/", end="")
        for x in range(26):
            print(" ", end="")
            tile = board[y][x]
            if tile == 0:
                print("  ", end="")
            else:
                tile_str = "0" + str(tile) if tile < 10 else str(tile % 100)
                print(tile_str, end="")

        print("/")
    print(" ", end="")

    for x in range(26):
        print("---", end="")
    print()


def move_encoding_general(node):
    """
    Given a struct node as defined in beekeeper, extract the information
    about the move and calculate the corresponding action encoding.
    """
    if ACTION_ENCODING == TILE_RELATIVE:
        return move_encoding_relative(node)
    else:
        return move_encoding_absolute(node)


def move_encoding_relative(node):
    """
    Return the move encoding based on the tile type (11),
    neighbour/next_to (22) and direction (7) of the move.
    The idea is to have a 22 'neigbour planes' of size 22x7
    (tile, direction). The move encoding is then the index within the
    flattened version of this 3-dimensional array of 22 planes.
    If the simplfied version is used, the direction is ignored.
    """
    board_struct = node.contents.board.contents
    turn = board_struct.turn
    player_to_move = turn % 2
    move_struct = node.contents.move
    tile = move_struct.tile
    next_to = move_struct.next_to
    direction = move_struct.direction
    location = move_struct.location

    # If no moves are available, the last bit (-1'th index) of the valid-moves binary array is set.
    if location == -1:
        return ACTION_SPACE_SIZE

    tile_encoding = TILE_DICT_CANONICAL[tile]
    next_to_encoding = TILE_DICT_FULL[player_to_move][next_to]

    # If the board was empty, we encode this as placing a piece next
    # to itself (this is safe, because it is impossible to place a piece
    # next to itself, so this encoding is unused otherwise).
    if next_to_encoding == 22:
        next_to_encoding = TILE_DICT_FULL[player_to_move][tile]

    if SIMPLE:
        # next_to corresponds to the row within the plane (0-21)
        # tile corresponds to the column within the plane (0-10)
        return (11 * next_to_encoding) + tile_encoding
    else:
        # next_to corresponds to the plane (0-21)
        # direction corresponds to the row within the plane (0-6)
        # tile corresponds to the column within the plane (0-10)
        return (11 * 7 * next_to_encoding) + (11 * direction) + tile_encoding


def move_encoding_absolute(node):
    """
    Return the move encoding based on the tile type (11),
    and location (26x26) of the move. The idea is to have a 22 'neigbour planes'
    of size 22x7 (tile, direction). The move encoding is then the index within
    the flattened version of this 3-dimensional array of 22 planes. If the
    simplified version is used, the tile that is move is ignored.
    """
    board_struct = node.contents.board.contents
    turn = board_struct.turn
    player_to_move = turn % 2
    move_struct = node.contents.move
    tile = move_struct.tile
    location = move_struct.location

    # If no moves are available, the last bit (-1'th index) of the valid-moves binary array is set.
    if location == -1:
        return ACTION_SPACE_SIZE

    tile_encoding = TILE_DICT_CANONICAL[tile]

    if SIMPLE:
        return location
    else:
        # location corresponds to the position within the plane.
        # tile-encoding corresponds to the plane number.
        return (BOARD_WIDTH**2 * tile_encoding) + location


def rotate_board_60(board: np.ndarray):
    """
    Mutate a 2-dimensional numpy array such that the corresponding hexagonal
    hive board gets rotated by 60 degrees. This is done using a sequence of
    three steps. First, we rotate the numpy array by 90 degrees. Next, we
    translate the board such that it will be centered after the rotation.
    Finally, we shift the rows to restore the structure of the board.
    """
    rot_board = np.rot90(board, -1)
    height, width = board.shape
    midway = height // 2 - 1

    nonzero_after = [x + (y - midway) for y, x in zip(*np.nonzero(rot_board))]

    # An empty board does not have to be shifted.
    if nonzero_after:
        midway_before = width // 2 - 1
        midway_after = (min(nonzero_after) + max(nonzero_after)) // 2
        center_shift = midway_before - midway_after
        rot_board = np.roll(rot_board, center_shift, axis=1)

    return np.array([np.roll(row, row_number - midway) for row_number, row in enumerate(rot_board)])


def reflect_board_vertically(board: np.ndarray):
    """
    Mutate a 2-dimensional numpy array such that the corresponding hexagonal
    hive board gets flipped vertically. This is done using a sequence of
    two steps. First, we flip the numpy array by 90 degrees. Finally, we shift
    the rows to restore the structure of the board.
    """
    flipped_board = np.flipud(board)
    height, width = board.shape
    midway = height // 2

    reflected_board = np.array([np.roll(row, row_number - midway) for row_number, row in enumerate(flipped_board)])

    return reflected_board

def rotate_pi_60(pi_board: np.ndarray):
    """
    Mutate the policy vector such that it corresponds to a board that is
    rotated by 60 degrees. This depends on the used action encoding.
    The rotation is not implemented for the absolute coordinate encoding.
    """
    if ACTION_ENCODING == TILE_RELATIVE:
        if SIMPLE:
            return pi_board
        else:
            return pi_board[:, ROT60_DIRECTION_PERMUTATION, :]
    else:
        if SIMPLE:
            return pi_board
        else:
            return pi_board

def reflect_pi_vertically(pi_board: np.ndarray):
    """
    Mutate the policy vector such that it corresponds to a board that is
    reflected vertically. This depends on the used action encoding.
    The reflection is not implemented for the absolute coordinate encoding.
    """
    if ACTION_ENCODING == TILE_RELATIVE:
        if SIMPLE:
            return pi_board
        else:
            return pi_board[:, FLIP_DIRECTION_PERMUTATION, :]
    else:
        if SIMPLE:
            return pi_board
        else:
            return pi_board


class TileStack(Structure):
    _fields_ = [
        ('type', c_ubyte),
        ('location', c_int),
        ('z', c_ubyte)
    ]


class Tile(Structure):
    _fields_ = [
        ('free', c_bool),
        ('type', c_ubyte)
    ]


class Player(Structure):
    _fields_ = [
        ('beetles_left', c_ubyte),
        ('grasshoppers_left', c_ubyte),
        ('queens_left', c_ubyte),
        ('ants_left', c_ubyte),
        ('spiders_left', c_ubyte)
    ]


class Board(Structure):
    _fields_ = [
        ('tiles', Tile * board_size * board_size),
        ('turn', c_int),
        ('players', Player * 2),

        ('queen1_position', c_int),
        ('queen2_position', c_int),

        ('min_x', c_int),
        ('min_y', c_int),
        ('max_x', c_int),
        ('max_y', c_int),

        ('n_stacked', c_byte),
        ('stack', TileStack * tile_stack_size),

        ('move_location_tracker', c_int),

        ('zobrist_hash', c_longlong),
        ('hash_history', c_longlong * 150)
    ]


class List(Structure):
    pass


List._fields_ = [
    ('head', POINTER(List)),
    ('next', POINTER(List)),
    ('prev', POINTER(List)),
]


class MMData(Structure):
    _fields_ = [
        ('mm_value', c_double),
        ('mm_evaluated', c_bool)
    ]


class Move(Structure):
    _fields_ = [
        ('tile', c_byte),
        ('next_to', c_byte),
        ('direction', c_byte),
        ('previous_location', c_int),
        ('location', c_int)
    ]

MINIMAX = 0
MCTS = 1
RANDOM = 2
MANUAL = 3


# The PlayerArguments structure is used to specify to MCTS or Minimax what parameters to use in the search.
# Algorithm can be between 0 and 3;
#   0 - Minimax
#   1 - MCTS
#   2 - Random
#   3 - Manual
# MCTS constant is the constant used for UCB1 to define the exploration factor.
# Time to move is the amount of allotted time to select a move.
# Prioritization is an MCTS playout prioritization strategy to reduce the amount of draws.
# First play urgency is an MCTS enhancement to quickly identify good branches early on.
# Verbose generates more output per algorithm.
# Evaluation function is a switch case for Minimax, it can be 0 or 1;
#   0 - Queen surrounding prioritization
#   1 - Opponent tile blocking prioritization
class PlayerArguments(Structure):
    _fields_ = [
        ('algorithm', c_int),
        ('mcts_constant', c_double),
        ('time_to_move', c_double),
        ('verbose', c_bool),
    ]

#
# The Arguments structure stored for each player what algorithm they use and what parameters to use for this algorithm.
#
class Arguments(Structure):
    _fields_ = [
        ('p1', PlayerArguments),
        ('p2', PlayerArguments),
    ]


class Node(Structure):
    _fields_ = [
        ('children', List),
        ('node', List),
        ('move', Move),
        ('board', POINTER(Board)),
        ('data', c_uint)
    ]

    def print(self):
        lib.print_board(self.board)


# Set return types for all functions we're using here.
lib.game_init.restype = POINTER(Node)
lib.list_get_node.restype = POINTER(Node)
lib.default_init.restype = POINTER(Node)
lib.init_board.restype = POINTER(Board)
lib.performance_testing.restype = ctypes.c_int
lib.minimax.restype = POINTER(Node)
lib.mcts.restype = POINTER(Node)


class Hive:
    """The Hive class stores a node struct from Beekeepers and contains
    several helper-functions."""
    def __init__(self, init_node=None):
        if init_node:
            self.node = init_node
        else:
            self.node = lib.game_init()

        lib.free_children(self.node)
        self.generate_moves()
        self.children = self.get_children()

    def generate_moves(self):
        """
        Expand the game tree by allocating all nodes that correspond to a
        valid move. The list of children is stored within the node.
        """
        lib.generate_moves(self.node)

    def convert_tile(self, tile):
        """
        Interprete the tile encoding used in Beekeeper to the tile encoding
        used in the board representation to be fed into the neural network.
        """
        # Mask out the number of the tile, which are the first 2 bits of the ubyte
        masked_tile = tile % 2**6

        return TILE_DICT[masked_tile]

    def invert_tile_stack(self, tile_stack: int):
        """
        Invert the colors of each tile in a tile stack represented by our
        integer encoding. On input 1083, we output 138 (10 <-> 1, 8 <-> 3).
        """
        str_stack = re.sub('10', 'X', str(tile_stack))
        array_stack = ['10' if c == 'X' else c for c in str_stack]

        return int(''.join([str(11 - int(c)) for c in array_stack]))

    def board_to_nparray(self):
        """
        Convert a board with the tile-encodings of Beekeeper to the
        a numpy-array to be fed into the neural network.
        """
        current_board = self.node.contents.board.contents.tiles

        visible_board = np.array([[self.convert_tile(tile.type) for tile in row]
                                    for row in current_board])

        # Process the tile-stacks. A tile stack is representated by
        # concatenating the digits of the tiles within the tile-stack
        tile_stacks = self.node.contents.board.contents.stack
        tile_stacks_dict = {}

        for tile_stack in tile_stacks:
            # If the location of the tile_stack is -1, the beetle is not placed
            # on top of another tile. In this case, we skip the tile stack.
            if tile_stack.location != -1:
                if tile_stack.location not in tile_stacks_dict:
                    tile_stacks_dict[tile_stack.location] = 0

                tile_stacks_dict[tile_stack.location] += 10**tile_stack.z * self.convert_tile(tile_stack.type)

        for loc, val in tile_stacks_dict.items():
            x = loc % BOARD_WIDTH
            y = loc // BOARD_WIDTH
            num_digits = len(str(val))

            # The leftmost digits are determined by the visible tile on top.
            visible_board[y][x] = visible_board[y][x] * (10**num_digits) + val

        return visible_board

    def move(self):
        """
        Print the boards of all children of the current node.
        """
        for i, child in enumerate(self.children):
            print(f"CHILD {i}:")
            child.contents.print()

    def detach_child(self, new_board, board):
        """
        Detach the child from the game tree and free its memory.
        """
        lib.list_remove(byref(new_board.node.contents.node))
        lib.node_free(board.node)

    def print(self):
        """
        Print the Hive board in a readable way.
        """
        self.node.contents.print()

    def dump(self):
        """"
        Print the Hive board and give some other useful statistics for debugging.
        """
        print("=====================================================================================")
        move_struct = self.node.contents.move

        print(f"Move taken to get to this board state: " +
              f"[tile: {move_struct.tile}, next_to: {move_struct.next_to}, " +
              f"direction: {move_struct.direction}, location: {move_struct.location}]")

        self.print()

        tile_stacks = self.node.contents.board.contents.stack
        for tile_stack in tile_stacks:
            print(f"TILE_STACK: {tile_stack.z, tile_stack.location, tile_stack.type}")
        print("STRUCT CONTENT:")
        board_content = self.node.contents.board.contents
        print(f"TURN: {board_content.turn}")
        print(board_content.players)
        print(board_content.min_x, board_content.min_y, board_content.max_x, board_content.max_y)
        print(board_content.n_stacked)
        print(self.board_to_nparray())
        print(sorted([move_encoding_general(node) for node in self.children]))

        for node in self.children:
            move_struct = node.contents.move
            print(f"tile: {move_struct.tile}, next_to: {move_struct.next_to}, " +
                  f"direction: {move_struct.direction}, location: {move_struct.location}")

    def finished(self):
        """Check whether a player has won from the current board state."""
        return lib.finished_board(self.node.contents.board)

    def get_children(self):
        """
        A generator returning child pointers
        """
        children_list = []

        assert self.node, "Node is NULL."

        if self.node.contents.board.contents.move_location_tracker == 0:
            self.generate_moves()
        elif move_encoding_general(lib.list_get_node(self.node.contents.children.next)) == ACTION_SPACE_SIZE:
            lib.free_children(self.node)
            self.generate_moves()

        head = self.node.contents.children.next

        while ctypes.addressof(head.contents) != ctypes.addressof(self.node.contents.children.head):
            # Get struct offset
            child = lib.list_get_node(head)

            children_list.append(child)

            head = head.contents.next

        return children_list

    def reinitialize(self):
        """
        Reinitilize the current node by free'ing the game tree.
        Allocated memory is re-used for the next game.
        """
        lib.node_free(self.node)

        node = lib.default_init()
        self.node = node
        self.node.contents.board = lib.init_board()
        self.children = self.get_children()

    def get_children_traditional(self):
        """
        Get the children without calling lib.generate_moves() (for minimax/MCTS)
        """
        children_list = []

        assert self.node, "node none"

        head = self.node.contents.children.next

        while ctypes.addressof(head.contents) != ctypes.addressof(self.node.contents.children.head):
            # Get struct offset
            child = lib.list_get_node(head)

            children_list.append(child)

            head = head.contents.next

        return children_list

    def minimax(self):
        """
        Based on the node, return the encoding of the best action according to minimax.
        This is used to compare the self-play based engine against traditional approaches.
        """
        # Remove the children and update the children list of the Hive object.
        lib.free_children(self.node)

        arguments = Arguments()
        arguments.p1.algorithm = MINIMAX
        arguments.p1.verbose = False
        arguments.p1.time_to_move = 1.0
        arguments.p2.algorithm = MINIMAX
        arguments.p2.verbose = False
        arguments.p2.time_to_move = 1.0

        best_move_node = lib.minimax(pointer(self.node), pointer(arguments))

        # update the children list (it still points to the one made by
        # generate_moves() at this point).
        self.children = self.get_children_traditional()
        best_move_encoding = move_encoding_general(best_move_node)

        return best_move_encoding

    def mcts(self):
        """Based on the node, return the encoding of the best action according to minimax.
        This is used to compare the neural network model against traditional approaches."""
        # Remove the children and update the children list of the Hive object.
        lib.free_children(self.node)

        arguments = Arguments()
        arguments.p1.algorithm = MCTS
        arguments.p1.verbose = False
        arguments.p1.time_to_move = 1.0
        arguments.p2.algorithm = MCTS
        arguments.p2.verbose = False
        arguments.p2.time_to_move = 1.0

        best_move_node = lib.mcts(pointer(self.node), pointer(arguments))

        # update the children list (it still points to the one made by
        # generate_moves() at this point).
        self.children = self.get_children_traditional()
        best_move_encoding = move_encoding_general(best_move_node)

        return best_move_encoding


class HiveGame(Game):
    """The HiveGame class defines the game logic of Hive such that it can be efficiently
    integrated into the training infastructure."""
    def __init__(self, n=14):
        self.n = n
        self.reusable_board = None

    def getInitBoard(self) -> Tuple[Hive, np.ndarray]:
        """
        Initialize the board for the game of Hive and store it.
        """
        if not self.reusable_board:
            self.reusable_board = Hive()
        else:
            self.reusable_board.reinitialize()

        new_board = self.reusable_board

        return (new_board, new_board.board_to_nparray())

    def getBoardSize(self):
        """
        Return the size of the board.
        """
        return (self.n, self.n)

    def getActionSize(self) -> int:
        """
        Return the size of the action space. 1 extra move is allocated for the
        case when there are no valid moves.
        """
        return ACTION_SPACE_SIZE + 1


    def getNextState(self, board: Tuple[Hive, np.ndarray], player: int, action: int) -> Tuple[Tuple[Hive, np.ndarray], int]:
        """
        Given a board and the encoding of an action, return the state of the board
        after performing this action on the board.
        """
        board_node, _ = board
        possible_moves = board_node.children

        if move_encoding_general(lib.list_get_node(board[0].node.contents.children.next)) == ACTION_SPACE_SIZE:
            board[0].children = board[0].get_children()
            board_node, _ = board
            possible_moves = board_node.children

        for node in possible_moves:
            if action == move_encoding_general(node):
                new_hive = Hive(node)
                new_board = (new_hive, new_hive.board_to_nparray())

                return (new_board, -player)

        new_hive = Hive(possible_moves[0])
        new_board = (new_hive, new_hive.board_to_nparray())

        return (new_board, -player)

        raise AssertionError(f"Action encoding was not valid... a = {action}")

    def getValidMoves(self, board: Tuple[Hive, np.ndarray], player: int) -> List:
        """
        Return a binary vector that contains a 1 in places that correspond
        to an valid action from the given board state.
        """
        board_node, _ = board
        possible_moves = board_node.children

        if move_encoding_general(lib.list_get_node(board[0].node.contents.children.next)) == ACTION_SPACE_SIZE:
            board[0].children = board[0].get_children()
            board_node, _ = board
            possible_moves = board_node.children

        valids = np.zeros(self.getActionSize())

        for node in possible_moves:
            move_encoding = move_encoding_general(node)
            valids[move_encoding] = 1

        return np.array(valids)

    def getGameEnded(self, board: Tuple[Hive, np.ndarray], player: int) -> int:
        """
        Return 0 if the game has not ended, 1 if player 1 won,
        and 2 if player 1 lost.
        """
        board, _ = board

        game_ended = board.finished()

        # We multiply by player to invert the game outcome player = -1 is given.
        if game_ended == NOT_ENDED:
            return 0
        elif game_ended == WIN:
            return 1 * player
        elif game_ended == LOSS:
            return -1 * player

        # If the game is a draw, then we declare that black is the winner, because
        # white has the advantage over black in general.
        return -1e-2

    def getCanonicalForm(self, board: Tuple[Hive, np.ndarray], player: int):
        """"
        Return the board from the perspective of the player who is to make a
        move. If the player is 1, simply return the given board, otherwise
        invert the colors of each tile and return the board.
        """
        if player == 1:
            return board
        else:
            board, board_array = board

            new_board_array = (11 - board_array) % 11

            # Process the tile-stacks
            tile_stacks = board.node.contents.board.contents.stack
            locs = [tile_stack.location for tile_stack in tile_stacks]

            for loc in set(locs):
                if loc != -1:
                    x = loc % BOARD_WIDTH
                    y = loc // BOARD_WIDTH

                    new_board_array[y][x] = board.invert_tile_stack(board_array[y][x])

            return (board, new_board_array)

    def getSymmetries(self, board: Tuple[Hive, np.ndarray], pi):
        """
        Return a list of tuples consisting of a board and a policy vector,
        each of these tuples correspond to one of the 12 symmetries of a hive
        board (6 rotations, 2 reflections).

        Note: This function is not completely coherent with the description that
        is given in game.py for memory efficiency reasons. Namely, we the boards
        in each tuple only consist of the numpy part of the board and not
        the board struct from Duncan Kampert's framework.
        """
        if not USE_SYMMETRIES:
            return [(board[1], pi)]

        board_struct, np_board = board

        if not SIMPLE and ACTION_ENCODING == TILE_RELATIVE:
            pi_board = np.reshape(pi[:-1], ACTION_SPACE_SHAPE)
        else:
            # Symmetry exploitation is not yet
            return [(board[1], pi)]

        l = []
        cur_board = np_board
        cur_pi = pi_board


        # First 6 symmetries
        for i in range(6):
            l += [(cur_board, list(cur_pi.ravel()) + [pi[-1]])]

            cur_board = rotate_board_60(cur_board)
            cur_pi = rotate_pi_60(cur_pi)

        cur_board = reflect_board_vertically(cur_board)
        cur_pi = reflect_pi_vertically(cur_pi)

        # Next 6 symmetries (after flipping the board)
        for i in range(6):
            l += [(cur_board, list(cur_pi.ravel()) + [pi[-1]])]

            cur_board = rotate_board_60(cur_board)
            cur_pi = rotate_pi_60(cur_pi)

        return l


    def stringRepresentation(self, board: Tuple[Hive, np.ndarray]):
        """
        Return a hashable string representation that uniquely identifies the
        current state of the board.
        """
        board_obj, board_array = board
        board_as_string = ":".join(board_array.astype(np.str).flatten().tolist())

        return board_as_string

    @staticmethod
    def display(board: Tuple[Hive, np.ndarray]):
        board, _ = board
        board.print()
