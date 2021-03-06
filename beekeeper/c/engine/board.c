//
// Created by duncan on 01-02-21.
//

#include "board.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>


unsigned int pboardsize = BOARD_SIZE;
unsigned int ptilestacksize = TILE_STACK_SIZE;
unsigned int pmaxturns = MAX_TURNS;

/*
 * Initialize the board.
 */
struct board *init_board() {
    struct board *board = calloc(1, sizeof(struct board));
    board->turn = 0;
    board->move_location_tracker = 0;
    board->n_stacked = 0;
    board->light_queen_position = -1;
    board->dark_queen_position = -1;

    for (int i = 0; i < 2; i++) {
        board->players[i].ants_left = N_ANTS;
        board->players[i].beetles_left = N_BEETLES;
        board->players[i].queens_left = N_QUEENS;
        board->players[i].grasshoppers_left = N_GRASSHOPPERS;
        board->players[i].spiders_left = N_SPIDERS;
    }

    board->min_x = board->min_y = BOARD_SIZE;
    board->max_x = board->max_y = 0;

    board->zobrist_hash = 0;

    memset(&board->stack, -1, TILE_STACK_SIZE * sizeof(struct tile_stack));

    board->has_updated = 0;

    return board;
}



void get_min_x_y(struct board* board, int* min_x, int* min_y) {
    int lower_x = *min_x;
    int lower_y = *min_y;
    *min_x = *min_y = BOARD_SIZE;
    for (int y = lower_y; y < BOARD_SIZE; y++) {
        for (int x = lower_x; x < *min_x; x++) {
            struct tile* tile = &board->tiles[y * BOARD_SIZE + x];
            if (tile->type == EMPTY) continue;

            // Set min_y to the first non-empty tile row
            if (*min_y == BOARD_SIZE) *min_y = y;

            if (x < *min_x) *min_x = x;
        }
    }
}

void get_max_x_y(struct board* board, int* max_x, int* max_y) {
    int upper_y = *max_y;
    int upper_x = *max_x;
    *max_x = *max_y = 0;
    for (int y = upper_y; y >= 0; y--) {
        for (int x = upper_x; x >= *max_x; x--) {
            struct tile* tile = &board->tiles[y * BOARD_SIZE + x];
            if (tile->type == EMPTY) continue;

            // Set min_y to the first non-empty tile row
            if (*max_y == 0) *max_y = y;

            if (x > *max_x) *max_x = x;
        }
    }
}

// This is to update the x/y boundaries
// of the board every move, because we want to recenter the board every move.
void force_set_bounds(struct board *board) {
    board->min_y = board->min_x = board->max_x = board->max_y = -1;
    // Set min Y
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board->tiles[y * BOARD_SIZE + x].type != EMPTY) {
                board->min_y = y;
                break;
            }
        }
        if (board->min_y != -1) break;
    }
    // Set max Y
    for (int y = BOARD_SIZE - 1; y >= 0; y--) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            if (board->tiles[y * BOARD_SIZE + x].type != EMPTY) {
                board->max_y = y;
                break;
            }
        }
        if (board->max_y != -1) break;
    }
    for (int x = 0; x < BOARD_SIZE; x++) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (board->tiles[y * BOARD_SIZE + x].type != EMPTY) {
                board->min_x = x;
                break;
            }
        }
        if (board->min_x != -1) break;
    }
    for (int x = BOARD_SIZE - 1; x >= 0; x--) {
        for (int y = 0; y < BOARD_SIZE; y++) {
            if (board->tiles[y * BOARD_SIZE + x].type != EMPTY) {
                board->max_x = x;
                break;
            }
        }
        if (board->max_x != -1) break;
    }
}

/*
 * Translates the board to the center coordinate space
 */
int translate_board(struct board *board) {
    force_set_bounds(board);
    int offset = board->min_y * BOARD_SIZE + board->min_x;
    int moffset = board->max_y * BOARD_SIZE + board->max_x;
    int size = (moffset - offset) + 1;

    int to_x = (BOARD_SIZE / 2) - (board->max_x - board->min_x + 1) / 2;
    int to_y = (BOARD_SIZE / 2) - (board->max_y - board->min_y + 1) / 2;

    // Move all the tile tracking structs the same amount as the rest of the board.
    int translate_offset = (to_y * BOARD_SIZE + to_x) - offset;
    if (board->light_queen_position != -1)
        board->light_queen_position += translate_offset;
    if (board->dark_queen_position != -1)
        board->dark_queen_position += translate_offset;
    for (int i = 0; i < TILE_STACK_SIZE; i++) {
        if (board->stack[i].location != -1) {
            board->stack[i].location += translate_offset;
        }
    }

    // Copy data into temp array
    struct tile t[BOARD_SIZE * BOARD_SIZE] = {0};
    void* temp = &t;

    memcpy(temp + (to_y * BOARD_SIZE + to_x) * sizeof(struct tile),
           ((void*)&board->tiles) + offset * sizeof(struct tile),
           (size) * sizeof(struct tile)
    );

    memset(&board->tiles, 0, BOARD_SIZE * BOARD_SIZE * sizeof(struct tile));
    // Copy data back into main array after clearing data.
    memcpy(&board->tiles, temp, BOARD_SIZE * BOARD_SIZE * sizeof(struct tile));

    int xdiff = to_x - board->min_x;
    int ydiff = to_y - board->min_y;

    board->min_x += xdiff;
    board->max_x += xdiff;
    board->min_y += ydiff;
    board->max_y += ydiff;

    return translate_offset;
}


/*
 * Translates the board to 2,2 coordinate space
 */
void translate_board_22(struct board *board) {
    int min_x = BOARD_SIZE;
    int min_y = BOARD_SIZE;
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < min_x; x++) {
            struct tile tile = board->tiles[y * BOARD_SIZE + x];
            if (tile.type == EMPTY) continue;

            // Set min_y to the first non-empty tile row
            if (min_y == BOARD_SIZE) min_y = y;

            if (min_x > x) min_x = x;
        }
    }

    int offset = min_y * BOARD_SIZE + min_x;
    int size = (BOARD_SIZE * BOARD_SIZE - offset);

    // Move all the tile tracking structs the same amount as the rest of the board.
    int translate_offset = (2 * BOARD_SIZE + 2) - offset;
    if (board->light_queen_position != -1)
        board->light_queen_position += translate_offset;
    if (board->dark_queen_position != -1)
        board->dark_queen_position += translate_offset;
    for (int i = 0; i < TILE_STACK_SIZE; i++) {
        if (board->stack[i].location != -1) {
            board->stack[i].location += translate_offset;
        }
    }

    // Copy data into temp array
    struct tile t[BOARD_SIZE * BOARD_SIZE] = {0};
    void* temp = &t;

    memcpy(temp + (2 * BOARD_SIZE + 2) * sizeof(struct tile),
           ((void*)&board->tiles) + offset * sizeof(struct tile),
           (size - (2 * BOARD_SIZE + 2)) * sizeof(struct tile)
    );

    memset(&board->tiles, 0, BOARD_SIZE * BOARD_SIZE * sizeof(struct tile));
    // Copy data back into main array after clearing data.
    memcpy(&board->tiles, temp, BOARD_SIZE * BOARD_SIZE * sizeof(struct tile));
}

// Return true if the queen is surrounded by at least 4 tiles
bool is_surrounded(struct board* board, int y, int x) {
    int p;
    int num_surrouned = 0;
    int* points = get_points_around(y, x);
    for (p = 0; p < 6; p++) {
        if (board->tiles[points[p]].type != EMPTY) {
            num_surrouned++;
        }
    }

    if (num_surrouned >= 4) return true;
    return false;
}

// To get the number of tiles that surround the queen (game rule simplification)
int num_surrounded(struct board* board, int y, int x) {
    int p;
    int num_surrouned = 0;
    int* points = get_points_around(y, x);
    for (p = 0; p < 6; p++) {
        if (board->tiles[points[p]].type != EMPTY) {
            num_surrouned++;
        }
    }

    return num_surrouned;
}

/*
 * Checks if the board is in a finished position
 * Meaning; either one of the two queens is completely surrounded.
 * Returns 1 if player 1 won, 2 if player 2 won, 0 if nobody won yet.
 * Or 3 if its a draw by force (both queens surrounded), or by repetition.
 */
int finished_board(struct board *board) {
    int x, y;
    int res = 0;
    if (board->light_queen_position != -1) {
        // Check queen 1
        x = board->light_queen_position % BOARD_SIZE;
        y = board->light_queen_position / BOARD_SIZE;
        if (is_surrounded(board, y, x)) {
            res = 2;
        }
    }

    if (board->dark_queen_position != -1) {
        // Check queen 2
        x = board->dark_queen_position % BOARD_SIZE;
        y = board->dark_queen_position / BOARD_SIZE;
        if (is_surrounded(board, y, x)) {
            if (res == 0)
                res = 1;
            else
                res = 3;
        }
    }

    // When the game reaches the turn limit, we check who
    // has the most tiles surrounding the enemy. If both have the same number,
    // then black wins.
    if (board->turn == MAX_TURNS - 1) {
        x = board->light_queen_position % BOARD_SIZE;
        y = board->light_queen_position / BOARD_SIZE;

        int light_queen_surrounded = num_surrounded(board, y, x);

        x = board->dark_queen_position % BOARD_SIZE;
        y = board->dark_queen_position / BOARD_SIZE;

        int dark_queen_surrounded = num_surrounded(board, y, x);

        return light_queen_surrounded > dark_queen_surrounded ? 2 : 1;
    }

    // Check draw by repetition.
    // int sum = 0;
    // for (int i = 0; i < board->turn; i++) {
    //     if (board->hash_history[i] == board->zobrist_hash) sum++;

    //     if (sum == 3) res = 3;
    // }
    return res;
}

/*
 * Prints the given Hive board to standard output.
 * Mainly for testing purposes
 */
void print_board(struct board *board) {
    for (int x = 0; x < BOARD_SIZE; x++) {
        printf("  ");
    }
    for (int x = 0; x < BOARD_SIZE; x++) {
        printf("---");
    }
    printf("\n");

    for (int y = 0; y < BOARD_SIZE; y++) {
        // Add spaces padding for the display
        for (int x = 0; x < BOARD_SIZE - y - 1; x++) {
            printf("  ");
        }
        printf("/");

        for (int x = 0; x < BOARD_SIZE; x++) {
            printf(" ");

            struct tile tile = board->tiles[y * BOARD_SIZE + x];
            int type = tile.type & (COLOR_MASK | TILE_MASK);
            int n = (tile.type & NUMBER_MASK) >> NUMBER_SHIFT;
            if (type == EMPTY) {
                printf(" ");
            } else if (type == L_ANT) {
                printf("A");
            } else if (type == L_GRASSHOPPER) {
                printf("G");
            } else if (type == L_BEETLE) {
                printf("B");
            } else if (type == L_SPIDER) {
                printf("S");
            } else if (type == L_QUEEN) {
                printf("Q");
            } else if (type == D_ANT) {
                printf("a");
            } else if (type == D_GRASSHOPPER) {
                printf("g");
            } else if (type == D_BEETLE) {
                printf("b");
            } else if (type == D_SPIDER) {
                printf("s");
            } else if (type == D_QUEEN) {
                printf("q");
            }

            // Add number after tile
            if (n > 0) {
//                n = board->tiles[y * BOARD_SIZE + x].free;
                printf("%d", n);
            } else {
                printf(" ");
            }
        }

        printf("/\n");
    }
    printf(" ");
    for (int x = 0; x < BOARD_SIZE; x++) {
        printf("---");
    }
    printf("\n");
}

