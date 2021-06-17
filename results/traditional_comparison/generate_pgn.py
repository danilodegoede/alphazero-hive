import numpy as np

# The results of a round-robin tournement of the engines, each tuple has a
# format (<p1>, <p2>, <num_won>), where <num_won> is the number of games
# won by p1 out of 100 games played.
results = [("4h", "untrained", 72), ("4h", "random", 64), ("4h", "nn", 34), ("4h", "mcts", 27), ("4h", "mm", 8),
           ("untrained", "random", 42), ("untrained", "nn", 14), ("untrained", "mcts", 6), ("untrained", "mm", 2),
           ("random", "nn", 23), ("random", "mcts", 8), ("random", "mm", 3),
           ("nn", "mcts", 38), ("nn", "mm", 21),
           ("mcts", "mm", 32)]

NUM_GAMES = 100

res_dict_white = {0: "0-1", 1: "1-0"}
res_dict_black = {0: "1-0", 1: "0-1"}


def wins_array(num_games, num_wins):
    """Create a random binary array where each element corresponds to a single game from a series of games.
    A value 1 indicates a win for the current player, and a 0 indicates a loss."""
    result = np.zeros(num_games, dtype=np.int8)
    result[:num_wins] = 1
    np.random.shuffle(result)

    return result


def pgn_entry(white, black, result):
    """Convert a tuple of the form (<p1>, <p2>, <result>) to PGN format.
    p1 is the name of the player that is playing white, p2 is the name of
    the player that is playing black."""
    return f"[White \"{white}\"][Black \"{black}\"][Result \"{result}\"] 1. c4 Nf6"


def generate_matches(results):
    """Generate a random sequence of matches """
    matches = []

    for result in results:
        match_results = wins_array(NUM_GAMES, result[-1])

        for i, res in enumerate(match_results):
            if i < 50:
                white = result[0]
                black = result[1]
                matches.append((white, black, res_dict_white[res]))
            else:
                white = result[1]
                black = result[0]
                matches.append((white, black, res_dict_black[res]))

    return [m for i in range(100) for m in matches[i::100]]



def main():
    matches = generate_matches(results)

    for match in matches:
        print(pgn_entry(*match))


if __name__ == '__main__':
    main()
