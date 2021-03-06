import matplotlib.pyplot as plt


def mcts_explore_compare():
    mcts_05 = [
        74,44,58,54,18,48,48,32,52,24,64,84,86,33,24,52,54,32,40,28,42,52,18,22,30,58,72,22,74,32,30,16,74,24,42,62,36,20,41,38,24,24,88,34,48,20,71,34,52,21,70,34,54,76,44,15,20,22,52,34,38,44,42,50,28,74,37,14,36,56,44,50,52,38,68,52,58,22,22,22,18,66,]
    mcts_1 = [
            80,26,82,46,40,82,32,22,64,14,82,70,22,70,24,38,40,36,38,58,24,38,84,80,44,28,28,24,40,86,86,26,68,16,44,40,50,76,22,30,86,20,44,64,31,44,52,76,98,22,82,54,52,69,40,14,30,32,71,50,28,22,54,32,24,82,54,28,30,20,56,78,46,36,30,50,50,80,24,19,28,19,24,]
    mcts_2 = [
        22, 34, 32, 22, 26, 38, 22, 36, 20, 28, 44, 40, 38, 20, 56, 86, 58, 36, 70, 30, 40, 12, 51, 24, 38, 58, 38, 66,
        80, 98, 16, 40, 32, 90, 52, 24, 26, 20, 72, 36, 94, 30, 46, 32, 82, 22, 44, 29, 61, 26, 52, 58, 28, 18, 14, 26,
        24, 40, 26, 46, 98, 79, 49, 22, 40, 66, 26, 36, 96, 34, 32, 97, 51, 24, 72, 60, 46, 52, 50, 25, 85, 80, 62, 68,
        38, ]
    mcts_5 = [
        83, 78, 34, 31, 30, 22, 42, 92, 64, 38, 24, 36, 25, 78, 60, 52, 22, 59, 38, 18, 26, 40, 52, 56, 32, 54, 30, 46,
        72, 67, 24, 38, 32, 24, 85, 40, 28, 14, 32, 30, 54, 23, 38, 26, 14, 50, 66, 38, 36, 50, 28, 16, 14, 40, 19, 80,
        18, 34, 32, 36, 34, 20, 16, 30, 16, 22, 72, 40, 22, 20, 94, ]
    mcts_10 = [
        78, 75, 30, 64, 36, 34, 50, 30, 82, 42, 20, 28, 24, 80, 84, 28, 27, 28, 28, 34, 56, 14, 99, 66, 30, 86, 82, 36,
        26, 94, 21, 86, 72, 24, 82, 34, 85, 35, 46, 38, 50, 50, 40, 39, 34, 26, 40, 78, 34, 20, 32, 59, 72, 34, 84, 28,
        58, 26, 64, 14, 30, 36, 24, 88, 64, 28, 30, 46, 78, 92, 20, 96, 24, 60, 26, 26, 20, 80, ]
    mcts_20 = [
        34, 60, 66, 38, 34, 16, 46, 40, 34, 76, 83, 18, 50, 46, 32, 26, 34, 52, 26, 26, 44, 38, 30, 46, 42, 24, 52, 42,
        85, 40, 54, 92, 88, 86, 22, 40, 32, 38, 38, 25, 81, 46, 27, 30, 62, 62, 56, 32, 32, 62, 56, 20, 62, 26, 32, 60,
        28, 26, 82, 68, 34, 32, 40, 40, 32, 16, 36, 74, 90, 64, 30, 14, 24, 93, 38, ]
    mcts_50 = [
        34,16,28,14,18,22,28,68,66,36,54,70,58,34,58,74,74,14,48,60,32,94,16,14,42,34,30,30,22,60,38,64,62,26,98,30,28,50,64,64,52,43,28,26,22,26,66,30,32,20,48,50,52,52,18,64,60,40,52,76,46,20,72,44,30,42,20,34,44,48,]

    mctss = [mcts_05, mcts_1, mcts_2, mcts_5, mcts_10, mcts_20, mcts_50]
    for mcts in mctss:
        print(len(mcts), sum(mcts) / len(mcts))

    plt.hist(mctss, bins=5)
    plt.legend(["0.5", "1", "2", "5", "10", "20", "50"])
    plt.show()


def main():
    mcts_explore_compare()
    exit(0)

    g = 101
    random = [84, 14, 62, g, g, 68, 88, g, 62, g, g, g, 46, 46, g, g, 62, g, g, 24, 90, 54, g, g, 80, 52, 64, 93, 38,
              g, 86, g, 46, 34, g, 74, 50, 68, 22, g, 84, 12, g, 76, g, g, 72, g, 14, g, 70, g, 76, 92, 70, g, g, 54,
              g, 82]
    uct1dot4 = [72, g, g, g, 32, 76, 68, g, g, 64, g, 54, g, 27, g, 84, g, g, 44, 71, g, g, g, 46, g, 94, 88, 56, 54,
                58,
                90, g, g, g, 96, 97, 54, 16, 34, 20, g, 99, 26, g, g, g, 85, 38, 68, g, g, 66, 20, 62, 56, g, 36, g, g,
                32, g, g, 28, 58, g, 91, 68, g, 60, g, 12]

    uct100 = [g, 74, g, g, 76, 58, 46, 32, g, 74, 35, g, g, g, 78, g, g, g, 30, g, g, g, 46, 60, g, 40, g, g, 24, g, g,
              61, g, 24, g, 80, 34, 38, 20, g, 70, g, 77, 82, g, g, g, g, 64]

    mm_allmoves = [88, 26, 19, 46, 58, 58, 16, 40, 98, 26, 30, 48, 28, 36, 78, 24, 50, 79, 34, 59, 32, 52, 26, 56, 78,
                   72, 48, 40, 18, 36, 86, 44, 22]

    mm_noants = [32, 32, 52, 28, 30, 48, 23, 24, 24, 42, 34, 58, 24, 22, 30, 30, 22, 22, 42, 40, 26, 32, 42, 30, 23, 86,
                 24, 54, 34, 38, 90, 20, 38, 30, 22, 48, 32, 30, 30, 36, 58, 24, 50]

    mcts_uct100_5sec = [22, 56, 97, 26, 15, 58, 38, 50, 24, 72, 46, 76, 22, 16, 13, 42, 24, 28, 34, 24, 24, 23, 64, 96,
                        18, 32, 14, 80, 56]

    mm_noants_5sec = [16, 32, 34, 34, 36, 38, 26, 32, 72, 22, 72, 26, 34, 62, 28, 34, 32, 26, 22, 24, 22, 30, 20, 52,
                      26, 50, 28, 26, 36]

    mm_20sec = [40, 23, 32, 28, 26, 16, 40, 36, 51, 18, 25, 22, 46, 24, 38, 42, 24, 37, 34, 34, 34, 28, 16, 14, 32, 36,
                30, 62, 28, 44, 20, 84, 26]

    mcts_20sec = [26, 36, 24, 30, 72, 26, 14, 26, 30, 88, 48, 28, 99, 14, 26, 28, 52, 36, 34, 85, 20, 98, 92, 26, 17,
                  88, 28, 36, 58, 69, 48, 62, 26, 21, 40, 70, 26, 42, 26, 46, 58, 54, 50, 18]

    random = list(filter(lambda x: x != g, random))
    uct1dot4 = list(filter(lambda x: x != g, uct1dot4))
    uct100 = list(filter(lambda x: x != g, uct100))

    print(len(random), sum(random) / len(random))
    print(len(uct1dot4), sum(uct1dot4) / len(uct1dot4))
    print(len(uct100), sum(uct100) / len(uct100))
    print(len(mm_allmoves), sum(mm_allmoves) / len(mm_allmoves))
    print(len(mm_noants), sum(mm_noants) / len(mm_noants))
    print(len(mcts_uct100_5sec), sum(mcts_uct100_5sec) / len(mcts_uct100_5sec))
    print(len(mm_noants_5sec), sum(mm_noants_5sec) / len(mm_noants_5sec))
    print(len(mm_20sec), sum(mm_20sec) / len(mm_20sec))
    print(len(mcts_20sec), sum(mcts_20sec) / len(mcts_20sec))

    plt.hist([random, uct1dot4, uct100, mm_allmoves, mm_noants], bins=5)
    plt.legend(["Random", "UCT c=1.4", "UCT c=100", "MM Full", "MM No ants"])
    plt.show()

    plt.hist([mm_noants_5sec, mcts_uct100_5sec, mm_20sec, mcts_20sec], bins=5)
    plt.legend(["MCTS UCT c=100 5 sec", "MM no ants 5 sec", "MM 20 sec", "MCTS 20 sec"])
    plt.show()


if __name__ == "__main__":
    main()

"""
 () - () - ()
 
     0 1 0
 M = 1 0 1
     0 1 0
  
 v = [1 0 0]
 
 v * M = [0 1 0] = v2
 
 v = v + v2 = [1 1 0]
 
 v * M = [1 1 1] = v3
 
 
 
 
 



"""
