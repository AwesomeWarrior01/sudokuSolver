from digitRec import *
from sudoku_algorithm import *

if __name__ == "__main__":
    digits = digitRec()
    print(digits)
    answer = runAlgorithm(digits)
    for i in range(9):
        print(answer[i])
