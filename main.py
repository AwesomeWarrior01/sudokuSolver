from digitRec import *
from sudoku_algorithm import *
from sudokuFiltering import *
import copy

if __name__ == "__main__":
    # Filter image
    warped = SudokuFilter()
    # Digit Recognizer
    digits = digitRec(warped)
    print(digits)
    digits_stored = copy.deepcopy(digits)
    runAlgorithm(digits) # Note that 'digits' is passed by ref, so it is also the output of the function.
    for i in range(9):
        print(digits[i])

    for j in range(9):
        for k in range(9):
            digits[j][k] = digits[j][k] - digits_stored[j][k] # eliminates any numbers already shown on the sudoku board.
    Overlay_solution(warped, digits)
    
