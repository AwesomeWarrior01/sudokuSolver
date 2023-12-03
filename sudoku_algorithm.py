class SudokuSolver:

    # Function to check if a number can be placed in a specific row.
    def used_in_row(self, grid, row, num):
        return num in grid[row]

    # Function to check if a number can be placed in a specific column.
    def used_in_col(self, grid, col, num):
        return num in [grid[i][col] for i in range(9)]

    # Function to check if a number can be placed in a specific 3x3 box.
    def used_in_box(self, grid, start_row, start_col, num):
        return num in [grid[i][j] for i in range(start_row, start_row + 3) for j in range(start_col, start_col + 3)]

    # Function to check if it is legal to assign a number to a specific location.
    def check_location_is_safe(self, grid, row, col, num):
        return not self.used_in_row(grid, row, num) and not self.used_in_col(grid, col, num) and not self.used_in_box(grid, row - row % 3, col - col % 3, num)

    # Function to find an empty location in the Sudoku grid.
    def find_empty_location(self, grid, loc):
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    loc[0], loc[1] = row, col
                    return True
        return False

    # Function to find a solved Sudoku.
    def SolveSudoku(self, grid):
        loc = [0, 0]

        # If there is no unassigned location, we are done.
        if not self.find_empty_location(grid, loc):
            return True

        row, col = loc

        # Considering digits from 1 to 9.
        for num in range(1, 10):

            if self.check_location_is_safe(grid, row, col, num):

                # Making a tentative assignment.
                grid[row][col] = num
                # If success, return true.
                if self.SolveSudoku(grid):
                    return True
                # Failure, unmake the assignment and try again.
                grid[row][col] = 0

        # This triggers backtracking.
        return False
    
def runAlgorithm(example_grid):
    # Create an instance of the SudokuSolver class
    solver = SudokuSolver()

    # Solve the Sudoku puzzle
    if solver.SolveSudoku(example_grid):
        pass
    else:
        print("No solution exists")
        # Fill the sudoku board with all zeros. This makes it so that no solution will be shown.
        example_grid = [[0 for _ in range(9)] for _ in range(9)]

if __name__ == "__main__":
# Example Sudoku puzzle
    example_grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    runAlgorithm(example_grid)

