# imports
import os
import numpy as np
from io import StringIO

def legal_board(grid):
    """Determines if the board is legal:
    For this, the board must have no duplicate numbers in a row, column or 3x3 box.
    
    Args:
        grid (Array of int64): 9x9 sudoku grid, where 0 means blank square

    Returns:
        board_legality (bool): legality of the grid
    """
    
    board_legality = True

    for i in range(9):
        row_vals, row_count = np.unique(grid[i,:], return_counts=True)
        col_vals, col_count = np.unique(grid[:,i], return_counts=True)
        
        # this returns numbers and counts within each 3x3 subgrid
        block_vals, block_count = np.unique(grid[3 * (i//3) : 3 * (i//3) + 3, 3 * (i%3) : 3 * (i%3) + 3], return_counts=True)
       
        row_count = row_count[row_vals != 0]
        row_vals = row_vals[row_vals != 0]
        col_count = col_count[col_vals != 0]
        col_vals = col_vals[col_vals != 0]
        block_count = block_count[block_vals != 0]
        block_vals = block_vals[block_vals != 0]
        
        if len(row_count) > 0 and np.max(row_count) > 1:
            board_legality = False
            print('Duplicated', row_vals[np.argmax(row_count)], 'on row', i)
            break
        elif len(col_count) > 0 and np.max(col_count) > 1:
            board_legality = False
            print('Duplicated', col_vals[np.argmax(col_count)], 'on column', i)
            break
        elif len(block_count) > 0 and np.max(block_count) > 1:
            board_legality = False
            print('Duplicated', block_vals[np.argmax(block_count)], 'on block', i//3, i%3)
            break

    return board_legality

def initial_pos_values(grid):
    """Fill up the 3 dimensional array with possibilities.
    For each blank space, the options are filled in based on what is in the same
    column, row and sub grid.
    
    Args:
        grid (Array of int64): 9x9 sudoku grid, where 0 means blank square

    Returns:
        vals (Array of float64): 9x9x10 array, where a 1 means that the number is a possibility
    """
    
    blank_spaces = np.argwhere(grid == 0)
    vals = np.zeros((9,9,10))
    
    for space in blank_spaces:

        # block refers to 3x3 sub grid
        block_row = np.floor_divide(space[0],3)
        block_col = np.floor_divide(space[1],3)
        
        block_vals = np.unique(grid[block_row*3:block_row*3+3, block_col*3:block_col*3+3])
        col_vals = np.unique(grid[:,space[1]])
        row_vals = np.unique(grid[space[0],:])

        combined_vals = np.unique(np.concatenate((block_vals, col_vals, row_vals)))

        vals[space[0], space[1], :] = np.full(10, 1)
        vals[space[0], space[1], combined_vals] = 0

    return vals

def single_options(grid_new_b, possible_values_c):
    """Update grids where there are spaces that have only one option 
    Designed to save time and avoid the need for creating another level of the tree
    
    Args:
        grid_new_b (Array of int64): 9x9 sudoku grid, where 0 means blank square
        possible_values_c (Array of int64): 9x9x10 array, where a 1 means that the number is a possibility

    Returns:
        grid_new_b (Array of int64): 9x9 sudoku grid, where 0 means blank square
        possible_values_c (Array of int64): 9x9x10 array, where a 1 means that the number is a possibility
        num_possible_values_c (Array of float64): 9x9 grid where the number means how many options can be put in that square
    """
    
    while True:
        num_possible_values_c = np.sum(possible_values_c, axis=2)
        
        if 1 not in num_possible_values_c:
            break
        
        blank_spaces = np.argwhere(num_possible_values_c == 1) 
        new_values_b = np.argmax(possible_values_c, axis=2)[blank_spaces[:,0], blank_spaces[:,1]]
        grid_new_b[blank_spaces[:,0], blank_spaces[:,1]] = new_values_b
        
        c = np.array([blank_spaces[:,0], blank_spaces[:,1], new_values_b])
        possible_values_c[:, c[1,:], c[2,:]] = 0 
        possible_values_c[c[0,:], :, c[2,:]] = 0 
        
        grid_row_p = (np.repeat((3 * (blank_spaces[:,0] // 3)), 9).reshape((len(blank_spaces),9)) + np.arange(9) // 3).reshape(-1)
        grid_col_p = (np.repeat((3 * (blank_spaces[:,1] // 3)), 9).reshape((len(blank_spaces),9)) + np.arange(9) % 3).reshape(-1)
        possible_values_c[grid_row_p, grid_col_p, np.repeat(new_values_b, 9)] = 0
 
        dummy = np.zeros((len(new_values_b) * 9, 3), dtype = int)
        dummy[:,0] = grid_row_p
        dummy[:,1] = grid_col_p
        dummy[:,2] = np.repeat(new_values_b, 9)
    return grid_new_b, possible_values_c, num_possible_values_c

# select best squares by finding which update would rule out the most options
# this provides a sense of how much information would be gained 
# these branches are selected first
def depth_first_search(temp_grid, temp_possible_values, temp_coordinates, temp_value, depth=1):
    """Select best squares by finding which update would rule out the most options
    This provides a sense of how much information would be gained and these branches are selected first
    
    Args:
        temp_grid (Array of int64): 9x9 sudoku grid, where 0 means blank square
        temp_possible_values (Array of int64): 9x9x10 array, where a 1 means that the number is a possibility
        temp_coordinates (tuple): Starting square coordinates
        temp_value (int): Starting value
        depth: Depth in depth first search tree
        
    Returns:
        glob_sol (Array of int64): Global variable of the finished grid
        glob_fin (bool): Global variable of the state of whether the grid is finished
    """

    global glob_sol                         # sudoku grid
    global glob_fin                         # sudoku solved state
    
    if glob_fin:                            # sudoku is solved                            
        return                              # exit tree
   
    if depth == 0:
        temp_num_possible_values = np.sum(temp_possible_values, axis=2)
    else:    
        temp_grid[temp_coordinates] = temp_value
        temp_possible_values[temp_coordinates[0],temp_coordinates[1],:] = np.zeros(10)
        temp_possible_values[temp_coordinates[0], :, temp_value] = 0
        temp_possible_values[:, temp_coordinates[1], temp_value] = 0 
        temp_possible_values[3 * (temp_coordinates[0]//3) : 3 * (temp_coordinates[0]//3) + 3, 3 * (temp_coordinates[1]//3) : 3 * (temp_coordinates[1]//3) + 3, temp_value] = 0
        temp_num_possible_values = np.sum(temp_possible_values, axis=2)

        if 1 in temp_num_possible_values:
            temp_grid, temp_possible_values, temp_num_possible_values = single_options(temp_grid, temp_possible_values)
            temp_num_possible_values = np.sum(temp_possible_values, axis=2)
            
    # if there are 81 possible numbers, then grid is solved unless there has been an error
    if np.count_nonzero(temp_grid) == 81:    
        
        # a double check that if each row and column add up to 45 then grid is solved
        if np.all(np.sum(temp_grid, axis = 0) == 45) and np.all(np.sum(temp_grid, axis = 1) == 45):
            if legal_board(temp_grid):
                glob_sol = temp_grid
                glob_fin = 1
                print('solved!')
                print(glob_sol)
                return
            else:
                return
    
    if np.count_nonzero(temp_num_possible_values) == 0:
        return
    
    
    best_value = np.min(temp_num_possible_values[temp_num_possible_values > 0])
    best_squares = np.where(temp_num_possible_values == best_value)
    best_squares = np.array(list(zip(best_squares[0], best_squares[1])))
    
    if len(best_squares) > 1:
        highest_score = -1
        for square in best_squares:
            score = np.sum(temp_num_possible_values[square[0], :] == best_value) + \
                    np.sum(temp_num_possible_values[:, square[1]] == best_value) #+ \
    
            if highest_score < score:
                highest_score = score
                best_square = (square)
    else:
        best_square = (best_squares[0])

    values_b = list(np.where(temp_possible_values[best_square[0], best_square[1]] != 0)[0])
    for value_b in values_b:
        depth_first_search(np.copy(temp_grid), np.copy(temp_possible_values), (best_square[0], best_square[1]), value_b)

def main():
    
    # input grid
    example_grid = StringIO("\
     0 0 0 5 2 0 3 0 0 \n\
     0 0 0 0 0 7 2 0 0 \n\
     6 0 8 0 3 0 0 0 1 \n\
     0 7 0 0 5 0 0 0 0 \n\
     1 0 0 0 0 0 0 0 7 \n\
     0 0 0 6 0 3 0 2 0 \n\
     5 0 0 0 0 0 0 0 0 \n\
     0 4 0 0 0 0 6 0 0 \n\
     0 0 3 0 9 5 0 4 0 \
    ")
        
    example_grid = np.loadtxt(example_grid).astype(int)    
    
    grid = np.copy(example_grid)
    
    print('input grid:\n', grid, '\n')
    
    global glob_sol                         # sudoku grid
    global glob_fin 

    glob_sol = np.full((9,9), -1, dtype=int)
    glob_fin = 0
    
    if not legal_board(grid):                          # If illegal board, stop
        print('illegal initial board')
    else:
        pos_vals = initial_pos_values(grid)
        num_possible_values_a = np.sum(pos_vals, axis=2)
        grid_new, pos_vals, num_pos_vals = single_options(grid, pos_vals)   # Solve all 1 option squares repeatedly
        
        if np.count_nonzero(grid_new) == 81:
            glob_sol = grid_new
            print('solved after initial analysis')
        elif len(np.argwhere(num_pos_vals == 0)) == 0:
            print('illegal board after initial analysis')
        else:                                                   # Otherwise, full tree analysis needed
            depth_first_search(grid_new, pos_vals, (0, 0), 0, 0)

if __name__ == "__main__":
    main()