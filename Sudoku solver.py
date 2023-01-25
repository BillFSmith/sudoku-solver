import os
import numpy as np
import time
from io import StringIO


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


example_solution = StringIO("\
 9 1 7 5 2 6 3 8 4 \n\
 3 5 4 8 1 7 2 9 6 \n\
 6 2 8 4 3 9 5 7 1 \n\
 2 7 6 9 5 1 4 3 8 \n\
 1 3 5 2 4 8 9 6 7 \n\
 4 8 9 6 7 3 1 2 5 \n\
 5 9 2 7 6 4 8 1 3 \n\
 7 4 1 3 8 2 6 5 9 \n\
 8 6 3 1 9 5 7 4 2 \
")     
    
example_grid = np.loadtxt(example_grid).astype(int)
example_solution = np.loadtxt(example_solution).astype(int)

def legal_board(grid):
    board_legality = True

    for i in range(9):
        row_vals, row_count = np.unique(grid[i,:], return_counts=True)
        col_vals, col_count = np.unique(grid[:,i], return_counts=True)
        block_vals, block_count = np.unique(grid[3 * (i//3) : 3 * (i//3) + 3, 3 * (i%3) : 3 * (i%3) + 3], return_counts=True)
       
        row_count = row_count[row_vals != 0]
        row_vals = row_vals[row_vals != 0]
        col_count = col_count[col_vals != 0]
        col_vals = col_vals[col_vals != 0]
        block_count = block_count[block_vals != 0]
        block_vals = block_vals[block_vals != 0]
        
        if len(row_count) > 0 and np.max(row_count) > 1:
            board_legality = False
            # print('Duplicated', row_vals[np.argmax(row_count)], 'on row', i)
            break
        elif len(col_count) > 0 and np.max(col_count) > 1:
            board_legality = False
            # print(col_vals, col_count)
            # print('Duplicated', col_vals[np.argmax(col_count)], 'on column', i)
            break
        elif len(block_count) > 0 and np.max(block_count) > 1:
            board_legality = False
            # print('Duplicated', block_vals[np.argmax(block_count)], 'on block', i//3, i%3)
            break

    return board_legality


def initial_pos_values(grid):
    
    blank_spaces = np.argwhere(grid == 0)
    vals = np.zeros((9,9,10))
    
    for space in blank_spaces:

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
    # grid_new_b = np.copy(grid_new_b)
    
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

    
def depth_first_search(temp_grid, temp_possible_values, temp_coordinates, temp_value, depth=1):

    global glob_sol
    global glob_fin
    
    if glob_fin:
        return
   
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
            
     
    # if len(np.argwhere(temp_grid == 0)) == 0:
    if np.count_nonzero(temp_grid) == 81:    
        if np.all(np.sum(temp_grid, axis = 0) == 45) and np.all(np.sum(temp_grid, axis = 1) == 45):
            if legal_board(temp_grid):
                glob_sol = temp_grid
                glob_fin = 1
                print('solved!')
                print(grid)
                print(provided_solution)
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
                    # np.sum(temp_num_possible_values[3 * (square[0]//3) : 3 * (square[0]//3) + 3, 3 * (square[1]//3) : 3 * (square[1]//3) + 3,] == best_value)
    
            if highest_score < score:
                highest_score = score
                best_square = (square)
    else:
        best_square = (best_squares[0])

    values_b = list(np.where(temp_possible_values[best_square[0], best_square[1]] != 0)[0])
    for value_b in values_b:
        depth_first_search(np.copy(temp_grid), np.copy(temp_possible_values), (best_square[0], best_square[1]), value_b)


difficulty = 'very_easy'
difficulty = 'easy'
difficulty = 'medium'
difficulty = 'hard'

sudoku = np.load("data/" + difficulty + "_puzzle.npy")
# print(difficulty + "_puzzle.npy has been loaded into the variable sudoku")

solutions = np.load("data/" + difficulty + "_solution.npy")
# print()

correct = []
times = []

# for i in range(0,15):
    
start_time = time.process_time()
# print()
# print(i)
# puzzle_num = i

# grid = sudoku[puzzle_num,:,:]
# provided_solution = solutions[puzzle_num,:,:]

grid = np.copy(example_grid)
provided_solution = np.copy(example_solution)

glob_sol = np.full((9,9), -1, dtype=int)
glob_fin = 0

if not legal_board(grid):                                      # If illegal board, stop
    None
    # print('illegal initial board')
else:
    pos_vals = initial_pos_values(grid)
    num_possible_values_a = np.sum(pos_vals, axis=2)
    grid_new, pos_vals, num_pos_vals = single_options(grid, pos_vals)   # Solve all 1 option squares repeatedly
    
    if np.count_nonzero(grid_new) == 81:
        glob_sol = grid_new
        # print('solved after initial analysis')
    elif len(np.argwhere(num_pos_vals == 0)) == 0:
        None
        # print('illegal board after initial analysis')
    else:                                                   # Otherwise, full tree analysis needed
        depth_first_search(grid_new, pos_vals, (0, 0), 0, 0)
