# sudoku-solver

For performance, this code heavily uses the numpy library

An array of size 9 x 9 x 10 is created. The first two coordinates refer to the coordinates on the board, and the last refers to which numbers are possible in that location. The array takes binary values, with 1 meaning the number is possible and 0 is impossible. 

For example, if pos_vals[0, 1, 3] = 1, then 3 is a possibility for this space. 
This means that if np.sum(pos_vals[0, 1, :]) = 0, then there are no options for this square and that branch of the depth first approach must be abandoned. 

However, if np.sum(pos_vals[0, 1, :]) = 1, then there is only one option for this space and this must be selected. 

This approach means that relevant rows, column, and sub grids can be set to zero when a new number has been inserted in the grid. This offers significant performance improvement over storing an array of lists. 
