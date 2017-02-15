•	Implemented a 15-tile puzzle solver with edge flip functionality ie. the tile at the edges can swap with the tiles on the other side of the edge. 
•	Given any state, it can solve the puzzle with minimum amount of steps.
•	Implemented using A* algorithm with modified Manhattan heuristic function to accommodate the edge flip functionality for the states.

Below is the working of the algorithm:
# The 15 tile puzzle problem is: From any given initial configuration, our goal is to find the shortest possible sequence
# of moves that restores the canonical configuration (all tiles in ascending order)

# I have used to A* approach to solve the above mentioned problem. Since our heuristic function (explained below)
# is admissible and consistent, we avoid re-visiting the already visited states by keeping a track of them in a list.
# 1)Abstraction:

# a) State Space: The state space of the problem contains all the successor states that can be generated from the
# initial configuration, until the goal is reached

# b) Successor Function: From any given state, the successor function generates 4 new states, by swapping a tile from
# each direction (L,R,U,D). We avoid visiting repeated states by

# c) Cost Function/Edge Weight: For each move a cost of 1 is added

# d) Heuristic Function: I have used the (modified) Manhattan Distance to calculate the Heuristic Cost of a State.
# By modified,I mean that I have incorporated the "important change" ie. the tiles being able to iterate over the edges.
# In some cases, it can reduce the usual Manhattan Distance for a tile, and by taking care of that, I have maintained
# the admissibility and the consistency of the heuristic function. Kindly refer the heuristic_cost() for more clarification.

# e) Goal State: The goal state is the state with all tiles in the ascending order.
# I am storing the coordinates of the goal state in goal_board_final to check if the goal state has been reached or not.
#
# 2) Working:
# Its a simple A* Search Algorithm with admissible and consistent heuristics (as explained above)
# It initially converts the input file into a initial_list which is then converted to a dictionary called initial_state_global
# Each successor state is then stored in the same format as this dictionary

# These dictionaries contain the following keys:
# a) grid: Stores the grid values and their coordinates
# b) moves: Stores all the moves done to reach the current state from the initial state
# c) level: The level (or the g(s) = distance of the current state from the initial state)

# The initial_list is passed to the is_solution()-->permutation_inversion() function, which determines the solvability of the initial board
# If it is even, only then the board is solvable, else the program says "Sorry, no solution found"
# If solvable, the initial_state_global dict is then passed to the solve() function.
# It contains 'fringe'(a priority queue-to keep a track of generated successors) and
# 'closed' (a list-to keep a track of visited states)

# We ignore the already visited states and do not add them to the fringe.
# Since the heuristic is admissible and consistent, we will always get the optimal path to the goal.

# 3)
# a) Challenges: Balancing between the priority queue functionality and its looping cost (to find whether a state already exists in the fringe)
# Created a list_fringe_track list which is kind of a duplicate of the fringe, to minimize the looping cost.

# b) Design Decisions: Implemented the state as a dict rather than a list of lists, so as to reduce the iterations while swapping and generating successors.
# Each state also contains the level and the moves. Advantages of which are as follows:
# Moves: Stoes moves of each state, so as to avoid any back tracking for the final set of moves upon reaching the goal state.
# Level: To calculate the g(s) of the successor state

# Based on the board size ie. N, the code can also work for some configurations of a 24 tile puzzle (N=5)


# Other Heuristics:
# Except for Manhattan, I had tried the following:
# a) No of misplaced tiles: Works worst than the Manhattan Distance
# b) Manhattan + Linear Conflict: Linear conflict adds 2 to the Manhattan distance, incase there is any linear conflicts
# (a greater no is ahead of a smaller no) in the same row. But in some cases, adding this makes our heuristic as non-admissible,
# since we can flip the slides from the border.


• Tests carried for professors input, we obtain a path of 24 steps (LLUURULULDRDRDRRULDRUUUL) within about 2 seconds (0:00:02.189130) on our local machines. 
There are various test cases we tried, some take a while and some happen quickly, it fairly depends upon the complexity of the input.
