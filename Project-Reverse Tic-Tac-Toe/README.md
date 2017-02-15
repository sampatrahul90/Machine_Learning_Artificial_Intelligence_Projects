PLAYING X&0 - TIC TAC TOE against gravity!

Problem:

N - K - Coh Coh - It is the opposite of Tic Tac Toe, where you avoid building series of same shapes, bang opposite of tic tac toe
Automate the game to provide the next best move - using Minimax Algorithm with Alpha-Beta pruning!

The problem was formulated in the following way:

State Space: All possible moves that involve placing a marble in an empty space on the board

State Space: All possible moves that involve placing a marble in an empty space on the board with one marble occupying exactly one place.

Initial State: Any legit move that satisfies that each marble occupy one space and should not be a terminal state.

Successor function: All possible successors from a given initial state where each successor would have an empty tile 
filled with a marble of a particular color (depending on the player).

Minimax prunes successors based on their evaluation score so some successors would not be generated/explored.

# Formulation:

I have done the following to get the maximum efficiency from the algorithm
Although while creating successors, I evaluate and increases the priority of the successors if there are any adjacent
pieces (upto the depth of K),hence pushing them in the end of the Priority Queue, which pops the low priority items first
This is for the first MAX node

From next level onwards, I have created a p_flag and I set it to 1 for MAX level and -1 for MIN level and then
multiply this flag with their successor's evaluation value (discussed above).
This way MAX always gets the best successors first, while MIN gets the worst (best for him) first

For evaluation of the leaf or any given board, I have used the following formula:
(MIN's chances of losing/MAX's chances of losing)*10000
This gives less value when MAX's chances of losing are more and vice versa

I am deciding the horizon as follows:
Horizon = time_limit * 10 / total_blanks
