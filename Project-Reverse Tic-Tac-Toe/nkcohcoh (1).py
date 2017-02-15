# nkcohcoh.py
# Rahul Sampat, September 2016

# I have used the alpha-beta pruning for finding the next best move
# Successors: Any new state created my placing my piece on the blank space will be a part of the successor set.

# I have done the following to get the maximum efficiency from the algorithm
# Although while creating successors, I evaluate and increases the priority of the successors if there are any adjacent
# pieces (upto the depth of K),hence pushing them in the end of the Priority Queue, which pops the low priority items first
# This is for the first MAX node

# From next level onwards, I have created a p_flag and I set it to 1 for MAX level and -1 for MIN level and then
# multiply this flag with their successor's evaluation value (discussed above).
# This way MAX always gets the best successors first, while MIN gets the worst (best for him) first

# For evaluation of the leaf or any given board, I have used the following formula:
# (MIN's chances of losing/MAX's chances of losing)*10000
# This gives less value when MAX's chances of losing are more and vice versa

# I am deciding the horizon as follows:
# Horizon = time_limit * 10 / total_blanks


from collections import defaultdict
from Queue import PriorityQueue
import sys
import time

# Global Variables
time_limit = 5
N = 8  # 8
K = 3  # 4
H = 2  # Need to implement a method to formulate the horizon based on the time_limit given H = a*t/b*n (a,b = constants)
# initial_board_string_global = ".w......b"
first_iteration_flag = True
# initial_board_string_global = "ww....b.b"
# initial_board_string_global = "w.ww.ww.bb.bbb.."
# initial_board_string_global = "w.b.wb..."
# initial_board_string_global = "....w......b...."
# initial_board_string_global = "w.b.w.bb.bbb..w.ww.ww.bb.bbb..w.ww.ww.bb.bbb..w.ww.ww.bb.bbb.."
# initial_board_string_global = "wbwww.bbbbwwwwbb"
# initial_board_string_global = "wbwww.bbbbwwwwbb"
# initial_board_string_global = "................................................................"
initial_board_string_global = ""
initial_board_dict_global = defaultdict(int)
our_piece_color = 'w'
opp_piece_color = 'b'
empty_piece = "."
eval_counter = 0
# Priority flag...tries to give more winnable states first when the player is MAX and more losable states first
# when the player is MIN
p_flag = 1
# Use this move if we have already lost the game
default_move = (0, 0)
# Flag used to check if we need to check the update_priority function for the successors
check_update_priority_flag = True


# Function used to set the value of H based on 'total blank spaces' and the 'time_limit' provided.
def set_horizon():
    global H
    total_blanks = initial_board_dict_global.values().count('.')
    local_horizon = time_limit * 10 / total_blanks
    print "Local Horizon", local_horizon, "total blanks", total_blanks
    H = max(H, local_horizon)


# Function to convert input string to a dictionary
def create_init_dict(initial_board_string):
    global initial_board_dict_global
    for x in range(0, N):
        for y in range(0, N):
            initial_board_dict_global[x, y] = initial_board_string[(N * x) + y]


# Function used to print the final board dictionary into a string
def convert_dict_to_string(dict):
    return ''.join('{0}'.format(val) for key, val in sorted(dict.items()))


# Function used to determine if we are white or black and sets the same
def set_our_piece_color():
    global our_piece_color, opp_piece_color
    if initial_board_dict_global.values().count('w') <= initial_board_dict_global.values().count('b'):
        our_piece_color = 'w'
        opp_piece_color = 'b'
    else:
        our_piece_color = 'b'
        opp_piece_color = 'w'


# Increases the priority of the successors if there are any adjacent pieces (upto the depth of K),
# hence pushing them in the end of the Priority Queue, which pops the low priority items first
# (Added if loop like in eval_new_3 # Added the new summation formula for the evaluation function)
def update_priority_new3(row, col):
    current_priority = defaultdict(int)
    flag_dict = defaultdict(int)
    p = 100
    no_of_our_pieces_existing = defaultdict(int)
    for i in range(K - 1, 0, -1):  # Declare global k
        if initial_board_dict_global.get((row - i, col - i), 0) == our_piece_color and flag_dict.get(0, 1):
            no_of_our_pieces_existing[0] += 2
            # Returns max priority if we find K in a row of our pieces
            if no_of_our_pieces_existing[0] / 2 == K - 1:
                return sys.maxint
            else:
                prev_piece = initial_board_dict_global.get((row - i + 1, col - i + 1), 0)
                penalty = K
                if prev_piece == our_piece_color:
                    penalty = p
                current_priority[0] += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing[0] * K)
                # Reset priority if a opp_piece_color comes in middle
                if prev_piece == opp_piece_color:
                    current_priority[0] = 0
                    flag_dict[0] = 0

        if initial_board_dict_global.get((row - i, col), 0) == our_piece_color and flag_dict.get(1, 1):
            no_of_our_pieces_existing[1] += 2
            if no_of_our_pieces_existing[1] / 2 == K - 1:
                return sys.maxint
            else:
                prev_piece = initial_board_dict_global.get((row - i + 1, col), 0)
                penalty = K
                if prev_piece == our_piece_color:
                    penalty = p
                current_priority[1] += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing[1] * K)
                # Reset priority if a opp_piece_color comes in middle
                if prev_piece == opp_piece_color:
                    current_priority[1] = 0
                    flag_dict[1] = 0

        if initial_board_dict_global.get((row - i, col + i), 0) == our_piece_color and flag_dict.get(2, 1):
            no_of_our_pieces_existing[2] += 2
            if no_of_our_pieces_existing[2] / 2 == K - 1:
                return sys.maxint
            else:
                prev_piece = initial_board_dict_global.get((row - i + 1, col + i - 1), 0)
                penalty = K
                if prev_piece == our_piece_color:
                    penalty = p
                current_priority[2] += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing[2] * K)
                # Reset priority if a opp_piece_color comes in middle
                if prev_piece == opp_piece_color:
                    current_priority[2] = 0
                    flag_dict[2] = 0

        if initial_board_dict_global.get((row, col - i), 0) == our_piece_color and flag_dict.get(3, 1):
            no_of_our_pieces_existing[3] += 2
            if no_of_our_pieces_existing[3] / 2 == K - 1:
                return sys.maxint
            else:
                prev_piece = initial_board_dict_global.get((row, col - i + 1), 0)
                penalty = K
                if prev_piece == our_piece_color:
                    penalty = p
                current_priority[3] += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing[3] * K)
                # Reset priority if a opp_piece_color comes in middle
                if prev_piece == opp_piece_color:
                    current_priority[3] = 0
                    flag_dict[3] = 0

        if initial_board_dict_global.get((row, col + i), 0) == our_piece_color and flag_dict.get(4, 1):
            no_of_our_pieces_existing[4] += 2
            if no_of_our_pieces_existing[4] / 2 == K - 1:
                return sys.maxint
            else:
                prev_piece = initial_board_dict_global.get((row, col + i - 1), 0)
                penalty = K
                if prev_piece == our_piece_color:
                    penalty = p
                current_priority[4] += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing[4] * K)
                # Reset priority if a opp_piece_color comes in middle
                if prev_piece == opp_piece_color:
                    current_priority[4] = 0
                    flag_dict[4] = 0

        if initial_board_dict_global.get((row + i, col - i), 0) == our_piece_color and flag_dict.get(5, 1):
            no_of_our_pieces_existing[5] += 2
            if no_of_our_pieces_existing[5] / 2 == K - 1:
                return sys.maxint
            else:
                prev_piece = initial_board_dict_global.get((row + i - 1, col - i + 1), 0)
                penalty = K
                if prev_piece == our_piece_color:
                    penalty = p
                current_priority[5] += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing[5] * K)
                # Reset priority if a opp_piece_color comes in middle
                if prev_piece == opp_piece_color:
                    current_priority[5] = 0
                    flag_dict[5] = 0

        if initial_board_dict_global.get((row + i, col), 0) == our_piece_color and flag_dict.get(6, 1):
            no_of_our_pieces_existing[6] += 2
            if no_of_our_pieces_existing[6] / 2 == K - 1:
                return sys.maxint
            else:
                prev_piece = initial_board_dict_global.get((row + i - 1, col), 0)
                penalty = K
                if prev_piece == our_piece_color:
                    penalty = p
                current_priority[6] += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing[6] * K)
                # Reset priority if a opp_piece_color comes in middle
                if prev_piece == opp_piece_color:
                    current_priority[6] = 0
                    flag_dict[6] = 0

        if initial_board_dict_global.get((row + i, col + i), 0) == our_piece_color and flag_dict.get(7, 1):
            no_of_our_pieces_existing[7] += 2
            if no_of_our_pieces_existing[7] / 2 == K - 1:
                return sys.maxint
            else:
                prev_piece = initial_board_dict_global.get((row + i - 1, col + i - 1), 0)
                penalty = K
                if prev_piece == our_piece_color:
                    penalty = p
                current_priority[7] += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing[7] * K)
                # Reset priority if a opp_piece_color comes in middle
                if prev_piece == opp_piece_color:
                    current_priority[7] = 0
                    flag_dict[7] = 0

    return p_flag * sum(current_priority.values())


# Find the successors of a board state
# Returns a priority queue of tuples of coordinates where a piece can be placed
# Returns empty PQ if depth == 0 or not successor found
def successors(depth):
    pq = PriorityQueue()
    if depth == 0 or empty_piece not in initial_board_dict_global.values():
        return pq
    else:
        for row in range(N):
            for col in range(N):
                current_priority = 0
                if initial_board_dict_global.get((row, col), 0) == empty_piece:
                    # Updating the priority of the move
                    current_priority = update_priority_new3(row, col)
                    pq.put((current_priority, (row, col)))
                    # if depth == H:
                    #    print "Current Priority:", current_priority, "Move: ", (row, col)
        return pq


# Evaluates the state of the board---Optimized evaluation function # Added if conditions
def eval_function_new3():
    eval_number_max = 1
    eval_number_min = 1
    p = 100
    # return eval_counter
    for index, value in initial_board_dict_global.items():
        if value == our_piece_color:
            no_of_our_pieces_existing1 = defaultdict(int)
            row = index[0]
            col = index[1]
            # for i in range(N-1, 0, -1): # GIVE A SPEED TEST FOR RANGE N-1
            # for i in range(K-1, 0, -1): # Declare global k
            for i in range(K - 1, 0, -1):  # Declare global k
                if initial_board_dict_global.get((row - i, col - i), 0) == our_piece_color:
                    no_of_our_pieces_existing1[0] += 2
                    if no_of_our_pieces_existing1[0] / 2 == K - 1:
                        return -sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row - i + 1, col - i + 1),
                                                                     0) == our_piece_color else K
                        eval_number_max += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing1[0] * K)
                if initial_board_dict_global.get((row - i, col), 0) == our_piece_color:
                    no_of_our_pieces_existing1[1] += 2
                    if no_of_our_pieces_existing1[1] / 2 == K - 1:
                        return -sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row - i + 1, col), 0) == our_piece_color else K
                        eval_number_max += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing1[1] * K)
                if initial_board_dict_global.get((row - i, col + i), 0) == our_piece_color:
                    no_of_our_pieces_existing1[2] += 2
                    if no_of_our_pieces_existing1[2] / 2 == K - 1:
                        return -sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row - i + 1, col + i - 1),
                                                                     0) == our_piece_color else K
                        eval_number_max += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing1[2] * K)
                if initial_board_dict_global.get((row, col - i), 0) == our_piece_color:
                    no_of_our_pieces_existing1[3] += 2
                    if no_of_our_pieces_existing1[3] / 2 == K - 1:
                        return -sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row, col - i + 1), 0) == our_piece_color else K
                        eval_number_max += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing1[3] * K)
                if initial_board_dict_global.get((row, col + i), 0) == our_piece_color:
                    no_of_our_pieces_existing1[4] += 2
                    if no_of_our_pieces_existing1[4] / 2 == K - 1:
                        return -sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row, col + i - 1), 0) == our_piece_color else K
                        eval_number_max += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing1[4] * K)
                if initial_board_dict_global.get((row + i, col - i), 0) == our_piece_color:
                    no_of_our_pieces_existing1[5] += 2
                    if no_of_our_pieces_existing1[5] / 2 == K - 1:
                        return -sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row + i - 1, col - i + 1),
                                                                     0) == our_piece_color else K
                        eval_number_max += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing1[5] * K)
                if initial_board_dict_global.get((row + i, col), 0) == our_piece_color:
                    no_of_our_pieces_existing1[6] += 2
                    if no_of_our_pieces_existing1[6] / 2 == K - 1:
                        return -sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row + i - 1, col), 0) == our_piece_color else K
                        eval_number_max += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing1[6] * K)
                if initial_board_dict_global.get((row + i, col + i), 0) == our_piece_color:
                    no_of_our_pieces_existing1[7] += 2
                    if no_of_our_pieces_existing1[7] / 2 == K - 1:
                        return -sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row + i - 1, col + i - 1),
                                                                     0) == our_piece_color else K
                        eval_number_max += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing1[7] * K)
        elif value == opp_piece_color:
            no_of_our_pieces_existing2 = defaultdict(int)
            row = index[0]
            col = index[1]
            for i in range(K, 0, -1):  # Declare global k
                if initial_board_dict_global.get((row - i, col - i), 0) == opp_piece_color:
                    no_of_our_pieces_existing2[8] += 2
                    if no_of_our_pieces_existing2[8] / 2 == K - 1:
                        return sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row - i + 1, col - i + 1),
                                                                     0) == opp_piece_color else K
                        eval_number_min += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing2[8] * K)
                if initial_board_dict_global.get((row - i, col), 0) == opp_piece_color:
                    no_of_our_pieces_existing2[9] += 2
                    if no_of_our_pieces_existing2[9] / 2 == K - 1:
                        return sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row - i + 1, col), 0) == opp_piece_color else K
                        eval_number_min += (N - K + 1) + ((N - i) * (penalty / K)) + (no_of_our_pieces_existing2[9] * K)
                if initial_board_dict_global.get((row - i, col + i), 0) == opp_piece_color:
                    no_of_our_pieces_existing2[10] += 2
                    if no_of_our_pieces_existing2[10] / 2 == K - 1:
                        return sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row - i + 1, col + i - 1),
                                                                     0) == opp_piece_color else K
                        eval_number_min += (N - K + 1) + ((N - i) * (penalty / K)) + (
                        no_of_our_pieces_existing2[10] * K)
                if initial_board_dict_global.get((row, col - i), 0) == opp_piece_color:
                    no_of_our_pieces_existing2[11] += 2
                    if no_of_our_pieces_existing2[11] / 2 == K - 1:
                        return sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row, col - i + 1), 0) == opp_piece_color else K
                        eval_number_min += (N - K + 1) + ((N - i) * (penalty / K)) + (
                        no_of_our_pieces_existing2[11] * K)
                if initial_board_dict_global.get((row, col + i), 0) == opp_piece_color:
                    no_of_our_pieces_existing2[12] += 2
                    if no_of_our_pieces_existing2[12] / 2 == K - 1:
                        return sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row, col + i - 1), 0) == opp_piece_color else K
                        eval_number_min += (N - K + 1) + ((N - i) * (penalty / K)) + (
                        no_of_our_pieces_existing2[12] * K)
                if initial_board_dict_global.get((row + i, col - i), 0) == opp_piece_color:
                    no_of_our_pieces_existing2[13] += 2
                    if no_of_our_pieces_existing2[13] / 2 == K - 1:
                        return sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row + i - 1, col - i + 1),
                                                                     0) == opp_piece_color else K
                        eval_number_min += (N - K + 1) + ((N - i) * (penalty / K)) + (
                        no_of_our_pieces_existing2[13] * K)
                if initial_board_dict_global.get((row + i, col), 0) == opp_piece_color:
                    no_of_our_pieces_existing2[14] += 2
                    if no_of_our_pieces_existing2[14] / 2 == K - 1:
                        return sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row + i - 1, col), 0) == opp_piece_color else K
                        eval_number_min += (N - K + 1) + ((N - i) * (penalty / K)) + (
                        no_of_our_pieces_existing2[14] * K)
                if initial_board_dict_global.get((row + i, col + i), 0) == opp_piece_color:
                    no_of_our_pieces_existing2[15] += 2
                    if no_of_our_pieces_existing2[15] / 2 == K - 1:
                        return sys.maxint
                    else:
                        penalty = p if initial_board_dict_global.get((row + i - 1, col + i - 1),
                                                                     0) == opp_piece_color else K
                        eval_number_min += (N - K + 1) + ((N - i) * (penalty / K)) + (
                        no_of_our_pieces_existing2[15] * K)
    return ((1.0 * eval_number_min) / (1.0 * eval_number_max)) * 10000


# Min Max algorithm with alpha beta pruning
# Returns a score, and the move coordinates
def minimax(depth, player, alpha, beta):
    global initial_board_dict_global, eval_counter, p_flag, first_iteration_flag, default_move
    score = 0
    best_move = (-1, -1)

    # To give good moves first for MAX and bad moves first for MIN
    p_flag = 1 if player == "MAX" else -1

    # next state is a list of tuples where the move can be made
    next_states = successors(depth)

    if next_states.qsize() == 0 or depth == 0:
        score = eval_function_new3()
        return score, best_move
    else:
        while next_states.qsize() > 0:
            next_state = next_states.get()[1]
            initial_board_dict_global[next_state] = our_piece_color
            # Print the first state no matter what to avoid time limit issues
            if first_iteration_flag:
                # print "Inside first iteration move: ", next_state
                print "New Board: "
                print convert_dict_to_string(initial_board_dict_global)
                # print "Time: ", time.time()-start_time
                default_move = next_state
                first_iteration_flag = False

            if player == "MAX":
                score = minimax(depth - 1, "MIN", alpha, beta)[0]
                if score > alpha:
                    alpha = score
                    best_move = next_state
            else:
                score = minimax(depth - 1, "MAX", alpha, beta)[0]
                if score < beta:
                    beta = score
                    best_move = next_state
            # Undo Move
            initial_board_dict_global[next_state] = empty_piece
            if depth == H and best_move != (-1, -1):
                initial_board_dict_global[best_move] = our_piece_color
                # print "New Move ", best_move
                # print "Time:          ", time.time() - start_time
                print "New Board: "
                print convert_dict_to_string(initial_board_dict_global)
                # print initial_board_dict_global
                initial_board_dict_global[best_move] = empty_piece

            # Cut-off
            if alpha > beta:
                eval_counter += 1
                break
        return [alpha, best_move] if player == "MAX" else [beta, best_move]


# Function used to return the board with the next best move from a given state
def next_move_board():  # Can avoid passing the parameter since it is global
    global initial_board_dict_global
    set_horizon()
    set_our_piece_color()

    move = minimax(H, "MAX", -sys.maxint, sys.maxint)[1]
    # if we had already lost when we got the board, then return the default move
    if move == (-1, -1):
        move = default_move
    print "Final Move", move
    initial_board_dict_global[move] = our_piece_color


# MAIN:
# Setting initial global variables with passed arguments
N = int(sys.argv[1])
K = int(sys.argv[2])
initial_board_string_global = sys.argv[3]
time_limit = int(sys.argv[4])

# file_name = "E:\Rahul\IUB Education\Fall 2016\AI (David Crandell)\Assignments\Assignment-2\Test Files\input3.txt"
# main(file_name)
print "Initial State Global:"
print initial_board_dict_global

start_time = time.time()
create_init_dict(initial_board_string_global)
print "Initial Dict"
print initial_board_dict_global
next_move_board()
end_time = time.time()
print "Eval Counter", eval_counter
print "Total Time:", end_time - start_time
# Convert the board back to String
print "Original Board:"
print initial_board_string_global
print "New Board:"
print convert_dict_to_string(initial_board_dict_global)
