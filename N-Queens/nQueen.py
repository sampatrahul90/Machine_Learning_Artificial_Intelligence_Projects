# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, August 2016
#
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

# This is N, the size of the board.
#N=10

import time
# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] )

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] )

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ " ".join([ "Q" if col else "_" for col in row ]) for row in board])

#Gives the 2 diagonals counts for a particular position on the board
def diagonal_count(board,r,c):
    d1=0
    d2=0
    row_no=r
    col_no=c
    while row_no < N and col_no < N:
        d1 = d1 + board[row_no][col_no]
        row_no+=1
        col_no+=1
    row_no=r-1
    col_no=c-1
    while row_no>=0 and col_no>=0:
        d1 = d1 + board[row_no][col_no]
        row_no-=1
        col_no-=1
    row_no=r
    col_no=c
    while row_no>=0 and col_no<N:
        d2 = d2 + board[row_no][col_no]
        row_no-=1
        col_no+=1
    row_no=r+1
    col_no=c-1
    while row_no<N and col_no>=0:
        d2 = d2 + board[row_no][col_no]
        row_no+=1
        col_no-=1
    return d1,d2

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    #if count_pieces(board) != N:
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state WITH duplicate states
def successors(board):
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]

# Get list of successors of given board state WITHOUT duplicate (no rook added) and N+1 rook states******** Question 3
def successors2(board):
    new_board=[ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]
    #Check added remove duplicate states and states with N+1 rooks
    x=len(new_board)-1
    while (-1<x<len(new_board)):
        if board == new_board[x] or count_pieces(new_board[x])>N:
            new_board.pop(x)
        x=x-1
    return new_board

# # Get list of successors of given board state for Rooks in a very OPTIMIZED WAY*********Question 4
# def successors3(board):
#     #new_board3=[ add_piece(board, r, r) for r in range(0, N)]
#     new_board3=[board]
#     for r in range(0,N):
#         new_board3=[add_piece(new_board3[0],r,r)]
#     return new_board3

# Does the same function as the above successors3(), but in a LESS optimized way. (Kept the function for its approach and learning purposes)
# Get list of successors of given board state for Rooks in a very OPTIMIZED WAY*********Question 4
def successors3(board):
    new_board3=[ add_piece(board, r, r) for r in range(0, N)]
    x=len(new_board3)-1
    #print x
    while (-1<x<len(new_board3)):
        if board == new_board3[x] or count_pieces(new_board3[x])>N:
            new_board3.pop(x)
        x=x-1
    return new_board3


# Get list of successors of given board state for Queen*********Question 5
def successors4(board):
    new_board=[ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]
    x=len(new_board)-1
    while (-1<x<len(new_board)):
        if board == new_board[x] or count_pieces(new_board[x])>N:
            new_board.pop(x)
        x=x-1
    return new_board


# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] )

# #I have divided the board into 4 diagonal quadrants or 4 different halves- 2 parts across each diagonal.
# #Now I am using the below 4 functions to check the diagonal sum for each part respectively.
# # Checks the 1st half  of the board starting from left bottom half of the board*********
# def count_on_diag_left_bottom(board,r):
#     return sum( [ board[x][abs(x-r)] for x in range(r,N) ] )
#
# # Checks the 2nd half  of the board starting from left top half of the board*********
# def count_on_diag_left_top(board,r):
#     return sum( [ board[x][abs(x-r)] for x in range(r,-1,-1) ] )
#
# # Checks the 3rd half  of the board starting from right top half of the board*********
# def count_on_diag_right_top(board,c):
#     return sum( [ board[abs(x-c)][abs(x)] for x in range(c,N) ] )
#
# # Checks the 4th half of the board starting from right bottom half of the board*********
# def count_on_diag_right_bottom(board,r):
#     col_index=N-1
#     total_sum=0
#     #return sum( [ board[abs(x)][abs(col_index-1)] for x in range(r,N) ] )
#     for x in range(r,N):
#         total_sum = total_sum + board[abs(x)][abs(col_index)]
#         col_index-=1
#     return total_sum
#
# #Checks if the board is a goal for n-Queens******************s
# def is_queen_valid(board):
#     return all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
#         all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] ) and \
#         all( [ count_on_diag_left_bottom(board, dr1) <=1 for dr1 in range(0, N-1) ] ) and \
#         all( [ count_on_diag_left_top(board, dr2) <=1 for dr2 in range(N-1, 0,-1) ] ) and \
#         all( [ count_on_diag_right_top(board, dr3) <=1 for dr3 in range(1, N-1) ] ) and \
#         all( [ count_on_diag_right_bottom(board, dr4) <=1 for dr4 in range(1, N) ] )


# Solve n-rooks with the default successors() function:
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors3( fringe.pop() ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# New Solve function which calls the new successors2() for better performance in n-rooks************Question 3 problems solved!**********
def solve_unique(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors2( fringe.pop() ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# New Solve function which calls the new successors3() for a new approach towards n-rooks problem************Question 4 problems solved!**********
def solve_rook(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors3( fringe.pop() ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# New Solve function for a solution towards the n-Queen problem************Question 5 problems solved!**********
def nqueens_solve(initial_board):
    new_fringe = initial_board
    row_no=0
    col_no=0
    col_list=[]
    new_fringe[row_no][col_no]=1
    row_no+=1
    queen_count=1
    col_list.append(col_no)
    pointer=0
    while queen_count<N:
        col_count=count_on_col(new_fringe,col_no)
        d1,d2=diagonal_count(new_fringe,row_no,col_no)
        if col_count!=1 and d1!=1 and d2!=1:
            new_fringe[row_no][col_no]=1
            col_list.append(col_no)
            row_no+=1
            col_no=0
            queen_count+=1
        elif col_no<N-1 :
            col_no+=1
        else:
            col_no=col_list.pop()
            row_no-=1
            new_fringe[row_no][col_no]=0
            queen_count-=1
            if col_no==N-1:
                while col_no==N-1:
                    row_no-=1
                    col_no=col_list.pop()
                    new_fringe[row_no][col_no]=0
                    queen_count-=1
            col_no+=1
    print new_fringe
    return new_fringe

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
#initial_board = [[0]*N]*N
print "Please enter the value of N:" + "\n"
# This is N, the size of the board.
N = int(raw_input())
print "Please enter 1 for Rooks and 2 for Queen:" + "\n"
b = int(raw_input())

initial_board = [[0 for x in range(N)] for y in range(N)]
#print initial_board
print "Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n"
start_time=time.time()
if b==2:
    solution = nqueens_solve(initial_board)
else:
    solution = solve(initial_board)
end_time=time.time()
print end_time-start_time
print printable_board(solution) if solution else "Sorry, no solution found. :("


