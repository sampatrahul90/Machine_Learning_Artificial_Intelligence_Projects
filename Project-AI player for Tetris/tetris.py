"""

How does it work ?

The computer player tries to generate all possible board states by placing the current piece in all possible columns
and rotations, and then placing the next piece in all possible columns/rotations. Then it evaluates which
is the best possible move based on certain features and its weight(Explained below).
It picks the one with the highest evaluation value. The computer
player then plays this move on the actual board with the chosen column and rotation of the current piece.
This process is repeated for each piece.


Explanation for Features and Weights:
 The Features is a metric which actually decides which is an ideal board that we want to see. We referred to certain
 articles and papers who have done analysis on the different features and we decided to pick 6 features \
 out of them for our implementation here. We have also added weights to each of these features based on the explanation
  in the paper as well as several tests that we ran.

  Feature 1 - Aggregate Height: Sum of heights of each column is a negative feature.
  Feature 2 - Complete Lines: Number of lines complete lines is a positive feature.
  Feature 3 - Bumpiness: sum of difference in height between each columns is a negative feature.
  Feature 4 - Holes count: Number of spaces with atleast a filled spot above it in the same column is a negative feature.
  Feature 5 - Altitude Delta: Different is height between the tallest and shortest column is a negative feature.
  Feature 6 - Weighted Holes: Sum of the holes weighed by the row they are located in is a negative feature.

  General Formula for Evalution:
  Feature1 * Weight1 + Feature2 * Weight2 +.......+ FeatureN * WeightN

    Here, the Weight(N) can be negative or positive based on whether it is a negative or positive feature.

How to Run ?

    python ./tetris.py computer animated

References:
    1. http://www.cs.uml.edu/ecg/uploads/AIfall10/eshahar_rwest_GATetris.pdf
    2. https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

"""


# Simple tetris program! v0.2
# D. Crandall, Sept 2016

from AnimatedTetris import *
from SimpleTetris import *
from kbinput import *
from TetrisGame import *

class HumanPlayer:
    def get_moves(self, tetris):
        print "Type a sequence of moves using: \n  b for move left \n  m for move right \n  n for rotation\nThen press enter. E.g.: bbbnn\n"
        moves = raw_input()
        return moves

    def control_game(self, tetris):
        while 1:
            c = get_char_keyboard()
            commands =  { "b": tetris.left, "n": tetris.rotate, "m": tetris.right, " ": tetris.down }
            commands[c]()

#####
# This is the part you'll want to modify!
# Replace our super simple algorithm with something better
#
class ComputerPlayer:
    # This function should generate a series of commands to move the piece into the "optimal"
    # position. The commands are a string of letters, where b and m represent left and right, respectively,
    # and n rotates. tetris is an object that lets you inspect the board, e.g.:
    #   - tetris.col, tetris.row have the current column and row of the upper-left corner of the 
    #     falling piece
    #   - tetris.get_piece() is the current piece, tetris.get_next_piece() is the next piece after that
    #   - tetris.left(), tetris.right(), tetris.down(), and tetris.rotate() can be called to actually
    #     issue game commands
    #   - tetris.get_board() returns the current state of the board, as a list of strings.
    #
    def get_moves(self, tetris):
        # super simple current algorithm: just randomly move left, right, and rotate a few times
        return random.choice("mnb") * random.randint(1, 10)
       
    # This is the version that's used by the animted version. This is really similar to get_moves,
    # except that it runs as a separate thread and you should access various methods and data in
    # the "tetris" object to control the movement. In particular:
    #   - tetris.col, tetris.row have the current column and row of the upper-left corner of the 
    #     falling piece
    #   - tetris.get_piece() is the current piece, tetris.get_next_piece() is the next piece after that
    #   - tetris.left(), tetris.right(), tetris.down(), and tetris.rotate() can be called to actually
    #     issue game commands
    #   - tetris.get_board() returns the current state of the board, as a list of strings.
    #
    def control_game(self, tetris):
        # another super simple algorithm: just move piece to the least-full column
        while 1:
            board = tetris.get_board()

            (col1, col2, rot1) = self.simulate(board, tetris.get_piece(), tetris.get_next_piece())

            while rot1 > 0:
                tetris.rotate()
                rot1 = rot1 - 1

            while col1 != tetris.col:
                if col1 > tetris.col:
                    tetris.right()
                else:
                    tetris.left()
            tetris.down()

    def place_piece(self, board, piece, col):
        row = 0
        state = (board, 0)
        while not TetrisGame.check_collision(state, piece, row + 1, col):
            row += 1
        return TetrisGame.place_piece(state, piece, row, col)

    def width_of_piece(self, piece):
        return len(piece[0])

    def simulate(self, board, piece, next_piece):
        final_tuple = (-99999999, 0, 0, 0)

        for piece1_rot_number in range(0, 4):
            piece1 = TetrisGame.rotate_piece(piece[0], piece1_rot_number * 90)
            for col1 in range(0, 11-self.width_of_piece(piece1)):
                new_board = self.place_piece(board, piece1, col1)
                for piece2_rot_number in range(0, 4):
                    piece2 = TetrisGame.rotate_piece(next_piece, piece2_rot_number * 90)
                    for col2 in range(0, 11-self.width_of_piece(piece2)):
                        new_final_board = self.place_piece(new_board[0], piece2, col2)
                        points = self.evaluate_board(new_final_board[0])

                        if final_tuple[0] < points:
                            final_tuple = (points, col1, col2, piece1_rot_number)

        return (final_tuple[1], final_tuple[2], final_tuple[3])

    def evaluate_board(self, board):
        comp_lines = self.get_complete_lines(board)
        (agg_height_new, holes_count_new, bumpiness_count_new, altitude_delta, weighted_holes) = self.get_features(board)

        return -5 * agg_height_new + 7.5 * comp_lines - 2.5 * holes_count_new - \
               1.5 * bumpiness_count_new - 1.5 * altitude_delta - 1.0 * weighted_holes

    def get_features(self, board):
        agg_height = 0
        holes_count = 0
        bumpiness = 0
        agg_height_done = False
        bumpiness_done = False
        last_col = -1
        weighted_holes = 0

        max_height = 0
        min_height = 0

        for col in range(0, TetrisGame.BOARD_WIDTH):
            gap_started = False
            col_added = False
            for row in range(0, TetrisGame.BOARD_HEIGHT):

                # Agg height and altitude delta
                if not agg_height_done and (board[row])[col] == 'x':
                    agg_height = agg_height + TetrisGame.BOARD_HEIGHT - row
                    agg_height_done = True
                    current_row = TetrisGame.BOARD_HEIGHT - row
                    if max_height < current_row:
                        max_height = current_row
                    elif min_height > current_row:
                        min_height = current_row

                # Holes count and weighted holes
                if not gap_started and (board[row])[col] == 'x':
                    gap_started = True
                elif gap_started and (board[row])[col] == ' ':
                    holes_count += 1
                    weighted_holes += float(row) / TetrisGame.BOARD_HEIGHT

                # Bumpiness count
                if not bumpiness_done and (board[row])[col] == 'x':
                    if last_col != -1:
                        bumpiness = bumpiness + abs(last_col - (TetrisGame.BOARD_HEIGHT - row))
                    last_col = TetrisGame.BOARD_HEIGHT - row
                    col_added = True
                    bumpiness_done = True

            agg_height_done = False
            bumpiness_done = False

            if col_added:
                continue
            if last_col != -1:
                bumpiness = bumpiness + last_col
            last_col = 0
        altitude_delta = max_height - min_height
        return (agg_height, holes_count, bumpiness, altitude_delta, weighted_holes)

    def get_complete_lines(self, board):
        complete_line_count = 0
        for row in range(0, TetrisGame.BOARD_HEIGHT):
            for col in range(0, TetrisGame.BOARD_WIDTH):
                if (board[row])[col] != 'x':
                    break
                if col == TetrisGame.BOARD_WIDTH - 1:
                    complete_line_count += 1
        return complete_line_count

###################
#### main program

(player_opt, interface_opt) = sys.argv[1:3]

try:
    if player_opt == "human":
        player = HumanPlayer()
    elif player_opt == "computer":
        player = ComputerPlayer()
    else:
        print "unknown player!"

    if interface_opt == "simple":
        tetris = SimpleTetris()
    elif interface_opt == "animated":
        tetris = AnimatedTetris()
    else:
        print "unknown interface!"

    tetris.start_game(player)

except EndOfGame as s:
    print "\n\n\n", s



