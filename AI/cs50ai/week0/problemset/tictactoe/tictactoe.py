"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None

""""
for all functions that accept boars as input, assume that it is a valid board.
Don't mod function declarations
I should never be able to beat the AI
YOu can add additional helper functs
Alpha_beta pruning is optional but can make my ai run more efficiently
"""


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    Takes board state as input and returns whose turn it is
    In the intial game state x gets the first move
    Any return val is acceptable if game is already over
    """
    # the following is the code i wrote
    """
    x_count = 0
    o_count = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                x_count += 1
            elif board[i][j] == O:
                o_count += 1
    if x_count == o_count:
        return X
    elif x_count > o_count:
        return O
    """
    # this is the code ai wrote
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count == o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    Each action should be repped as a tuple where i = row of move(0,1 or 2) and j
    corresponds to cell (0,1 or 2)
    Possible moves are all cells that don't have an x or an o in them
    Any return value is acceptable if terminal board is given
    """
    # the following is the code  i wrote
    """
    act= set()
    if terminal(board):
        return set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                #yield (i, j)
                act.add((i,j))
    return act
    """
    # this is the code ai wrote
    if terminal(board):
        return set()
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    should return a new board state withoud modding the original
    if action  isn't valid raise an exception
    the returned state should be the board that would result from taking original input
    board and doing the indicated action on it
    thee original board must be left unmodified. Therefore dont simply update a cell in
    a board but make a deep copy of the board first before making any changes
    """
    import copy

    # the following is the code i wrote
    """    
    brd_copy = copy.deepcopy(board)
    playa = player(brd_copy)
    try:
        brd_copy[action[0]][action[1]] = playa
    except IndexError:
        raise Exception("Invalid action")
    else:
        return brd_copy
        """
    # this is the code ai wrote
    if action not in actions(board):
        raise ValueError("invalid action")
    brd_copy = copy.deepcopy(board)
    brd_copy[action[0]][action[1]] = player(brd_copy)
    return brd_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.either X or O
    One can win the game with three of their moves ina row horizontally, vertically or diagonally
    Assume there will be at most one winner
    return none if there is no winner

    """
    # check rows
    # for i in range(3):
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != EMPTY:  # is not None:
            return row[0]
    # check columns
    for col in range(3):
        if (
            board[0][col] == board[1][col] == board[2][col] and board[0][col] != EMPTY
        ):  # is not None:
            return board[0][col]
    # check diagonals
    if (
        board[0][0] == board[1][1] == board[2][2] and board[0][0] != EMPTY
    ):  # is not None:
        return board[0][0]
    if (
        board[0][2] == board[1][1] == board[2][0] and board[0][2] != EMPTY
    ):  # is not None:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    true if all cells are filled or if someone has won
    """
    if winner(board) or all(cell != EMPTY for row in board for cell in row):
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    Accepts as input a terminal board
    Utility is only called if terminal returns true
    """

    winner_ = winner(board)
    if winner_ == X:
        return 1
    elif winner_ == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    Move should be optimal action (i, j) that is valid/allowable
    If multiple, return any
    if the board is terminal return none
    """
    if terminal(board):
        return None
    current_player = player(board)
    if current_player == X:  # maximiser
        best_move = None
        best_value = -math.inf
        # call to the minimaxing algo
        for action in actions(board):
            move_value = minimaxer(result(board, action), 0, -math.inf, math.inf)
            if move_value > best_value:
                best_value = move_value
                best_move = action
        return best_move
    else:  # minimiser
        best_value = math.inf
        best_move = None
        # call to the minimaxing algo
        for action in actions(board):
            move_value = minimaxer(result(board, action), 0, -math.inf, math.inf)
            if move_value < best_value:
                best_value = move_value
                best_move = action
        return best_move


def minimaxer(board, depth, alpha, beta):
    if terminal(board):
        return utility(board)
    if player(board) == X:  # maximising player
        max_eval = -math.inf
        for action in actions(board):
            eval = minimaxer(
                result(board, action), depth + 1, alpha, beta
            )  # recursive call
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:  # prune
                break
        # print(max_eval)
        # print(depth)
        return max_eval

    else:  # minimising player
        min_eval = math.inf
        for action in actions(board):
            eval = minimaxer(
                result(board, action), depth + 1, alpha, beta
            )  # recursive call
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:  # prune
                break
        # print(min_eval)
        # print(depth)
        return min_eval
