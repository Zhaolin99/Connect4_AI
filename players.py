import random
import time
from copy import deepcopy

import numpy as np
import pygame
import math
import sys


class connect4Player(object):
    def __init__(self, position, seed=0):
        self.position = position
        self.opponent = None
        self.seed = seed
        random.seed(seed)

    def play(self, env, move):
        move = [-1]


class human(connect4Player):

    def play(self, env, move):
        move[:] = [int(input('Select next move: '))]
        while True:
            if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
                break
            move[:] = [int(input('Index invalid. Select next move: '))]


class human2(connect4Player):

    def play(self, env, move):
        done = False
        while (not done):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if self.position == 1:
                        pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                    else:
                        pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))
                    move[:] = [col]
                    done = True


class randomAI(connect4Player):

    def play(self, env, move):
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        move[:] = [random.choice(indices)]


class stupidAI(connect4Player):

    def play(self, env, move):
        possible = env.topPosition >= 0
        indices = []

        for i, p in enumerate(possible):
            if p: indices.append(i)
        if 3 in indices:
            move[:] = [3]
        elif 2 in indices:
            move[:] = [2]
        elif 1 in indices:
            move[:] = [1]
        elif 5 in indices:
            move[:] = [5]
        elif 6 in indices:
            move[:] = [6]
        else:
            move[:] = [0]


class minimaxAI(connect4Player):
    def weight_board(self, env):
        print("env shape", env.shape)
        a = np.zeros(env.shape).astype('int32')
        for i in range(env.shape[0]):
            a[i] = [2, 5, 6, 7, 6, 5, 2]
        b = np.zeros(env.shape).astype('int32')
        c = np.array([[1], [2], [3], [3], [2], [1]])
        b[:, 0:7] = c

        weighted_board = a * b

        # weight_on_cross = np.array([[4, 3, 3, 2, 3, 3, 4]])

        weighted_board[5] = [20, 20, 20, 30, 20, 20, 20]

        """
        for i in range(3):
            weighted_board[i] = weighted_board[i] + weight_on_cross - i
        weighted_board[5] = weighted_board[5] * 2.5
        weighted_board[4] = weighted_board[4] * 2
        weighted_board[3] = weighted_board[3] * 2
        weighted_board[2] = weighted_board[2] * 2
        """
        print("scores")
        print(weighted_board)
        return weighted_board

    def evaluation(self, env, move, weighted_matrix):
        # If one more step needed; set it to infinity
        # Either win the game, or block
        if self.position == 1:
            opp_player = 2
        else:
            opp_player = 1

        if env.gameOver(move, self.position):
            print("one step to finish game, move", move)
            print("eval finish game board", env.board)
            return 100000

        if env.gameOver(move, opp_player):
            return -100000

        print("In evaluation move is ", move, " topPosition", env.topPosition, "  board\n", env.board)

        scores = 0

        for x in range(0, 6):
            for y in range(0, 7):
                if env.board[x][y] == self.position:
                    scores += weighted_matrix[x][y]
                elif env.board[x][y] == opp_player:
                    scores -= weighted_matrix[x][y]

        print("temp scores", scores)

        # Horizontol Check
        for row in range(env.shape[0]):
            row_array = []
            for i in list(env.board[row, :]):
                row_array.append(i)
            for col in range(env.shape[1] - 3):
                window = row_array[col: col + 4]
                if window.count(self.position) == 4:
                    scores += 500 * 1.2
                elif window.count(self.position) == 3 and window.count(0) == 1:
                    scores += 200 * 1.2
                elif window.count(opp_player) == 3 and window.count(0) == 1:
                    scores -= 150 * 1.2
                elif window.count(self.position) == 2 and window.count(0) == 2:
                    scores += 50 * 1.2
                elif window.count(opp_player) == 2 and window.count(0) == 2:
                    scores -= 50

        # Vertical Check
        for col in range(env.shape[1]):
            column = []
            for i in list(env.board[:, col]):
                column.append(i)
            for row in range(env.shape[0] - 3):
                window = column[row: row + 4]
                if window.count(self.position) == 4:
                    scores += 500 * 1.1
                elif window.count(self.position) == 3 and window.count(0) == 1:
                    scores += 200 * 1.1
                elif window.count(opp_player) == 3 and window.count(0) == 1:
                    scores -= 150 * 1.1
                elif window.count(self.position) == 2 and window.count(0) == 2:
                    scores += 50 * 1.1

        # Left-up Check
        for row in range(env.shape[0] - 3):
            for col in range(env.shape[1] - 3):
                window = []
                for i in range(4):
                    window.append([env.board[row + 3 - i][col + i]])
                if window.count(self.position) == 4:
                    scores += 500 * 1.1
                elif window.count(self.position) == 3 and window.count(0) == 1:
                    scores += 200 * 1.1
                elif window.count(opp_player) == 3 and window.count(0) == 1:
                    scores -= 150 * 1.1
                elif window.count(self.position) == 2 and window.count(0) == 2:
                    scores += 50 * 1.1

        # Right-up Check
        for row in range(env.shape[0] - 3):
            for col in range(env.shape[1] - 3):
                window = []
                for i in range(4):
                    window.append([env.board[row + i][col + i]])
                window = [j for sub in window for j in sub]
                if window.count(self.position) == 4:
                    scores += 500 * 1.1
                elif window.count(self.position) == 3 and window.count(0) == 1:
                    scores += 200 * 1.1
                elif window.count(opp_player) == 3 and window.count(0) == 1:
                    scores -= 150 * 1.1
                elif window.count(self.position) == 2 and window.count(0) == 2:
                    scores += 50 * 1.1

        # eva_value = weighted_matrix[env.topPosition[move] + 1][move]
        print("In evaluation eva_value ", scores)
        return scores

    def Max(self, move, env, depth, weighted_matrix):
        print("In MAX, mov is", move, "current pos: ", self.position, " Depth", depth, "TopPosition", env.topPosition)
        try:
            if env.gameOver(move, self.position) or (depth == 0):
                print("Game ends in Max, move is", move)
                eva_value = self.evaluation(env, move, weighted_matrix)
                print("Evaluation value in Max is", eva_value)
                return eva_value
        except:
            print("Error from Max gameOver move ", move, " env.topPosition ", env.topPosition, "\nenv.board\n",
                  env.board)
            raise

        value = -1000000

        new_env = self.simulateMove(deepcopy(env), move, self.position)
        # Update possible after one simulate move
        possible = new_env.topPosition >= 0
        possible_col = []

        for i, p in enumerate(possible):
            if p: possible_col.append(i)

        for child in possible_col:
            value = max(value, self.Min(child, new_env, depth - 1, weighted_matrix))
        return value

    def Min(self, move, env, depth, weighted_matrix):
        print("In Min, mov is", move, "current pos: ", self.position, " Depth", depth, "TopPosition", env.topPosition)
        try:
            if env.gameOver(move, self.position) or (depth == 0):
                print("Game ends in Min, move is", move)
                eva_value = self.evaluation(env, move, weighted_matrix)
                print("Evaluation value in Min is", eva_value)
                return eva_value
        except:
            print("Error from Min gameOver move ", move, " env.topPosition ", env.topPosition, "env.board ", env.board)
            raise

        value = 100000

        new_env = self.simulateMove(deepcopy(env), move, self.position)
        # Update possible after one simulate move
        possible = new_env.topPosition >= 0
        possible_col = []

        for i, p in enumerate(possible):
            if p: possible_col.append(i)

        for child in possible_col:
            value = min(value, self.Max(child, new_env, depth - 1, weighted_matrix))
        return value

    def simulateMove(self, env, move, player):
        # print("In simulate move, player with move", player, move)
        # print("top position", env.topPosition)
        # print("board before move", env.board)
        # print("[env.topPosition[move]][move]",env.topPosition[move], move)
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1

        return env

    def MinMax(self, env, move, depth, is_Max, weighted_matrix, self_player, opposite_player):
        print("Depth:", depth, " is Max-turn:", is_Max, " move:", move, "self_player: ", self_player, "opp_player: ",
              opposite_player)
        if is_Max:
            cur_player = self_player
            new_env = self.simulateMove(deepcopy(env), move, self_player)

        else:
            cur_player = opposite_player
            new_env = self.simulateMove(deepcopy(env), move, opposite_player)

        try:
            # new_env = self.simulateMove(deepcopy(env), move, self_player)

            if (depth == 0) or new_env.gameOver(move, cur_player):
                # print("Game ends in MinMax, move is",move)
                eva_value = self.evaluation(new_env, move, weighted_matrix)
                # print("Evaluation value in MinMax is", eva_value)
                return eva_value
        except:
            if is_Max:
                print("Max turn error")
            else:
                print("Min turn error")
            print("Error from Min_Max gameOver move ", move, " env.topPosition ", new_env.topPosition, "env.board \n",
                  new_env.board)
            raise

        if is_Max:
            value = -10000000
            # new_env = self.simulateMove(deepcopy(env), move, self_player)

            # update possible
            possible = env.topPosition >= 0
            indices = []
            for i, p in enumerate(possible):
                if p: indices.append(i)

            for child in indices:
                is_Max = False
                value = max(value, self.MinMax(new_env, child, depth - 1, is_Max, weighted_matrix, self_player,
                                               opposite_player))
            return value

        else:
            value = 10000000
            # new_env = self.simulateMove(deepcopy(env), move, opposite_player)

            # update possible
            possible = env.topPosition >= 0
            indices = []
            for i, p in enumerate(possible):
                if p: indices.append(i)

            for child in indices:
                is_Max = True
                value = min(value, self.MinMax(new_env, child, depth-1, is_Max, weighted_matrix, self_player, opposite_player))
            return value

    def last_step_block(self, env, move, self_player, opposite_player):
        print("llast step check board\n", env.board)
        new_env = self.simulateMove(deepcopy(env), move, opposite_player)

        sec_env = self.simulateMove(deepcopy(env), move, self_player)

        if sec_env.gameOver(move, self_player):  #Last step to win
            return True
        elif new_env.gameOver(move, opposite_player): #Block last step not lose
            print("last step to lose!! if opposite goes", move)
            return True
        else:
            return False
    """
        possible = sec_env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)

        for child in indices:
            new_next_env = self.simulateMove(deepcopy(sec_env), child, opposite_player)
            if new_next_env.gameOver(child, opposite_player):
                print("last step to lose!! if opposite goes", move)
                return True
    """
    def stop_move(self, env, move, self_player, opposite_player):
        print("stop_move\n", env.board)
        new_env = self.simulateMove(deepcopy(env), move, self_player)

        possible = new_env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)


        for child in indices:
            sec_env = self.simulateMove(deepcopy(new_env), child, opposite_player)
            if sec_env.gameOver(child, opposite_player):
                return True


    def play(self, env, move):
        env = deepcopy(env)
        env.visualize = False
        possible = env.topPosition >= 0
        # Set up the weighted_matrix
        weighted_matrix = self.weight_board(deepcopy(env))
        # print("self shape", env.shape)
        depth = 3

        temp_max_child = 0
        indices = []

        for i, p in enumerate(possible):
            if p: indices.append(i)
        # print("first indices ", indices)

        # Root Max
        # set up
        value = -1000000
        self_player = self.position
        if self.position == 1:
            opposite_player = 2
        else:
            opposite_player = 1

        # run possiblilties
        for child in indices:
            # print("current child, self_player",  child, self_player)


            # print("after init move", env.board)
            x = self.MinMax(env, child, depth - 1, True, weighted_matrix, self_player, opposite_player)
            #value = max(value, x)

            """
            if self.last_step_block(env, child, self_player, opposite_player):
                x = 2000000
            elif self.stop_move(env, child, self_player, opposite_player):
                x = -2000000
            """
            value = max(value, x)

            print("current MinMax value is", x, "compare to previous largest", temp_max_child)
            # update the best child so far
            if x > temp_max_child:
                temp_max_child = x

                print("update child move in root, current temp_max_child", child, temp_max_child)
                move[:] = [child]

        print("Finish in time move", move)

class alphaBetaAI(connect4Player):  # TODO

    def weight_board(self, env):
        print("env shape", env.shape)
        a = np.zeros(env.shape).astype('int32')
        for i in range(env.shape[0]):
            a[i] = [2, 5, 6, 7, 6, 5, 2]
        b = np.zeros(env.shape).astype('int32')
        c = np.array([[1], [2], [3], [3], [2], [1]])
        b[:, 0:7] = c

        weighted_board = a * b

        # weight_on_cross = np.array([[4, 3, 3, 2, 3, 3, 4]])

        weighted_board[5] = [20, 20, 20, 30, 20, 20, 20]

        """
        for i in range(3):
            weighted_board[i] = weighted_board[i] + weight_on_cross - i
        weighted_board[5] = weighted_board[5] * 2.5
        weighted_board[4] = weighted_board[4] * 2
        weighted_board[3] = weighted_board[3] * 2
        weighted_board[2] = weighted_board[2] * 2
        """
        print("scores")
        print(weighted_board)
        return weighted_board

    def evaluation(self, env, move, weighted_matrix):
        # If one more step needed; set it to infinity
        # Either win the game, or block
        if self.position == 1:
            opp_player = 2
        else:
            opp_player = 1


        if env.gameOver(move, self.position):
            print("one step to finish game, move", move)
            print("eval finish game board", env.board)
            return 100000

        if env.gameOver(move, opp_player):
            return -100000

        print("In evaluation move is ", move, " topPosition", env.topPosition, "  board\n", env.board)

        scores = 0

        for x in range(0, 6):
            for y in range(0, 7):
                if env.board[x][y] == self.position:
                    scores += weighted_matrix[x][y]
                elif env.board[x][y] == opp_player:
                    scores -= weighted_matrix[x][y]

        print("temp scores", scores)

        # Horizontol Check
        for row in range(env.shape[0]):
            row_array = []
            for i in list(env.board[row, :]):
                row_array.append(i)
            for col in range(env.shape[1] - 3):
                window = row_array[col: col + 4]
                if window.count(self.position) == 4:
                    scores += 500*1.2
                elif window.count(self.position) == 3 and window.count(0) == 1:
                    scores += 200*1.2
                elif window.count(opp_player) == 3 and window.count(0) == 1:
                    scores -= 150*1.2
                elif window.count(self.position) == 2 and window.count(0) == 2:
                    scores += 50*1.2
                elif window.count(opp_player) == 2 and window.count(0) == 2:
                    scores -= 50

        # Vertical Check
        for col in range(env.shape[1]):
            column = []
            for i in list(env.board[:, col]):
                column.append(i)
            for row in range(env.shape[0] - 3):
                window = column[row: row + 4]
                if window.count(self.position) == 4:
                    scores += 500*1.1
                elif window.count(self.position) == 3 and window.count(0) == 1:
                    scores += 200*1.1
                elif window.count(opp_player) == 3 and window.count(0) == 1:
                    scores -= 150*1.1
                elif window.count(self.position) == 2 and window.count(0) == 2:
                    scores += 50*1.1

        # Left-up Check
        for row in range(env.shape[0] - 3):
            for col in range(env.shape[1] - 3):
                window = []
                for i in range(4):
                    window.append([env.board[row + 3 - i][col + i]])
                if window.count(self.position) == 4:
                    scores += 500*1.1
                elif window.count(self.position) == 3 and window.count(0) == 1:
                    scores += 200*1.1
                elif window.count(opp_player) == 3 and window.count(0) == 1:
                    scores -= 150*1.1
                elif window.count(self.position) == 2 and window.count(0) == 2:
                    scores += 50*1.1

        # Right-up Check
        for row in range(env.shape[0] - 3):
            for col in range(env.shape[1] - 3):
                window = []
                for i in range(4):
                    window.append([env.board[row + i][col + i]])
                window = [j for sub in window for j in sub]
                if window.count(self.position) == 4:
                    scores += 500*1.1
                elif window.count(self.position) == 3 and window.count(0) == 1:
                    scores += 200*1.1
                elif window.count(opp_player) == 3 and window.count(0) == 1:
                    scores -= 150*1.1
                elif window.count(self.position) == 2 and window.count(0) == 2:
                    scores += 50*1.1


        # eva_value = weighted_matrix[env.topPosition[move] + 1][move]
        print("In evaluation eva_value ", scores)

        return scores

    def simulateMove(self, env, move, player):
        # print("In simulate move, player with move", player, move)
        # print("top position", env.topPosition)
        # print("board before move", env.board)
        # print("[env.topPosition[move]][move]",env.topPosition[move], move)
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1

        return env

    def last_step_block(self, env, move, self_player, opposite_player):
        print("llast step check board\n", env.board)
        new_env = self.simulateMove(deepcopy(env), move, opposite_player)

        if new_env.gameOver(move, opposite_player):
            print("last step to lose!! if opposite goes", move)
            return True
        else:
            return False

    def alphaBeta(self, env, move, depth, is_Max, weighted_matrix, self_player, opposite_player, alpha, beta):
        #print("Alpha-beta Depth:", depth, " is Max-turn:", is_Max, " move:", move, "self_player: ", self_player, "opp_player: ", opposite_player)

        # Simulate Move
        if is_Max:
            cur_player = self_player
            new_env = self.simulateMove(deepcopy(env), move, self_player)

        else:
            cur_player = opposite_player
            new_env = self.simulateMove(deepcopy(env), move, opposite_player)

        try:
            if (depth == 0) or new_env.gameOver(move, cur_player):
                # print("Game ends in MinMax, move is",move)
                eva_value = self.evaluation(new_env, move, weighted_matrix)
                # print("Evaluation value in MinMax is", eva_value)
                return eva_value
        except:
            if is_Max:
                print("Max turn error")
            else:
                print("Min turn error")
            print("Error from Min_Max gameOver move ", move, " env.topPosition ", new_env.topPosition, "env.board \n",
                  new_env.board)
            raise

        if is_Max:
            value = -10000000
            # new_env = self.simulateMove(deepcopy(env), move, self_player)

            # update possible
            possible = env.topPosition >= 0
            indices = []
            for i, p in enumerate(possible):
                if p: indices.append(i)

            for child in indices:
                is_Max = False
                value = max(value, self.alphaBeta(new_env, child, depth - 1, is_Max, weighted_matrix, self_player,
                                               opposite_player, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else: #Min Turn
            value = 10000000
            # new_env = self.simulateMove(deepcopy(env), move, opposite_player)

            # update possible
            possible = env.topPosition >= 0
            indices = []
            for i, p in enumerate(possible):
                if p: indices.append(i)

            for child in indices:
                is_Max = True
                value = min(value, self.alphaBeta(new_env, child, depth - 1, is_Max, weighted_matrix, self_player,
                                               opposite_player, alpha, beta))
                beta = min (beta, value)
                if beta <= alpha:
                    break
            return value

    def play(self, env, move):
        env = deepcopy(env)
        env.visualize = False
        possible = env.topPosition >= 0
        # Set up the weighted_matrix
        weighted_matrix = self.weight_board(deepcopy(env))
        # print("self shape", env.shape)
        depth = 7

        temp_max_child = 0
        indices = []

        for i, p in enumerate(possible):
            if p: indices.append(i)
        print("first indices ", indices, type(indices))

        # Root Max
        # set up
        value = -1000000
        self_player = self.position
        if self.position == 1:
            opposite_player = 2
        else:
            opposite_player = 1

        # Successor function; re-order nodes
        order_indices = []

        if len(indices) % 2 != 0:
            mid = (len(indices) - len(indices) % 2) / 2
            mid = int(mid)
            order_indices.append(indices[mid])
            for i in range(1, mid + 1):
                order_indices.append(indices[i + mid])
                order_indices.append(indices[mid - i])
        else:
            mid = (len(indices) - len(indices) % 2) / 2
            mid = int(mid - 1)
            order_indices.append(indices[mid])
            i = 1
            while i <= mid:
                order_indices.append(indices[mid + i])
                order_indices.append(indices[mid - i])
                i += 1
            order_indices.append(len(indices))

        print("Ordered indices ", order_indices)


        # run possiblilties
        for child in order_indices:
            # print("current child, self_player",  child, self_player)

            # print("after init move", env.board)
            x = self.alphaBeta(env, child, depth - 1, True, weighted_matrix, self_player, opposite_player,1000000,-1000000)
            # value = max(value, x)

            if self.last_step_block(env, child, self_player, opposite_player):
                x = 2000000

            value = max(value, x)

            #print("current MinMax value is", x, "compare to previous largest", temp_max_child, "current move", child)
            # update the best child so far
            if x > temp_max_child:
                temp_max_child = x

                #print("update child move in root, current temp_max_child", child, temp_max_child)
                move[:] = [child]

        print("Finish in time move")


SQUARESIZE = 100
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
