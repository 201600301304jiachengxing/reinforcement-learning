"""
Created on Tue Nov 12 19:40:10 2019

@author: Amy Su
"""


class SaveingError(Exception):
    pass


def newGame(player1, player2):
    game = dict()
    game['player1'] = player1
    game['player2'] = player2
    game['who'] = 1
    game['board'] = [[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]]
    return game


def printBoard(board):
    for a in range(1, 7):
        print('|' + str(a) + ' ', end='')
    print('|7 |')
    print('+-+-+-+-+-+-+-+-+-+-+-+')
    for k in range(6):
        for h in range(7):
            if h != 6:
                if board[k][h] == 0:
                    print('|' + '  ', end='')
                elif board[k][h] == 1:
                    print('|' + 'O ', end='')
                elif board[k][h] == 2:
                    print('|' + 'X ', end='')
            else:
                if board[k][h] == 0:
                    print('|' + '  ' + '|')
                elif board[k][h] == 1:
                    print('|' + 'O ' + '|')
                elif board[k][h] == 2:
                    print('|' + 'X ' + '|')


def getValidMoves(board):
    L = []
    for j in range(7):
        for i in range(6):
            if board[i][j] == 0:
                L.append(j + 1)
                break
            else:
                continue
    return L


def makeMove(board, move, who):
    msg = True
    for k in range(6):
        if board[k][move - 1] != 0:
            msg = False
            break
    if k == 5 and msg == True:
        board[k][move - 1] = who
    else:
        board[k - 1][move - 1] = who
    return board


def hasWon(board, who):
    # checking the horizontal lines
    for k in range(6):
        for r in range(4):
            if board[k][r] == board[k][r + 1] == board[k][r + 2] == board[k][r + 3] == int(who):
                return True
    # checking the vertical lines
    for k in range(3):
        for r in range(7):
            if board[k][r] == board[k + 1][r] == board[k + 2][r] == board[k + 3][r] == int(who):
                return True
    # checking the positive diagnal lines
    for k in range(3):
        for r in range(3, 7):
            if board[k][r] == board[k + 1][r - 1] == board[k + 2][r - 2] == board[k + 3][r - 3] == int(who):
                return True
    # checking the negative diagnol lines
    for k in range(3):
        for r in range(4):
            if board[k][r] == board[k + 1][r + 1] == board[k + 2][r + 2] == board[k + 3][r + 3] == int(who):
                return True
    return False


def suggestMove(board, who):
    from random import choice
    import copy
    L = getValidMoves(board)
    opponent = who % 2 + 1
    for k in L:
        board2 = copy.deepcopy(board)
        board2 = makeMove(board2, k, who)
        if hasWon(board2, who) == True:
            return k
    for k in L:
        board2 = copy.deepcopy(board)
        board2 = makeMove(board2, k, opponent)
        if hasWon(board2, opponent):
            return k
    random_number = choice(L)
    return random_number
    # my next step is goona to make me win


def loadGame(filename='game'):
    # not tested so long
    try:
        with open(filename + '.txt', mode='rt', encoding='utf8') as file:
            player1 = file.readline().strip('\n')
            player2 = file.readline().strip('\n')
            who = file.readline().strip('\n')
            board = list()
            for _ in range(6):
                L = ''.join(file.readline().split(','))
                L = list(L)
                L = L[:7]
                L = [int(a) for a in L]
                board.append(L)
            game = dict()
            game['player1'] = player1
            game['player2'] = player2
            game['who'] = who
            game['board'] = board
    except FileNotFoundError:
        print('the file cannot be loaded')
        pass
    except ValueError:
        print('the content is not of the correct format')
        pass
    return game


def play():
    print('This is a connect Four game!')
    print(
        "Enter the players' names, or type 'C' for a simple opponent , enter 'Hard'for a difficult opponent or enter L for load the last game")
    print("Okey, let's play!")
    gameover = False
    turn = 1
    load = False
    while True:
        try:
            player1 = str.title(input('Name of player 1: '))
            if player1 == 'L':
                filename = input('please enter the filename')
                try:
                    game = loadGame(filename)
                    printBoard(game['board'])
                    load = True
                    break
                except Exception:
                    return
            player2 = str.title(input('Name of player2: '))
            break
        except Exception:
            pass
    if load == False:
        game = newGame(player1, player2)
        printBoard(game['board'])
    while not gameover:
        # ask for Player 1 input
        if turn == 1:
            game['who'] = 1
            while True:
                # print(game)
                if game['player1'] == 'C':
                    move1 = suggestMove(game['board'], game['who'])
                    game['board'] = makeMove(game['board'], move1, game['who'])
                    turn = 2
                    print('Computer (O) is thinking... and selected column ' + str(move1))
                    break
                elif game['player1'] == 'Hard':
                    move1 = suggestMove2(game['board'], game['who'])
                    game['board'] = makeMove(game['board'], move1, game['who'])
                    turn = 2
                    print('Computer (O) is thinking... and selected column ' + str(move1))
                    break
                elif game['player1'] == 'Tf':
                    move1 = suggestMove3(game['board'], game['who'])
                    game['board'] = makeMove(game['board'], move1, game['who'])
                    turn = 2
                    print('Computer (O) is thinking... and selected column ' + str(move1))
                    break
                else:
                    move2 = input(game['player1'] + ' (O): Which column to select? ')
                    if move2.title() == 'S':
                        try:
                            filename = input('which file name you want to save')
                            if filename == '':
                                filename = 'game'
                            saveGame(game, filename)
                            print('the game has been saved to a file')
                        except SaveingError:
                            print('saving process fails')
                            pass
                    else:
                        try:
                            move2 = int(move2)
                        except Exception:
                            pass
                    if move2 in getValidMoves(game['board']):
                        game['board'] = makeMove(game['board'], move2, game['who'])
                        turn = 2
                        break
                    else:
                        continue
            if hasWon(game['board'], 1):
                print(game['player1'] + ' has WON!!!')
                gameover = True
        else:
            game['who'] = 2
            while True:
                if game['player2'] == 'C':
                    move3 = suggestMove(game['board'], game['who'])
                    game['board'] = makeMove(game['board'], move3, game['who'])
                    turn = 1
                    print('Computer (X) is thinking... and selected column ' + str(move3))
                    break
                elif game['player2'] == 'Hard':
                    move3 = suggestMove2(game['board'], game['who'])
                    game['board'] = makeMove(game['board'], move3, game['who'])
                    turn = 1
                    print('Computer (X) is thinking... and selected column' + str(move3))
                    break
                elif game['player2'] == 'Tf':
                    move3 = suggestMove3(game['board'], game['who'])
                    game['board'] = makeMove(game['board'], move3, game['who'])
                    turn = 1
                    print('Computer (X) is thinking... and selected column' + str(move3))
                    break
                else:
                    move4 = input(game['player2'] + ' (X): Which column to select? ')
                    if move4.title() == 'S':
                        try:
                            filename = input('which file name you want to save')
                            if filename == '':
                                filename = 'game'
                            saveGame(game, filename)
                            print('the game has been saved to a file')
                        except SaveingError:
                            print('saving process fails')
                            pass

                    else:
                        try:
                            move4 = int(move4)
                        except Exception:
                            pass
                    if move4 in getValidMoves(game['board']):
                        game['board'] = makeMove(game['board'], move4, game['who'])
                        turn = 1
                        break
                    else:
                        continue
            if hasWon(game['board'], 2):
                print(game['player2'] + ' has WON!!!')
                gameover = True
        L = getValidMoves(game['board'])
        if gameover == False and L == []:
            print('there was a draw')
            gameover = True
        printBoard(game['board'])
    return 0


def saveGame(game, filename):
    try:
        with open(filename + '.txt', mode='wt', encoding='utf8') as file:
            file.write(game['player1'] + '\n')
            file.write(game['player2'] + '\n')
            file.write(str(game['who']) + '\n')
            for L in game['board']:
                string = '' + str(L[0])
                for m in L[1:]:
                    string = string + ',' + str(m)
                file.write(string + '\n')
    except Exception:
        raise SaveingError('the file fails to be saved')
    return 0


def ranking(board, who):
    Lit = list()
    L = list()
    import copy
    # checking the horizontal lines
    L = getValidMoves(board)

    for k in L:
        board2 = copy.deepcopy(board)
        board2 = makeMove(board2, k, who)
        k = k - 1
        for m in range(6):
            if board2[m][k] == who:
                break
        try:
            for r in range(-2, 1):
                if board2[m][k + r] == board2[m][k + r + 1] == board2[m][k + r + 2] == int(
                        who) and k + r >= 0 and k + r + 2 <= 6:
                    printBoard(board2)
                    Lit.append(k + 1)
                    Lit.append('triple')
                    return Lit
        except Exception:
            pass
        # checking the vertical lines
        try:
            for r in range(-2, 1):
                if board2[m + r][k] == board2[m + r + 1][k] == board2[m + r + 2][k] == int(
                        who) and m + r + 2 <= 5 and m + r > 0:
                    Lit.append(k + 1)
                    Lit.append('triple')
                    return Lit
        except Exception:
            pass
        # checking the positive diagnal lines
        try:
            for r in range(-2, 1):
                if board2[m + r][k + r] == board2[m + r + 1][k + r + 1] == board2[m + r + 2][k + r + 2] == int(
                        who) and m + r >= 0 and m + r + 2 <= 5 and k + r >= 0 and k + r + 2 <= 6:
                    Lit.append(k + 1)
                    Lit.append('triple')
                    return Lit
        except Exception:
            pass
        # checking the negative diagnol lines
        try:
            for r in range(-2, 1):
                if board2[m + r][k - r] == board2[m + r + 1][k - r + 1] == board2[m + r + 2][k - r + 2] == int(
                        who) and m + r >= 0 and m + r + 2 <= 5 and k - r >= 0 and k - r + 2 <= 6:
                    Lit.append(k + 1)
                    Lit.append('triple')
                    return Lit
        except Exception:
            pass
    for k in L:
        board2 = copy.deepcopy(board)
        board2 = makeMove(board2, k, who)
        # cheking for horizontal line
        try:
            for r in range(-1, 1):
                if board2[m][k + r] == board2[m][k + r + 1] == int(who):
                    Lit.append(k + 1)
                    Lit.append('double')
                    return Lit
        except Exception:
            pass
        # checking the vertical lines
        try:
            for r in range(-1, 1):
                if board2[m + r][k] == board2[m + r + 1][k] == int(who):
                    Lit.append(k + 1)
                    Lit.append('double')
                    return Lit
        except Exception:
            pass
        # checking the positive diagnal lines
        try:
            for r in range(-1, 1):
                if board2[m + r][k + r] == board2[m + r + 1][k + r + 1] == int(who):
                    Lit.append(k + 1)
                    Lit.append('double')
                    return Lit
        except Exception:
            pass
        # checking the negative diagnol lines
        try:
            for r in range(-1, 1):
                if board2[m + r][k - r] == board2[m + r + 1][k - r + 1] == int(who):
                    Lit.append(k + 1)
                    Lit.append('double')
                    return Lit
        except Exception:
            pass
    Lit = [0, 'single']
    return Lit



def suggestMove2(board, who):
    from random import choice
    import copy
    # my next step is goona to make me win
    L = getValidMoves(board)
    opponent = who % 2 + 1
    for k in L:
        board2 = copy.deepcopy(board)
        board2 = makeMove(board2, k, who)
        if hasWon(board2, who) == True:
            return k
        else:
            continue
    # my next step is to prevent the opponent form winning
    for k in L:
        board2 = copy.deepcopy(board)
        board2 = makeMove(board2, k, opponent)
        if hasWon(board2, opponent) == True:
            return k
        else:
            continue
    result_me = ranking(board, who)
    result = ranking(board, opponent)
    if result_me[1] == 'triple':
        return result_me[0]
    if result == 'triple':
        return result[0]
    if result_me[1] == 'double':
        return result_me[0]
    if result[1] == 'double':
        return result[0]
    random_number = choice(L)
    return random_number


def suggestMove3(board, who):
    from random import choice
    import copy
    # my next step is goona to make me win
    L = getValidMoves(board)
    opponent = who % 2 + 1
    for k in L:
        board2 = copy.deepcopy(board)
        board2 = makeMove(board2, k, who)
        if hasWon(board2, who) == True:
            return k
        else:
            continue
    # my next step is to prevent the opponent form winning
    for k in L:
        board2 = copy.deepcopy(board)
        board2 = makeMove(board2, k, opponent)
        if hasWon(board2, opponent) == True:
            return k
        else:
            continue

    import numpy as np
    mask = np.zeros(7)
    for k in L:
        mask[k - 1] = 1
    observation = copy.deepcopy(board)
    ob[0:3] = ob[1:4]
    ob[3] = np.array([observation])
    action, _ = RL.plan_with_mask([ob], mask)

    print(action)
    return action[0] + 1

from c4_game import *
import torch
RL = MOdel((4,6,7), 7, 250)
RL.nets.dynamics = torch.load('d.pkl')
RL.nets.prediction = torch.load('p.pkl')
RL.nets.representation = torch.load('r.pkl')

ob = np.zeros((4,6,7))


play()
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

board = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0]]





