def minimax_agent(observation, configuration):
    import random
    import time
    import numpy as np
    from copy import deepcopy
    start_time = time.time()
    # Number of Columns on the Board.
    columns = configuration.columns
    # Number of Rows on the Board.
    rows = configuration.rows
    # Number of Checkers "in a row" needed to win.
    inarow = configuration.inarow
    # The current serialized Board (rows x columns).
    board = observation.board
    # Which player the agent is playing as (1 or 2).
    mark = observation.mark
    boardarray = np.array(board).reshape(rows, columns).tolist()
    inf = np.inf
    nodesExpanded = 0
    depth = 0
    maxDepth = 1
    timeAmount = configuration.timeout - 1
    
    #a class representing the board and containing helper functions for board tree search
    class Connect(object):
        def __init__(self, board, columns, rows, mark, inarow, depth=0, parent=None, indexNum=None):
            self.board = board #board state
            self.columns = columns #number of columns
            self.rows = rows #number of rows
            self.mark = mark #what the newly placed mark should be
            self.inarow = inarow #how many to match in a row
            self.depth = depth #how far the tree has been expanded so far
            self.parent = parent #the parent that the board came from
            self.indexNum = indexNum #the piece that was just placed
        def getMoves(self):
            #get all possible moves by checking if the top of the board is empty for each column
            moves = []
            if len(moves) == 0:
                for col in range(self.columns):
                    if self.board[0][col] == 0:
                        moves.append(col)
            return moves
        
        
        def evaluate(self):
            
            #return a score evaluating the board, positive for favoring player 1, negative for favoring player 2, and magnitude of infinity for if a certain player has won
            score = 0
            #how much more weight a longer sequence of pieces should have over a shorter one (ex. 3 in a row is [branchConstant] times more important than 2 in a row)
            branchConstant = 6
            #lists marks on board that are beneficial for each respective player (ex. player 1 can have 0 or 1, player 2 can have 0 or 2)
            allowed = [[0, self.mark], [0, 3 - self.mark]]
            
            #code that counts the marks of each side to determine which player moves next
            playerTurn = 0
            for row in range(self.rows):
                for col in range(self.columns):
                    if self.board[row][col] == 1:
                        playerTurn += 1
                    elif self.board[row][col] == 2:
                        playerTurn -= 1
            playerTurn += 1
            
            
            #checks conditions for both players to compare them
            for turn in range(len(allowed)):
                #iterates over rows from bottom to top
                for row in range(self.rows - 1, -1, -1):
                    #iterates over columns
                    for col in range(self.columns):
                        #the following code checks for certain conditions in the patterns:
                        #the four patterns: vertical line, horizontal line, diagonal 1 and diagonal 2
                        #consistency: how much of a line is filled with 0's or the player's mark
                        
                        
                        #vertical lines
                        #don't go so far down the board that the vertical line goes off the board
                        if row < self.rows - (self.inarow - 1):
                            consistency = 0
                            
                            #counts how many places in the vertical line are 0 or the player's mark
                            for inc in range(self.inarow):
                                if self.board[row + inc][col] in allowed[turn]:
                                    consistency += 1
                            #if there are no opponent's marks in that line, then calculate and add score
                            if consistency == self.inarow:
                                consistency = 0
                                
                                #code to check how many marks are in a row
                                for inc2 in range(self.inarow):
                                    consistency += allowed[turn].index(self.board[row + inc2][col])
                                #add score
                                score += branchConstant ** consistency if turn == 0 else -1 * branchConstant ** consistency
                                #if the winning amount is in a row then return winning magnitude score
                                if consistency == self.inarow:
                                    return inf if turn == 0 else -inf
                                #if one more mark in a row to win and its that players turn then return winning magnitude score
                                if consistency == self.inarow - 1:
                                    for inc3 in range(self.inarow):
                                        if self.board[row + inc3][col] == 0:
                                            if playerTurn == allowed[turn][1] and (row + inc3 == self.rows - 1 or self.board[row + inc3 + 1][col] != 0):
                                                return inf if turn == 0 else -inf
                        
                        #horizontal lines
                        #don't go so far left on the board that the horizontal line goes off the board
                        if col < self.columns - (self.inarow - 1):
                            consistency = 0
                            
                            #counts how many places in the vertical line are 0 or the player's mark
                            for inc in range(self.inarow):
                                if self.board[row][col + inc] in allowed[turn]:
                                    consistency += 1
                            #if there are no opponent's marks in that line, then calculate and add score
                            if consistency == self.inarow:
                                consistency = 0
                                
                                #code to check how many marks are in a row
                                for inc2 in range(self.inarow):
                                    consistency += allowed[turn].index(self.board[row][col + inc2])
                                #add score
                                score += branchConstant ** consistency if turn == 0 else -1 * branchConstant ** consistency
                                #if the winning amount is in a row then return winning magnitude score
                                if consistency == self.inarow:
                                    return inf if turn == 0 else -inf
                                #if one more mark in a row to win and its that players turn then return winning magnitude score
                                if consistency == self.inarow - 1:
                                    for inc3 in range(self.inarow):
                                        if self.board[row][col + inc3] == 0:
                                            if playerTurn == allowed[turn][1] and (row == self.rows - 1 or self.board[row + 1][col + inc3] != 0):
                                                return inf if turn == 0 else -inf
                        
                        #diagonal 1
                        #don't go so far to the bottom right of the board that the diagonal line goes off the board
                        if row < self.rows - (self.inarow - 1) and col < self.columns - (self.inarow - 1):
                            consistency = 0
                            
                            #counts how many places in the vertical line are 0 or the player's mark
                            for inc in range(self.inarow):
                                if self.board[row + inc][col + inc] in allowed[turn]:
                                    consistency += 1
                            #if there are no opponent's marks in that line, then calculate and add score
                            if consistency == self.inarow:
                                consistency = 0
                                
                                #code to check how many marks are in a row
                                for inc2 in range(self.inarow):
                                    consistency += allowed[turn].index(self.board[row + inc2][col + inc2])
                                #add score
                                score += branchConstant ** consistency if turn == 0 else -1 * branchConstant ** consistency
                                #if the winning amount is in a row then return winning magnitude score
                                if consistency == self.inarow:
                                    return inf if turn == 0 else -inf
                                #if one more mark in a row to win and its that players turn then return winning magnitude score
                                if consistency == self.inarow - 1:
                                    for inc3 in range(self.inarow):
                                        if self.board[row + inc3][col + inc3] == 0:
                                            if playerTurn == allowed[turn][1] and (row + inc3 == self.rows - 1 or self.board[row + inc3 + 1][col + inc3] != 0):
                                                return inf if turn == 0 else -inf
                        
                        #diagonal 2
                        #don't go so far to the bottom left of the board that the diagonal line goes off the board
                        if row > self.inarow - 2 and col < self.columns - (self.inarow - 1):
                            consistency = 0
                            
                            #counts how many places in the vertical line are 0 or the player's mark
                            for inc in range(self.inarow):
                                if self.board[row - inc][col + inc] in allowed[turn]:
                                    consistency += 1
                            #if there are no opponent's marks in that line, then calculate and add score
                            if consistency == self.inarow:
                                consistency = 0
                                
                                #code to check how many marks are in a row
                                for inc2 in range(self.inarow):
                                    consistency += allowed[turn].index(self.board[row - inc2][col + inc2])
                                #add score
                                score += branchConstant ** consistency if turn == 0 else -1 * branchConstant ** consistency
                                #if the winning amount is in a row then return winning magnitude score
                                if consistency == self.inarow:
                                    return inf if turn == 0 else -inf
                                #if one more mark in a row to win and its that players turn then return winning magnitude score
                                if consistency == self.inarow - 1:
                                    for inc3 in range(self.inarow):
                                        if self.board[row - inc3][col + inc3] == 0:
                                            if playerTurn == allowed[turn][1] and (row - inc3 == self.rows - 1 or self.board[row - inc3 + 1][col + inc3] != 0):
                                                return inf if turn == 0 else -inf
            
            return score
        def makeMove(self, col, marker):
            #gets the 'child' of the current board created by making move in the [col] column
            board2 = [row[:] for row in self.board]
            for row in range(self.rows - 1,-1,-1):
                if board2[row][col] == 0:
                    board2[row][col] = marker
                    return Connect(board2, self.columns, self.rows, self.mark, self.inarow, self.depth + 1, self, row * self.columns + col)
                    break
        def display(self):
            #displays the connect grid
            boardstring = ""
            for row in range(self.rows):
                for col in range(self.columns):
                    boardstring += str(self.board[row][col])
                boardstring += "\n"
            print(boardstring)
        def terminal_test(self):
            #returns -2 if the maximum depth or search time is exceeded, 0 if the game isn't over, or 1/-1 for which player won
            #its the same code as evaluate but without the score adding
            
            nonlocal maxDepth, timeAmount
            #no need to check if the game is won if depth is 0 because then it wouldn't be called
            if self.depth == 0:
                return 0
            allowed = [self.mark, 3 - self.mark]
            for turn in allowed:
                for row in range(self.rows - 1, -1, -1):
                    for col in range(self.columns):
                        #vertical
                        if row < self.rows - (self.inarow - 1):
                            consistency = 0
                            for inc in range(self.inarow):
                                if self.board[row + inc][col] == turn:
                                    consistency += 1
                            if consistency == self.inarow:
                                return 1 if turn == self.mark else -1
                        #horizontal
                        if col < self.columns - (self.inarow - 1):
                            consistency = 0
                            for inc in range(self.inarow):
                                if self.board[row][col + inc] == turn:
                                    consistency += 1
                            if consistency == self.inarow:
                                return 1 if turn == self.mark else -1
                        #diagonal 1
                        if row < self.rows - (self.inarow - 1) and col < self.columns - (self.inarow - 1):
                            consistency = 0
                            for inc in range(self.inarow):
                                if self.board[row + inc][col + inc] == turn:
                                    consistency += 1
                            if consistency == self.inarow:
                                return 1 if turn == self.mark else -1
                        #diagonal 2
                        if row > self.inarow - 2 and col < self.columns - (self.inarow - 1):
                            consistency = 0
                            for inc in range(self.inarow):
                                if self.board[row - inc][col + inc] == turn:
                                    consistency += 1
                            if consistency == self.inarow:
                                return 1 if turn == self.mark else -1
            #if depth or time is exceeded, return -2
            if time.time() - start_time > timeAmount or self.depth == maxDepth:
                return -2
            return 0
    
    def minimize(state, alpha, beta, useAB):     
        #minimax with useAB determining whether to use alpha-beta pruning
        nonlocal depth, nodesExpanded
        depth = state.depth if state.depth > depth else depth
        termScore = state.terminal_test()
        if termScore == -1:
            return (None, -inf)
        elif termScore == 1:
            return (None, inf)
        elif termScore == -2:
            return (None, state.evaluate())

        minChildMinUtility = (None, inf)
        possibleMoves = state.getMoves()
        for move in possibleMoves:
            nodesExpanded += 1
            child = state.makeMove(move, 3 - state.mark)
            maxChildMaxUtility = maximize(child, alpha, beta, useAB)
            if maxChildMaxUtility[1] < minChildMinUtility[1]:
                minChildMinUtility = (child, maxChildMaxUtility[1])
            if minChildMinUtility[1] == -inf:
                return minChildMinUtility
            if useAB and minChildMinUtility[1] <= alpha:
                break
            if useAB and minChildMinUtility[1] < beta:
                beta = minChildMinUtility[1]
        return minChildMinUtility

    def maximize(state, alpha, beta, useAB):
        #minimax with useAB determining whether to use alpha-beta pruning
        nonlocal depth, nodesExpanded
        depth = state.depth if state.depth > depth else depth
        termScore = state.terminal_test()
        if termScore == -1:
            return (None, -inf)
        elif termScore == 1:
            return (None, inf)
        elif termScore == -2:
            return (None, state.evaluate())
        
        maxChildMaxUtility = (None, -inf)
        possibleMoves = state.getMoves()
        for move in possibleMoves:
            nodesExpanded += 1
            child = state.makeMove(move, state.mark)
            minChildMinUtility = minimize(child, alpha, beta, useAB)
            if minChildMinUtility[1] > maxChildMaxUtility[1]:
                maxChildMaxUtility = (child, minChildMinUtility[1])
            if maxChildMaxUtility[1] == inf:
                return maxChildMaxUtility
            if useAB and maxChildMaxUtility[1] >= beta:
                break
            if useAB and maxChildMaxUtility[1] > alpha:
                alpha = maxChildMaxUtility[1]
        return maxChildMaxUtility
    #create board, and check if there is only one move to make, and make that move if there is (no need to search the tree)
    currentBoard = Connect(boardarray, columns, rows, mark, inarow)
    moves = currentBoard.getMoves()
    if len(moves) == 0:
        return None
    if len(moves) == 1:
        return moves[0]
    
    #iterative deepening with the minimax search
    #starts at depth 1 and continues increasing depth until time limit is reached and makes a conclusion on the best move to make
    childUtility = (None, 0)
    prevChildUtility = (None, 0)
    utilities = []
    #while loop keeps expanding depth until the time limit is exceeded or a guaranteed win is calculated
    while time.time() - start_time < timeAmount and childUtility[1] != inf:
        prevChildUtility = childUtility
        childUtility = maximize(currentBoard, -inf, inf, True)
        utilities.append(childUtility[1])
        maxDepth += 1
    #print(str(utilities) + " " + str(time.time() - start_time)[0:4] + " " + str(nodesExpanded) + " " + str(maxDepth) + " " + str(mark))
    #code to check possible errors before selecting the move
    if time.time() - start_time < timeAmount and childUtility[1] == inf:
        return childUtility[0].indexNum % childUtility[0].columns
    if prevChildUtility[0] == None:
        if childUtility[0] == None:
            #print(str(maxDepth) + "error found")
            return moves[random.randint(0, len(moves) - 1)] if moves else None
        return childUtility[0].indexNum % childUtility[0].columns
    return prevChildUtility[0].indexNum % prevChildUtility[0].columns
def minimax_agent(observation, configuration):
    import random
    import time
    import numpy as np
    from copy import deepcopy
    start_time = time.time()
    # Number of Columns on the Board.
    columns = configuration.columns
    # Number of Rows on the Board.
    rows = configuration.rows
    # Number of Checkers "in a row" needed to win.
    inarow = configuration.inarow
    # The current serialized Board (rows x columns).
    board = observation.board
    # Which player the agent is playing as (1 or 2).
    mark = observation.mark
    boardarray = np.array(board).reshape(rows, columns).tolist()
    inf = np.inf
    nodesExpanded = 0
    depth = 0
    maxDepth = 1
    timeAmount = configuration.timeout - 1
    
    #a class representing the board and containing helper functions for board tree search
    class Connect(object):
        def __init__(self, board, columns, rows, mark, inarow, depth=0, parent=None, indexNum=None):
            self.board = board #board state
            self.columns = columns #number of columns
            self.rows = rows #number of rows
            self.mark = mark #what the newly placed mark should be
            self.inarow = inarow #how many to match in a row
            self.depth = depth #how far the tree has been expanded so far
            self.parent = parent #the parent that the board came from
            self.indexNum = indexNum #the piece that was just placed
        def getMoves(self):
            #get all possible moves by checking if the top of the board is empty for each column
            moves = []
            if len(moves) == 0:
                for col in range(self.columns):
                    if self.board[0][col] == 0:
                        moves.append(col)
            return moves
        
        
        def evaluate(self):
            
            #return a score evaluating the board, positive for favoring player 1, negative for favoring player 2, and magnitude of infinity for if a certain player has won
            score = 0
            #how much more weight a longer sequence of pieces should have over a shorter one (ex. 3 in a row is [branchConstant] times more important than 2 in a row)
            branchConstant = 6
            #lists marks on board that are beneficial for each respective player (ex. player 1 can have 0 or 1, player 2 can have 0 or 2)
            allowed = [[0, self.mark], [0, 3 - self.mark]]
            
            #code that counts the marks of each side to determine which player moves next
            playerTurn = 0
            for row in range(self.rows):
                for col in range(self.columns):
                    if self.board[row][col] == 1:
                        playerTurn += 1
                    elif self.board[row][col] == 2:
                        playerTurn -= 1
            playerTurn += 1
            
            
            #checks conditions for both players to compare them
            for turn in range(len(allowed)):
                #iterates over rows from bottom to top
                for row in range(self.rows - 1, -1, -1):
                    #iterates over columns
                    for col in range(self.columns):
                        #the following code checks for certain conditions in the patterns:
                        #the four patterns: vertical line, horizontal line, diagonal 1 and diagonal 2
                        #consistency: how much of a line is filled with 0's or the player's mark
                        
                        
                        #vertical lines
                        #don't go so far down the board that the vertical line goes off the board
                        if row < self.rows - (self.inarow - 1):
                            consistency = 0
                            
                            #counts how many places in the vertical line are 0 or the player's mark
                            for inc in range(self.inarow):
                                if self.board[row + inc][col] in allowed[turn]:
                                    consistency += 1
                            #if there are no opponent's marks in that line, then calculate and add score
                            if consistency == self.inarow:
                                consistency = 0
                                
                                #code to check how many marks are in a row
                                for inc2 in range(self.inarow):
                                    consistency += allowed[turn].index(self.board[row + inc2][col])
                                #add score
                                score += branchConstant ** consistency if turn == 0 else -1 * branchConstant ** consistency
                                #if the winning amount is in a row then return winning magnitude score
                                if consistency == self.inarow:
                                    return inf if turn == 0 else -inf
                                #if one more mark in a row to win and its that players turn then return winning magnitude score
                                if consistency == self.inarow - 1:
                                    for inc3 in range(self.inarow):
                                        if self.board[row + inc3][col] == 0:
                                            if playerTurn == allowed[turn][1] and (row + inc3 == self.rows - 1 or self.board[row + inc3 + 1][col] != 0):
                                                return inf if turn == 0 else -inf
                        
                        #horizontal lines
                        #don't go so far left on the board that the horizontal line goes off the board
                        if col < self.columns - (self.inarow - 1):
                            consistency = 0
                            
                            #counts how many places in the vertical line are 0 or the player's mark
                            for inc in range(self.inarow):
                                if self.board[row][col + inc] in allowed[turn]:
                                    consistency += 1
                            #if there are no opponent's marks in that line, then calculate and add score
                            if consistency == self.inarow:
                                consistency = 0
                                
                                #code to check how many marks are in a row
                                for inc2 in range(self.inarow):
                                    consistency += allowed[turn].index(self.board[row][col + inc2])
                                #add score
                                score += branchConstant ** consistency if turn == 0 else -1 * branchConstant ** consistency
                                #if the winning amount is in a row then return winning magnitude score
                                if consistency == self.inarow:
                                    return inf if turn == 0 else -inf
                                #if one more mark in a row to win and its that players turn then return winning magnitude score
                                if consistency == self.inarow - 1:
                                    for inc3 in range(self.inarow):
                                        if self.board[row][col + inc3] == 0:
                                            if playerTurn == allowed[turn][1] and (row == self.rows - 1 or self.board[row + 1][col + inc3] != 0):
                                                return inf if turn == 0 else -inf
                        
                        #diagonal 1
                        #don't go so far to the bottom right of the board that the diagonal line goes off the board
                        if row < self.rows - (self.inarow - 1) and col < self.columns - (self.inarow - 1):
                            consistency = 0
                            
                            #counts how many places in the vertical line are 0 or the player's mark
                            for inc in range(self.inarow):
                                if self.board[row + inc][col + inc] in allowed[turn]:
                                    consistency += 1
                            #if there are no opponent's marks in that line, then calculate and add score
                            if consistency == self.inarow:
                                consistency = 0
                                
                                #code to check how many marks are in a row
                                for inc2 in range(self.inarow):
                                    consistency += allowed[turn].index(self.board[row + inc2][col + inc2])
                                #add score
                                score += branchConstant ** consistency if turn == 0 else -1 * branchConstant ** consistency
                                #if the winning amount is in a row then return winning magnitude score
                                if consistency == self.inarow:
                                    return inf if turn == 0 else -inf
                                #if one more mark in a row to win and its that players turn then return winning magnitude score
                                if consistency == self.inarow - 1:
                                    for inc3 in range(self.inarow):
                                        if self.board[row + inc3][col + inc3] == 0:
                                            if playerTurn == allowed[turn][1] and (row + inc3 == self.rows - 1 or self.board[row + inc3 + 1][col + inc3] != 0):
                                                return inf if turn == 0 else -inf
                        
                        #diagonal 2
                        #don't go so far to the bottom left of the board that the diagonal line goes off the board
                        if row > self.inarow - 2 and col < self.columns - (self.inarow - 1):
                            consistency = 0
                            
                            #counts how many places in the vertical line are 0 or the player's mark
                            for inc in range(self.inarow):
                                if self.board[row - inc][col + inc] in allowed[turn]:
                                    consistency += 1
                            #if there are no opponent's marks in that line, then calculate and add score
                            if consistency == self.inarow:
                                consistency = 0
                                
                                #code to check how many marks are in a row
                                for inc2 in range(self.inarow):
                                    consistency += allowed[turn].index(self.board[row - inc2][col + inc2])
                                #add score
                                score += branchConstant ** consistency if turn == 0 else -1 * branchConstant ** consistency
                                #if the winning amount is in a row then return winning magnitude score
                                if consistency == self.inarow:
                                    return inf if turn == 0 else -inf
                                #if one more mark in a row to win and its that players turn then return winning magnitude score
                                if consistency == self.inarow - 1:
                                    for inc3 in range(self.inarow):
                                        if self.board[row - inc3][col + inc3] == 0:
                                            if playerTurn == allowed[turn][1] and (row - inc3 == self.rows - 1 or self.board[row - inc3 + 1][col + inc3] != 0):
                                                return inf if turn == 0 else -inf
            
            return score
        def makeMove(self, col, marker):
            #gets the 'child' of the current board created by making move in the [col] column
            board2 = [row[:] for row in self.board]
            for row in range(self.rows - 1,-1,-1):
                if board2[row][col] == 0:
                    board2[row][col] = marker
                    return Connect(board2, self.columns, self.rows, self.mark, self.inarow, self.depth + 1, self, row * self.columns + col)
                    break
        def display(self):
            #displays the connect grid
            boardstring = ""
            for row in range(self.rows):
                for col in range(self.columns):
                    boardstring += str(self.board[row][col])
                boardstring += "\n"
            print(boardstring)
        def terminal_test(self):
            #returns -2 if the maximum depth or search time is exceeded, 0 if the game isn't over, or 1/-1 for which player won
            #its the same code as evaluate but without the score adding
            
            nonlocal maxDepth, timeAmount
            #no need to check if the game is won if depth is 0 because then it wouldn't be called
            if self.depth == 0:
                return 0
            allowed = [self.mark, 3 - self.mark]
            for turn in allowed:
                for row in range(self.rows - 1, -1, -1):
                    for col in range(self.columns):
                        #vertical
                        if row < self.rows - (self.inarow - 1):
                            consistency = 0
                            for inc in range(self.inarow):
                                if self.board[row + inc][col] == turn:
                                    consistency += 1
                            if consistency == self.inarow:
                                return 1 if turn == self.mark else -1
                        #horizontal
                        if col < self.columns - (self.inarow - 1):
                            consistency = 0
                            for inc in range(self.inarow):
                                if self.board[row][col + inc] == turn:
                                    consistency += 1
                            if consistency == self.inarow:
                                return 1 if turn == self.mark else -1
                        #diagonal 1
                        if row < self.rows - (self.inarow - 1) and col < self.columns - (self.inarow - 1):
                            consistency = 0
                            for inc in range(self.inarow):
                                if self.board[row + inc][col + inc] == turn:
                                    consistency += 1
                            if consistency == self.inarow:
                                return 1 if turn == self.mark else -1
                        #diagonal 2
                        if row > self.inarow - 2 and col < self.columns - (self.inarow - 1):
                            consistency = 0
                            for inc in range(self.inarow):
                                if self.board[row - inc][col + inc] == turn:
                                    consistency += 1
                            if consistency == self.inarow:
                                return 1 if turn == self.mark else -1
            #if depth or time is exceeded, return -2
            if time.time() - start_time > timeAmount or self.depth == maxDepth:
                return -2
            return 0
    
    def minimize(state, alpha, beta, useAB):     
        #minimax with useAB determining whether to use alpha-beta pruning
        nonlocal depth, nodesExpanded
        depth = state.depth if state.depth > depth else depth
        termScore = state.terminal_test()
        if termScore == -1:
            return (None, -inf)
        elif termScore == 1:
            return (None, inf)
        elif termScore == -2:
            return (None, state.evaluate())

        minChildMinUtility = (None, inf)
        possibleMoves = state.getMoves()
        for move in possibleMoves:
            nodesExpanded += 1
            child = state.makeMove(move, 3 - state.mark)
            maxChildMaxUtility = maximize(child, alpha, beta, useAB)
            if maxChildMaxUtility[1] < minChildMinUtility[1]:
                minChildMinUtility = (child, maxChildMaxUtility[1])
            if minChildMinUtility[1] == -inf:
                return minChildMinUtility
            if useAB and minChildMinUtility[1] <= alpha:
                break
            if useAB and minChildMinUtility[1] < beta:
                beta = minChildMinUtility[1]
        return minChildMinUtility

    def maximize(state, alpha, beta, useAB):
        #minimax with useAB determining whether to use alpha-beta pruning
        nonlocal depth, nodesExpanded
        depth = state.depth if state.depth > depth else depth
        termScore = state.terminal_test()
        if termScore == -1:
            return (None, -inf)
        elif termScore == 1:
            return (None, inf)
        elif termScore == -2:
            return (None, state.evaluate())
        
        maxChildMaxUtility = (None, -inf)
        possibleMoves = state.getMoves()
        for move in possibleMoves:
            nodesExpanded += 1
            child = state.makeMove(move, state.mark)
            minChildMinUtility = minimize(child, alpha, beta, useAB)
            if minChildMinUtility[1] > maxChildMaxUtility[1]:
                maxChildMaxUtility = (child, minChildMinUtility[1])
            if maxChildMaxUtility[1] == inf:
                return maxChildMaxUtility
            if useAB and maxChildMaxUtility[1] >= beta:
                break
            if useAB and maxChildMaxUtility[1] > alpha:
                alpha = maxChildMaxUtility[1]
        return maxChildMaxUtility
    #create board, and check if there is only one move to make, and make that move if there is (no need to search the tree)
    currentBoard = Connect(boardarray, columns, rows, mark, inarow)
    moves = currentBoard.getMoves()
    if len(moves) == 0:
        return None
    if len(moves) == 1:
        return moves[0]
    
    #iterative deepening with the minimax search
    #starts at depth 1 and continues increasing depth until time limit is reached and makes a conclusion on the best move to make
    childUtility = (None, 0)
    prevChildUtility = (None, 0)
    utilities = []
    #while loop keeps expanding depth until the time limit is exceeded or a guaranteed win is calculated
    while time.time() - start_time < timeAmount and childUtility[1] != inf:
        prevChildUtility = childUtility
        childUtility = maximize(currentBoard, -inf, inf, True)
        utilities.append(childUtility[1])
        maxDepth += 1
    #print(str(utilities) + " " + str(time.time() - start_time)[0:4] + " " + str(nodesExpanded) + " " + str(maxDepth) + " " + str(mark))
    #code to check possible errors before selecting the move
    if time.time() - start_time < timeAmount and childUtility[1] == inf:
        return childUtility[0].indexNum % childUtility[0].columns
    if prevChildUtility[0] == None:
        if childUtility[0] == None:
            #print(str(maxDepth) + "error found")
            return moves[random.randint(0, len(moves) - 1)] if moves else None
        return childUtility[0].indexNum % childUtility[0].columns
    return prevChildUtility[0].indexNum % prevChildUtility[0].columns
