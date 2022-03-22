def mcts_agent(observation, configuration):
    import random
    import time
    import numpy as np
    import math
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
    nodesExpanded = 0
    timeAmount = configuration.timeout - 1
    c = 1 #UCT score parameter
    global connectStates
    try:
        connectStates["x"]
    except:
        #print("reset dict")
        connectStates = dict()
        connectStates["x"] = "y"
        
    def toBoardStr(board):
        return ''.join(str(e) for row in board for e in row)
    
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
            self.totalReward = 0 #the rewards propagated from children
            self.numTrials = 0 #the number of propagated rewards from children
        def uct_score(self):
            if self.numTrials == 0:
                return math.inf
            else:
                return -self.totalReward / self.numTrials + c * math.sqrt(math.log(connectStates[self.parent].numTrials) / self.numTrials)
        def getMoves(self):
            #get all possible moves by checking if the top of the board is empty for each column
            moves = []
            if len(moves) == 0:
                for col in range(self.columns):
                    if self.board[0][col] == 0:
                        moves.append(col)
            return moves
        def getChildrenIndices(self):
            global connectStates
            #gets the 'child' of the current board created by making move in the [col] column
            childrenIndices = []
            board2 = [row[:] for row in self.board]
            for col in range(self.columns):
                for row in range(self.rows - 1,-1,-1):
                    if board2[row][col] == 0:
                        board2[row][col] = self.mark
                        newBoardStr = toBoardStr(board2)
                        if not newBoardStr in connectStates:
                            newBoard = Connect(board2, self.columns, self.rows, 3 - self.mark, self.inarow, self.depth + 1, toBoardStr(self.board), row * self.columns + col)
                            connectStates[newBoardStr] = newBoard
                        childrenIndices.append(newBoardStr)
                        board2 = [row[:] for row in self.board]
                        break
            return childrenIndices
        def display(self):
            #displays the connect grid
            boardstring = ""
            for row in range(self.rows):
                for col in range(self.columns):
                    boardstring += str(self.board[row][col])
                boardstring += "\n"
            print(boardstring)
        def tie(self):
            return not(any(mark == 0 for row in self.board for mark in row))
        def terminal_test(self):
            #returns 0 if the game isn't over/no one won, or 1/-1 for which player won
            #no need to check if the game is won if depth is 0 because then it wouldn't be called
            if self.depth == 0 or self.tie():
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
            return 0
    def default_policy_simulation(startStr):
        statesStr = [startStr]
        scoreMult = 1
        newStr = random.choice(connectStates[startStr].getChildrenIndices())
        statesStr.append(newStr)
        scoreMult *= -1
        while connectStates[newStr].terminal_test() == 0 and not connectStates[newStr].tie():
            newStr = random.choice(connectStates[newStr].getChildrenIndices())
            statesStr.append(newStr)
            scoreMult *= -1
        if connectStates[startStr].numTrials != 0:
            pass
            #connectStates[startStr].display()
            #connectStates[newStr].display()
            #print(str(connectStates[startStr].totalReward) + " " + str(connectStates[startStr].numTrials))
        return scoreMult * connectStates[newStr].terminal_test()
    def choose_UCT_child(boardStr):
        childrenStr = connectStates[boardStr].getChildrenIndices()
        children_scores = [connectStates[childStr].uct_score() for childStr in childrenStr]
        return childrenStr[children_scores.index(max(children_scores))]
    def choose_best_child(boardStr):
        childrenStr = connectStates[boardStr].getChildrenIndices()
        children_scores = [-connectStates[childStr].totalReward for childStr in childrenStr]
        return childrenStr[children_scores.index(max(children_scores))]
    def tree_run_single(boardStr):
        nonlocal nodesExpanded
        state = connectStates[boardStr]
        if state.terminal_test() != 0 or state.tie():
            backpropagate(state, state.terminal_test())
            return
        childrenStr = state.getChildrenIndices()
        children = [connectStates[childStr] for childStr in childrenStr]
        for x in range(len(children)):
            if children[x].numTrials == 0:
                nodesExpanded += 1
                expand_simulate(childrenStr[x])
                return
        tree_run_single(choose_UCT_child(boardStr))
    def backpropagate(state, score):
        state.totalReward += score
        state.numTrials += 1
        #state.display()
        #print(score)
        if state.parent is not None:
            backpropagate(connectStates[state.parent], -score)
    def expand_simulate(boardStr):
        score = simulate(boardStr)
        backpropagate(connectStates[boardStr], score)
    def simulate(boardStr):
        state = connectStates[boardStr]
        if state.terminal_test() != 0 or state.tie():
            return state.terminal_test()
        return -(default_policy_simulation(boardStr))
    #create board, and check if there is only one move to make, and make that move if there is (no need to search the tree)
    currentBoard = Connect(boardarray, columns, rows, mark, inarow)
    moves = currentBoard.getMoves()
    if len(moves) == 0:
        return None
    if len(moves) == 1:
        return moves[0]
    try:
        connectStates = connectStates
        currentBoard = connectStates[toBoardStr(currentBoard.board)]
    except:
        connectStates = dict()
        connectStates[toBoardStr(currentBoard.board)] = currentBoard
    currStr = toBoardStr(currentBoard.board)
    while time.time() - start_time < timeAmount:
        tree_run_single(currStr)
    gameStr = choose_best_child(currStr)
    #print(nodesExpanded)
    for x in range(len(gameStr)):
        if gameStr[x] != currStr[x]:
            return x % columns
