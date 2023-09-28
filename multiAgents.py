# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        # legal moves are; ['North', 'South', 'East', 'West', 'Stop']
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print(f'successorGameState: {successorGameState}')
        # print(f'newPos: {newPos}')
        # print(f'newFood: {newFood}')
        # print(f'newGhostStates: {newGhostStates}')
        # print(f'newScaredTimes: {newScaredTimes}')

        "*** YOUR CODE HERE ***"
        # get distance to each food and take minimum of these
        # return negative as the score --> inverse proportionality
        # initialize dist with 0 and min with inf 
        curr_pos = list(successorGameState.getPacmanPosition())
        dist = 0
        min = float('inf')

        curr_food = currentGameState.getFood().asList()
        # returns list of current food positions

        # loop through all food positions and calculate manhattan distance from curr_pos to each pieve of food
        # keep track of the smallest of these distances to prioritize smallest (closest) value
        for i in curr_food:
            # find min distance to each food
            # return minimum of these
            dist = manhattanDistance(i, curr_pos)
            if dist < min:
                min = dist
        # reciprocal of distance
        min = -min

        # stay away from ghosts
        # loops through new ghost states. If a ghost is in pacmans position, return -inf to indicate avoid
        for state in newGhostStates:
            if state.getPosition() == tuple(curr_pos):
                return float('-inf')

        # keep pac man moving to limit time and to keep away from ghosts
        if action == 'Stop':
            return float('-inf')

        return min

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # action, score for pacman (agentIndex = 0)
        action, score = self.minimax(0,0, gameState)
        # return the action decided from minimax algorithm
        return action
    
    def minimax(self, agentIndex, depth, gameState):
        # minimax DFS recursive algorithm to return the best score for an agent
        # pacman agentIndex = 0 --> max player so return max score from successor states
        # ghosts agentIndex > 0 --> min player so return min score from successor states
        # base case --> no successor states or depth is equal to max depth
        
        # param agentIndex: index of current agent (pacman == 0 and ghosts != 0)
        # param depth: current depth
        # param fameState: current game state
        # return: action, score


        # print('\n')
        # print(f'agentIndex, depth: {agentIndex}, {depth}')

        # Base cases
        # if all agents have finished playing their turn, go to next level
        if agentIndex >= gameState.getNumAgents():
            # print('agentIndex >= numAgents')
            agentIndex = 0
            depth += 1
        # return score if max depth is reached
        if depth == self.depth:
            # print('max depth reached')
            return None, self.evaluationFunction(gameState)
        
        # keep track of best_score and best_action
        best_score = None
        best_action = None
        
        # pacmans turn
        if agentIndex == 0:
            # for each legal pacman action
            for action in gameState.getLegalActions(agentIndex):
                # get successor state
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                # get minimax score of successor state
                # increase agent by one to indicate new ghost and pass to minimax
                # pass successorGameState to minimax
                __, score = self.minimax((agentIndex + 1), depth, successorGameState)
                # if better than current best score (max for pacman), then update best variables
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
                # print(f"pacman best_score, best_action: {best_score}, {best_action}")
        # otherwise its ghosts turn (agentIndex != 0)
        else: 
            # for each legal ghost action
            # pretty much same thing as for pacman just min instead of max
            for action in gameState.getLegalActions(agentIndex):
                # get successor state
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                # get minimax score of successor state
                # increase agent by one to indicate next ghost or base case of no agents left
                # pass successorGameState to minimax
                __, score = self.minimax((agentIndex + 1), depth, successorGameState)
                # if better than current best score (min for ghosts), then update best variables
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
                # print(f"ghost best_score, best_action: {best_score}, {best_action}")
        # print(f'agentIndec = {agentIndex}')
        # print(f"best_score, best_action: {best_score}, {best_action}")

        # if leaf ie. no successor states (no legal actions?) best_score is None
        if best_score is None:
            # print(self.evaluationFunction(gameState))
            return None, self.evaluationFunction(gameState)
        return best_action, best_score

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, score = self.alpha_beta(0,0, gameState, float('-inf'), float('inf'))
        return action
    
    def alpha_beta(self, agentIndex, depth, gameState, alpha, beta):
        # alphabeta algorithm for best score with pruning
        # pacman agentIndex = 0 --> max player so return max score from successor states
        # ghosts agentIndex > 0 --> min player so return min score from successor states
        # base case --> no successor states or depth is equal to max depth
        # if alpha > beta, prune the search tree (because alpha is lower bound and beta is upper bound?)

        # param agentIndex: index of current agent (pacman == 0 and ghosts != 0)
        # param depth: current depth
        # param gameState: current game state
        # param alpha: alpha value of parent
        # param beta: beta value of parent
        # return: action, score

        ####### Essentially the same as minimax above with some tweaks ######


        # Base cases
        # if all agents have finished playing their turn, go to next level
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth += 1
        # return score if max depth is reached
        if depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        # keep track of best_score and best_action
        best_score = None
        best_action = None
        
        # pacmans turn
        if agentIndex == 0:
            # for each legal pacman action
            for action in gameState.getLegalActions(agentIndex):
                # get successor state
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                # get minimax score of successor state
                # increase agent by one to indicate new ghost and pass to minimax
                # pass successorGameState to minimax
                __, score = self.alpha_beta((agentIndex + 1), depth, successorGameState, alpha, beta)
                # if better than current best score (max for pacman), then update best variables
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
                # update alpha value for pacman
                alpha = max(alpha, score)
                # prune if alpha > beta 
                if alpha > beta:
                    break
        # otherwise its ghosts turn (agentIndex != 0)
        else: 
            # for each legal ghost action
            # pretty much same thing as for pacman just min instead of max
            for action in gameState.getLegalActions(agentIndex):
                # get successor state
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                # get minimax score of successor state
                # increase agent by one to indicate next ghost or base case of no agents left
                # pass successorGameState to minimax
                __, score = self.alpha_beta((agentIndex + 1), depth, successorGameState, alpha, beta)
                # if better than current best score (min for ghosts), then update best variables
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
                # update beta value for ghost
                beta = min(beta, score)
                # prune if beta < alpha
                if beta < alpha:
                    break

        # if leaf ie. no successor states (no legal actions?) best_score is None
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score






class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, score = self.expectimax(0,0,gameState)
        return action
    
    def expectimax(self, agentIndex, depth, gameState):
        # expectimax recursive algorithm to return the best score for an agent
        # pacman agentIndex = 0 --> max player so return max score from successor states
        # ghosts agentIndex > 0 --> min player so return min score from successor states
        # base case --> no successor states or depth is equal to max depth
        
        # param agentIndex: index of current agent (pacman == 0 and ghosts != 0)
        # param depth: current depth
        # param fameState: current game state
        # return: action, score


        # Base cases
        # if all agents have finished playing their turn, go to next level
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth += 1
        # return score if max depth is reached
        if depth == self.depth:
            return None, self.evaluationFunction(gameState)
        
        # keep track of best_score and best_action
        best_score = None
        best_action = None
        
        # pacmans turn
        if agentIndex == 0:
            # for each legal pacman action
            for action in gameState.getLegalActions(agentIndex):
                # get successor state
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                # get minimax score of successor state
                # increase agent by one to indicate new ghost and pass to minimax
                # pass successorGameState to minimax
                __, score = self.expectimax((agentIndex + 1), depth, successorGameState)
                # if better than current best score (max for pacman), then update best variables
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
                
        # otherwise its ghosts turn (agentIndex != 0)
        else: 
            
            # first calculate the probabilityof each action, assuming each action is equaly likely
            g_actions = gameState.getLegalActions(agentIndex)
            # account for no actions ie. len(g_actions = 0)
            if len(g_actions) != 0:
                prob = 1.0 / len(g_actions) # uniform probability for each action

            # loop over all ghosts legal actions
            for action in g_actions:
                # get successor state
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                # get expectimax score of successor state
                # increase agent by one to indicate next ghost or base case of no agents left
                # pass successorGameState to minimax
                __, score = self.expectimax((agentIndex + 1), depth, successorGameState)

                # update best_score as an expected score (weighted averageof scores by prob)
                if best_score is None:
                    best_score = 0.0
                best_score += prob * score
                best_action = action

        # if leaf ie. no successor states (no legal actions?) best_score is None
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

