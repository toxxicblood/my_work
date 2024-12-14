# **Search**:

- This is the process of an algorithm finding solutions to problems presented to it through doing actions on environments/states to achieve a desired  goal.
- The following are terms of art:
  1. **Agent**: This is the AI that acts upon a presented state.
  2. **State**: The current configuration of the environment at a point of time.
        -From the above we get this subterm:
        i. **initial state**: this is the state in which the agent begins.
        ii. **goal state**: this is the state that the agent tries to achieve.
                it can be either a solution or an other state.
                note that not all states have a goal, some may be infinite.
  3. **Actions**: This is the set of *mathematically programmable and actionablevfunctions* that can be executed on a state to get from the initial/current to the goal.*actions(s)* reurns a *list* of executable actions on s.
  4. **Transition Model**: This is a function that returns the state achieved after taking a particular action on the current state. Defined as *result(s,a)* where *s* is a state and *a* is an action.
  5. **State space**: This is a set of all possible reachable state from the initial state by any sequence of actions gotten by applying the transition model on all possible actions and states and is generally represented as a graph with nodes for states and edges for actions.
  6. **Goal test**: This is a function that checks whether a given state is the goal or not. While some problems may have one goal, others have multiple.
  7. **Path cost**: This is the cost incurred to achieve a certain goal.
            - it is used to determine whether the path used is the most optimal.
            - this is repped as a graph with a weighted edge.
  8. **Solution**: this is a sequence of actions that leads from the initial state to the goal state.
        - **Optimal solution**: this is the solution with the most optimal path cost.
  9. **Node**: this is a data structure used to keep track of the following:
        1. *state* of the node
        2. *parent node*(node that generated this node)
        3. *action*: sequence of actions used to get to this node
        4. *path cost* from initial state to current node

    [^]:Note:

      Nodes are data structures and only hold the aove info and noting more. 

  10. **Frontier**: a data sructure where all available unexplored states are stored.
 
- The following is the basic syntax these algorithms follow:

    ```python
    1. So we start with a frontier that contains the *initial* state.
    2. Start with an empty explored set(a list or dict)
    3. **Repeat**:
        If the frontier is empty:
            there is no solution.
        Else:
            *Select/remove* a node from the frontier.(node is current consideration)
            If the node contains the goal state:
                We have found a solution.
            Else:
                Add current node to the *explored set*( to avoid repeating nodes)
                Expand the node( look at all possible actions from the current node) and add the resulting nodes to the frontier(*if thet arent already in the frontier or the explored set*).(expanding is like branching)
    ```
There are various types of search algorithm that are used.

## **Uninformed search**

- These are search strategies that are implemented assuming we have no information about the problem domain.
- The following are examples of uninformed search algos:
    1. **Depth first search**: 
        - in this search algo, we explore the deepest node first.
        - The logic is implemented as a **stack** meaning *last in first out* basis.
        - The last node added to the frontier is the first node we explore.
        - The problem with depth first search is that it performs very slowly on big tasks but can be good for shallow searches.
        - It may also end up finding a logner path to the solution.
    2. **Breadth first search**:
        - In this algo we explore the shallowest node first.
        - The logic implemented is a **queue** meaning *first in first out* basis.
        - The first node added is the first explored.
        - We go in a layer by layer basis.
        - This algo can be very memory and time intensive especially if the goal is deep.
        - We may end up exploring all paths before reaching the solution.
        - Although with this approach we are more likely to find the most optimal path.

## **Informed search**

- These are search strategies tht use domain specific knowledge to solve the problem faster and more efficiently.
- Because we have context, we are better able to solve problems.
- The following are examples of informed search algos:
    1. **Greedy best-first search**:
        - This is a search algo that expands the node it thinks is closest to the goal.
        - This 'closeness' is estimated by a *heuristic function*.
        - Therefore the speed and efficiency of this AI is determined by how well the heuristic performs.
        - THe heurisic function *h(n)* takes an input state n and returns an estimate of how far it thinks n is from the goal.
        - We can use the **manhattan distance** which, by relaxing the problem counts how far the goal is through (x,y) coordinates.
        ![alt text](image.png)
        - This can however mislead because a path that seems shorter may actually be wrong.
        - This algo therefore follows the estimated shortest dist to the goal avoiding nodes that lead away.
    2. __A* search__:
        - This is a mod of the *GBFS* algo.
        - - This search algo expands the node with the lowest value of **g(n) + h(n)**
        - **g(n)**: the cost accrued to reach current node.(*how far did i have to travel to get here*)
        - **h(n)**: estimated distance from curreent node to the goal(*how far am i from the goal*)(this is the heuristic value)
        - Therefore the AI combines the distance from the goal and the cost to reach the current node to determine its weight and makes a decision choosing the node with the lowest total value in the frontier.
        - But also as with all algos, this algorithm is only as good as the heuristic it employs.
        - FOr  it to be optimal the heuristic must be:
            1. *Admissible*: never overestimates the true cost(either correct or underestimated)
            2. *Consistent*: no subsequent node should cost less than its parent node. for parent node __n__ and successor __n'__ with a step cost __c__ , __h(n) ≤ h(n')+c__
        - However this search algo is known to use quite a bit of memory so it can be further refined.
  
## **Adverserial search**

- THese are search algos that face opponents(a player 1 vs player 2 kind of situweshen.)
- Here the question *what does an intelligent decision in a two player game look like*
    
    1. **Minimax**:

        - This is an algorithm for deterministic games like tic tac toe.
        - Deterministic games are games where the players moove one after the other and only one player can win or we both draw.
        - We assign a value to each possible outcome:

            - O winning = -1(*min player*)
            - X winning = +1(*max player*)
            - Draw      =  0
        - THe max player aims to maximise the  score while the min player aims to minimise it.
        **Game**:

        1. **S₀** : Initial state
        2. **Players(s)**: takes a *state __s__* as input and returns which player's turn it is.
        3. **Actions(s)**: takes a *state __s__* as input and returns all possible moves in this state.(as a list)
        4. **Result(s,a)**: takes a *state __s__* and an *action __a__* as input and returns a new state.(result of action a on state s)(*transition model*)
        5. **Terminal(s)**: takes a *state __s__* as input and returns true if the game is over and false otherwise.
        6. **Utility(s)**: takes a *state __s__* as input and returns the utility(value/score) of this state(__-1, 0, or 1__).

        - The algorithm works by *recursively* stimulating all possible games that can take place by taking both personas untill a terminal state is reached.
        ![alt text](image-1.png)
        -Basically minimax considers all options and chooses the safest route to the goal.
        - Through running therough the recursion to the terminal state i am able to see which sequence of moves can lead to a win, loss or draw. Therefor i can infer that beginning at the current state the best move is the one that leads to the best possible outcome of a desired goal.
        - The following is an abstract representation of a minimax tree:
            ![alt text](image-2.png)
            the green arrows are the maximizer
            the red arrows are the minimizer.
        - in this method the min player considers the maximum that the max palyer can get on their next turn and chooses the node with the smallest max value.
        - The maximum player considers the minimum that the min player can get on their next turn and chooses the node with the largest min value.
        - Minimax is good but its biggest drawback is that it is very slow and memory intensive because it explores all possible moves.
  
    2. **Alpha-Beta Pruning**:
        - With minimax if i am playing as the maximizing player, i can never gurantee a higher score than the lowest possible score in my opponent's next move.
        - The lowest score in any subsequent node is hypothetically guranteed, therefore i will often go for the node(not with the highest max score) but with the highest min score of all available nodes.
        - *In alpha beta pruning, we skip some of the recursive computations that are __guaranteed__ to be suboptimal.*
        ![alt text](image-3.png)
        - When i explore the first node i keep that minimum score untill i find an other node with a higher min score.
        - Although this algo is very efficient its main drawback is that as problems grow more complex it can become inefficient to explore all nodes of even a single branch.

    3. **Depth-limited Minimax**:
        - This is like minimax on steroids. We limit the depth of each node and therefore increase the efficiency and speed of the algo.
        - Here though we introduce a new problem of knowing the value of each node without reaching a terminal state.
        - To solve this we add an *evaluation function* which estimates the expected utility of the game from a given stae.(how likely are we to win)
        - Therefore the power of our algo is only limited by the power of our evaluation function.
        - This algo has no guranteed end state.
        - THe better the evaluation function the better the AI.