"""
You can create any other helper funtions.
Do not modify the given functions
"""


import heapq
from collections import deque


def goalTest(state, goals):
    return state in goals



def getNeighbours(adjList):
    neighbourList = []
    for index, node in enumerate(adjList[1::], start=1):
        if node > 0:
            neighbourList.append(index)

    return neighbourList[::-1]



def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """

    path = []
    n = len(cost[0])

    node = [heuristic[start_point], [start_point], start_point, 0]

    f = []
    f.append(node)
    exp = set()

    while (True):
        if (len(f) == 0):
            return []

        popped_node = heapq.heappop(f)
        if (popped_node[2] in goals):
            return popped_node[1]

        exp.add(popped_node[2])

        for i in range(1, n):
            if ((cost[popped_node[2]][i] != -1) and (cost[popped_node[2]][i] != 0)):
                flag = False  
                for j in f:
                    if(j[2] == i): 
                        flag = True  
                        break


                if ((flag is False) and (i not in exp)):
                    temp = popped_node[1] + list((i,))
                    heapq.heappush(f, list(
                        (popped_node[3] + cost[popped_node[2]][i] + heuristic[i], temp, i, popped_node[3] + cost[popped_node[2]][i])))


                elif (flag is True):

                    for j in f:
                        if j[2] == i:
                            if (j[3] >= popped_node[3] + cost[popped_node[2]][i]):
                                if (j[3] == popped_node[3] + cost[popped_node[2]][i]) and (j[1] <= popped_node[1] + list((i,))):
                                    break
                                j[0] = popped_node[3] + cost[popped_node[2]][i] + heuristic[i]
                                j[3] = popped_node[3] + cost[popped_node[2]][i]
                                j[1] = popped_node[1] + list((i,))
                                heapq.heapify(f)
                            break

    return path
# TODO




def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """


    path = []
    # TODO

    stack = deque()

    stack.append({
        "node": start_point,
        "path": [start_point]
    })

    expSet = set()

    while (stack):
        popNode = stack.pop()

        if popNode["node"] in expSet:
            continue

        expSet.add(popNode["node"])

        if goalTest(popNode["node"], goals) is True:
            return popNode["path"]
        popNodeNeighs = getNeighbours(cost[popNode["node"]])

        for popNodeNeigh in popNodeNeighs:
            if popNodeNeigh not in expSet:
                popNodeNeighRec = {
                    "node": popNodeNeigh,
                    "path": popNode["path"] + [popNodeNeigh]
                }
                stack.append(popNodeNeighRec)

    return path
    
