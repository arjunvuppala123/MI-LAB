import sys
import importlib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--SRN', required=True)

args = parser.parse_args()
subname = args.SRN


try:
   mymodule = importlib.import_module(subname)
except:
    print("Rename your written program as YOUR_SRN.py and run python3.7 SampleTest.py --SRN YOUR_SRN ")
    sys.exit()



def testcase(mymodule):
    cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1],
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
    start = 1
    goals = [6, 7, 10]
    try:
        if mymodule.A_star_Traversal(cost, heuristic, start, goals)==[1,5,4,7]:
            print("Test Case 1 for A* Traversal PASSED")
        else:
            print("Test Case 1 for A* Traversal FAILED")
    except exception as e:
        print("Test Case 1 for A* Traversal FAILED due to ",e)


    try:
        if mymodule.DFS_Traversal(cost,start, goals)==[1, 2, 3, 4, 7]:
            print("Test Case 2 for DFS Traversal PASSED")
        else:
            print("Test Case 2 for DFS Traversal FAILED")
    except exception as e:
        print("Test Case 2 for DFS Traversal FAILED due to ",e)         

testcase(mymodule)