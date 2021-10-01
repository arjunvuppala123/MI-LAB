import sys
import importlib
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--SRN', required=True)

args = parser.parse_args()
subname = args.SRN


try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(e)
    print("Rename your written program as YOUR_SRN.py and run python3.7 SampleTest.py --SRN YOUR_SRN ")
    sys.exit()

KNN = mymodule.KNN
def test_case1():
    data = np.array([[2.7810836, 2.550537003, 0],
                 [1.465489372, 2.362125076, 0],
                 [3.396561688, 4.400293529, 0],
                 [1.38807019, 1.850220317, 0],
                 [3.06407232, 3.005305973, 0],
                 [7.627531214, 2.759262235, 1],
                 [5.332441248, 2.088626775, 1],
                 [6.922596716, 1.77106367, 1],
                 [8.675418651, -0.242068655, 1],
                 [7.673756466, 3.508563011, 1]])
    X = data[:, 0:2]
    y = data[:, 2]

    dist = np.array([[0.0, 1.3290173915275787, 1.9494646655653247, 1.5591439385540549, 0.5356280721938492,
                    4.850940186986411, 2.592833759950511, 4.214227042632867, 6.522409988228337, 4.985585382449795],
                    [1.3290173915275787, 0.0, 2.80769851166859, 0.5177260009197887, 1.7231219074407058, 6.174826117844725,
                    3.876611681862114, 5.4890230596711325, 7.66582710454398, 6.313232155500879]])

    model = KNN(k_neigh=2, p=2)
    model.fit(X, y)

    kneigh_dist = np.array([[0., 0.53562807],
                            [0., 0.517726]])

    kneigh_idx = np.array([[0, 4],
                        [1, 3]], dtype=np.int64)

    sample = np.array([[2.6, 3.4], [5.2, 4.33]])

    pred = np.array([0, 0])
    try:
        np.testing.assert_array_almost_equal(
            model.find_distance(X[0:2, :]), dist, decimal=2)
        print("Test Case 1 for the function find_distance PASSED")
    except:
        print("Test Case 1 for the function find_distance FAILED")

    try:
        np.testing.assert_array_almost_equal(
            model.k_neighbours(X[0:2, :])[0], kneigh_dist, decimal=2)
        print("Test Case 2 for the function k_neighbours (distance) PASSED")
    except:
        print("Test Case 2 for the function k_neighbours (distance) FAILED")

    try:
        np.testing.assert_array_equal(
            model.k_neighbours(X[0:2, :])[1], kneigh_idx)
        print("Test Case 3 for the function k_neighbours (idx) PASSED")
    except:
        print("Test Case 3 for the function k_neighbours (idx) FAILED")

    try:
        np.testing.assert_array_equal(
            model.predict(sample), pred)
        print("Test Case 4 for the function predict PASSED")
    except:
        print("Test Case 4 for the function predict FAILED")

    try:
        assert model.evaluate(sample, np.array([0, 1])) == 50
        print("Test Case 5 for the function evaluate PASSED")
    except:
        print("Test Case 5 for the function evaluate FAILED")

def test_case2():
    data = np.array([[0.68043616, 0.39113473, 0.1165562 , 0.70722573, 0],
       [0.67329238, 0.69782966, 0.73278321, 0.78787406, 0],
       [0.56134898, 0.25358895, 0.10497708, 0.05846073, 1],
       [0.6515744 , 0.85627836, 0.44305142, 0.53280211, 0],
       [0.47014548, 0.18108572, 0.3235044 , 0.45490616, 0],
       [0.33544621, 0.51322212, 0.98769111, 0.53091437, 0],
       [0.4577167 , 0.80579291, 0.19350921, 0.46502849, 0],
       [0.25709202, 0.06937377, 0.92718944, 0.54662592, 1],
       [0.07637632, 0.3176806 , 0.74102328, 0.32849423, 1],
       [0.2334587 , 0.67725537, 0.4323325 , 0.38766629, 0]])

    X_train = data[:, :4]
    y_train = data[:, 4]
    samples = np.array([[0.41361609, 0.45603303, 0.33195254, 0.09371524, 1],
       [0.19091752, 0.07588166, 0.03198771, 0.15245555, 1],
       [0.29624916, 0.80906772, 0.35025253, 0.78940926, 0],
       [0.96729604, 0.89730852, 0.39105022, 0.37876973, 0],
       [0.52963052, 0.29303055, 0.27697515, 0.67815307, 1]])
    X_test = samples[:, :4]
    y_test = samples[:, 4]
    kneigh_dist = np.array([[0, 0.87960746, 0.91697707], [0, 0.72497042, 1.01071404]])
    kneigh_idx = np.array([[0, 4, 2], [1, 3, 0]])
    pred = np.array([0, 1, 0, 0, 0])

    model = KNN(k_neigh = 3, p = 1, weighted=True)
    model.fit(X_train, y_train)
    try:
        np.testing.assert_array_almost_equal(
            model.k_neighbours(X_train[0:2, :])[0], kneigh_dist, decimal=2)
        print("Test Case 1 for the function k_neighbours (distance) PASSED")
    except:
        print("Test Case 1 for the function k_neighbours (distance) FAILED")

    try:
        np.testing.assert_array_equal(
            model.k_neighbours(X_train[0:2, :])[1], kneigh_idx)
        print("Test Case 2 for the function k_neighbours (idx) PASSED")
    except:
        print("Test Case 2 for the function k_neighbours (idx) FAILED")

    try:
        np.testing.assert_array_equal(
            model.predict(X_test), pred)
        print("Test Case 3 for the function predict PASSED")
    except:
        print("Test Case 3 for the function predict FAILED")

    try:
        assert model.evaluate(X_test, y_test) == 60
        print("Test Case 4 for the function evaluate PASSED")
    except:
        print(model.evaluate(X_test,y_test))
        print("Test Case 4 for the function evaluate FAILED")

if __name__ == "__main__":
    print("------Dataset 1-------")
    test_case1()
    print("\n------Dataset 2-------")
    test_case2()