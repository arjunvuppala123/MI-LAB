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


Tensor = mymodule.Tensor

a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
b = Tensor(np.array([[3.0, 2.0], [1.0, 5.0]]), requires_grad=False)
c = Tensor(np.array([[3.2, 4.5], [6.1, 4.2]]))
z = np.array([[0.0, 0.0], [0.0, 0.0]])
sans = a+b
sans2 = a+a
mulans = a@b
mulans2 = (a+b)@c
sgrad = np.array([[1.0, 1.0], [1.0, 1.0]])
sgrad2 = np.array([[2.0, 2.0], [2.0, 2.0]])
mulgrad = np.array([[5.0, 6.0], [5.0, 6.0]])
mulgrad2 = np.array([[4.0, 4.0], [6.0, 6.0]])
mulgrad3 = np.array([[7.7, 10.29], [7.7, 10.29]])
mulgrad4 = np.array([[8.0, 8.0], [13.0, 13.0]])


def test_case():

    try:
        sans.backward()
        np.testing.assert_array_almost_equal(a.grad, sgrad, decimal=2)
        print("Test Case 1 for the function Add Grad PASSED")
    except:
        print("Test Case 1 for the function Add Grad FAILED")

    try:
        np.testing.assert_array_almost_equal(b.grad, z, decimal=2)
        print("Test Case 2 for the function Add Grad PASSED")
    except:
        print("Test Case 2 for the function Add Grad FAILED")

    a.zero_grad()
    b.zero_grad()

    try:
        sans2.backward()
        np.testing.assert_array_almost_equal(a.grad, sgrad2, decimal=2)
        print("Test Case 3 for the function Add Grad PASSED")
    except:
        print("Test Case 3 for the function Add Grad FAILED")

    a.zero_grad()
    b.zero_grad()

    try:
        mulans.backward()
        np.testing.assert_array_almost_equal(a.grad, mulgrad, decimal=2)
        print("Test Case 4 for the function Matmul Grad PASSED")
    except:
        print("Test Case 4 for the function Matmul Grad FAILED")

    try:
        np.testing.assert_array_almost_equal(b.grad, z, decimal=2)
        print("Test Case 5 for the function Matmul Grad PASSED")
    except:
        print("Test Case 5 for the function Matmul Grad FAILED")

    a.zero_grad()
    b.zero_grad()
    b.requires_grad = True

    try:
        mulans.backward()
        np.testing.assert_array_almost_equal(b.grad, mulgrad2, decimal=2)
        print("Test Case 6 for the function Matmul Grad PASSED")
    except:
        print("Test Case 6 for the function Matmul Grad FAILED")

    a.zero_grad()
    b.zero_grad()
    c.zero_grad()

    try:
        mulans2.backward()
        np.testing.assert_array_almost_equal(a.grad, mulgrad3, decimal=2)
        np.testing.assert_array_almost_equal(b.grad, mulgrad3, decimal=2)
        print("Test Case 7 for the function Matmul and add Grad PASSED")
    except:
        print("Test Case 7 for the function Matmul and add Grad FAILED")

    try:
        np.testing.assert_array_almost_equal(c.grad, mulgrad4, decimal=2)
        print("Test Case 8 for the function Matmul and add Grad PASSED")
    except:
        print("Test Case 8 for the function Matmul and add Grad FAILED")


if __name__ == "__main__":
    test_case()
