import sys
import importlib
import argparse
import pandas as pd


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

data = pd.read_csv('test.csv')
X_test = data.iloc[:, 0:-1]
y_test = data.iloc[:, -1]

try:
    model = mymodule.SVM('train.csv').solve()
    print(f'Accuracy: {model.score(X_test, y_test)*100:.2f}%')
except Exception as e:
    print(f'Failed {e}')
