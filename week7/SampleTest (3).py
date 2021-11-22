import sys
import importlib
import argparse
import numpy as np

np.random.seed(1)


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

y = np.array([1, 1, 2, 2, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 0, 2, 0])
y_pred1 = np.array([0, 0, 0, 0, 1, 1, 0, 2, 1, 0,
                   2, 1, 2, 0, 0, 2, 2, 1, 2, 0])
sample_weights1 = np.array([0.0653004, 0.02312196, 0.06222008, 0.0726062, 0.03265601,
                            0.09225839, 0.07963233, 0.01636507, 0.03234338, 0.02016581,
                            0.09547754, 0.04031077, 0.04511949, 0.04275803, 0.0686561,
                            0.0461075, 0.00019727, 0.0772279, 0.06631641, 0.02115935])

sample_weights2 = np.array([0.0206071, 0.01021368, 0.09169434, 0.03507312, 0.01725237,
                            0.04129091, 0.03164545, 0.06239069, 0.08194187, 0.09529935,
                            0.01143968, 0.03046731, 0.09011263, 0.09432673, 0.0634836,
                            0.03835849, 0.04538364, 0.06186011, 0.06910147, 0.00805747])

error1 = 0.8396728291461197
error2 = 0.0

alpha1 = -0.827897894503495
alpha2 = 10.361632917973205

updated_weights1 = np.array([0.03888443, 0.01376843, 0.03705019, 0.04323481, 0.01944567,
                             0.0549371, 0.04741866, 0.00974491, 0.10086681, 0.01200814,
                             0.05685401, 0.12571409, 0.0268673, 0.02546112, 0.04088265,
                             0.02745563, 0.00061521, 0.0459869, 0.20681588, 0.06598803])

updated_weights2 = np.array([0.0206071, 0.01021368, 0.09169434, 0.03507312, 0.01725237,
                             0.04129091, 0.03164545, 0.06239069, 0.08194187, 0.09529935,
                             0.01143968, 0.03046731, 0.09011263, 0.09432673, 0.0634836,
                             0.03835849, 0.04538364, 0.06186011, 0.06910147, 0.00805747])

pred1 = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1])

X1 = np.array([[1.14136883, -1.48104647,  3.84863206, -1.61239191],
               [0.60392219,  0.43760365,  0.58072179,  0.53858057],
               [-0.82307506,  0.47917457, -0.15974252, -1.10924546],
               [0.76056751, -0.84907666,  2.45921914, -0.96665366],
               [0.1913179, -0.05151451,  0.2804089,  0.0628512],
               [1.26320469,  0.3019505,  1.73597889,  0.60305083],
               [1.00500484,  0.30881211,  1.13852523,  0.6925198],
               [1.15596028, -1.91282324,  0.66773559,  1.01031492],
               [2.43630148, -0.57445261,  1.29796309,  2.71333714],
               [2.61882879,  0.53815572,  3.09560595,  1.65879465],
               [0.04870553, -1.34893436,  1.18622109, -1.10692641],
               [0.87854304,  0.84526745,  1.35195506,  0.38895161],
               [-1.89651336,  1.87828663, -2.42682997, -0.72335481],
               [-0.77189084, -0.6442479, -1.2024248, -0.31550686],
               [-0.00949665, -2.06392772, -2.82415712,  2.05337445],
               [-1.12580677, -1.13809428,  0.54422029, -2.41080203],
               [2.01174605, -1.20204156,  1.63693532,  1.66404837],
               [-1.60935528, -1.36705291, -1.48590159, -1.5154089],
               [-1.29895793, -1.55074715, -0.75686927, -1.65699323],
               [-0.1831263, -1.8830683,  1.25663465, -1.61123255]])

y1 = np.array([1, 0, 2, 1, 2, 0, 0, 0, 0, 0, 1, 0, 2, 1, 2, 1, 0, 1, 1, 1])

X1_test = np.array([[-1.12361632,  0.53389836, -1.21651525, -0.69616366],
                    [0.70030769, -1.85986897, -1.93267596,  2.46738513],
                    [-0.49329264, -3.13075392,  0.13054756, -1.34105094],
                    [0.94186341, -3.4507372, -1.49477894,  2.25920815],
                    [-0.89120175, -0.24610264, -1.67555578, -0.05304651],
                    [-1.40626263, -1.21849769,  0.13709837, -2.52844101],
                    [-0.36395778,  0.18057193,  0.10824789, -0.64460777],
                    [1.26208553,  0.27145237,  0.81816103,  1.36472315],
                    [-1.60872351,  1.43350397, -1.55309072, -1.05920398],
                    [-1.86673191,  2.48543687, -0.92831001, -1.84299981]])

y1_test = np.array([2, 2, 0, 2, 1, 1, 2, 0, 2, 2])

alphas1 = [0.6931471774349455, 0.5493061395840547]


model1 = mymodule.AdaBoost(2)


def assert_close(x, y):

    assert abs(x-y) < 1.5 * 1e-2


try:
    oerror1 = model1.stump_error(y, y_pred1, sample_weights1)
    assert_close(error1, oerror1)
    print('Passed Stump Error Case 1')
except Exception as e:
    print(f'Failed Stump Error Case 1 {e}')

try:
    oerror2 = model1.stump_error(y, y, sample_weights2)
    assert_close(error2, oerror2)
    print('Passed Stump Error Case 2')
except Exception as e:
    print(f'Failed Stump Error Case 2 {e}')

try:
    oalpha1 = model1.compute_alpha(error1)
    assert_close(alpha1, oalpha1)
    print('Passed Compute Alpha Case 1')
except Exception as e:
    print(f'Failed Compute Alpha Case 1 {e}')

try:
    oalpha2 = model1.compute_alpha(error2)
    assert_close(alpha2, oalpha2)
    print('Passed Compute Alpha Case 2')
except Exception as e:
    print(f'Failed Compute Alpha Case 2{e}')

try:
    oupdated_weights1 = model1.update_weights(
        y, y_pred1, sample_weights1, alpha1)
    np.testing.assert_array_almost_equal(
        updated_weights1, oupdated_weights1, decimal=2)
    print('Passed Update weights Case 1')
except Exception as e:
    print(f'Failed Update weights Case 1{e}')

try:
    oupdated_weights2 = model1.update_weights(y, y, sample_weights2, alpha2)
    np.testing.assert_array_almost_equal(
        updated_weights2, oupdated_weights2, decimal=2)
    print('Passed Update weights Case 2')
except Exception as e:
    print(f'Failed Update weights Case 2{e}')

try:
    model1 = mymodule.AdaBoost(2)
    opred1 = model1.fit(X1, y1).predict(X1_test)
    np.testing.assert_array_almost_equal(pred1, opred1, decimal=2)
    print('Passed Prediction Case 1')
except Exception as e:
    print(f'Failed Prediction Case 1{e}')

try:
    for x, y in zip(model1.alphas, alphas1):
        assert_close(x, y)
    print('Passed Compute Alpha Case 3')
except Exception as e:
    print(f'Failed Compute Alpha Case 3{e}')
