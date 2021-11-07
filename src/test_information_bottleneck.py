from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from src.information_bottleneck import IB


class Test(TestCase):
    def test_IB(self):
        pXy = np.column_stack((np.arange(9, 0, -1), np.arange(1, 10)))
        pXy = pXy / np.sum(pXy)
        p0Xhat_X = np.eye(pXy.shape[0])
        beta = 1
        results = IB(pXy, beta, p0Xhat_X, verbose=True, track_loss=True)
        mi_compression = results["I(Xhat;X)"]
        mi_accuracy = results["I(Xhat;Y)"]
        pXhat_X = results["p(Xhat|X)"]
        pY_Xhat = results["p(y|Xhat)"]
        loss_vals = results["loss"]
        plt.plot(loss_vals)
        plt.show()

