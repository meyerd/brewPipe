#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from brewPipe.data.winton import WintonStockData
from brewPipe.preprocess.numpy_null import NumpyNullPreprocessor
from brewPipe.models.tf_leastsquares import TensorflowLeastSquares

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


if __name__ == '__main__':
    winton_training = WintonStockData(data_directory="data/", intermediate_directory="intermediates/",
                                      data_source="train")

    x = winton_training.intraday_2_120()
    y = winton_training.intraday_120_180()

    preproc_x = NumpyNullPreprocessor()
    preproc_y = NumpyNullPreprocessor()
    x = preproc_x.preprocess(x)
    y = preproc_y.preprocess(y)

    lsq = TensorflowLeastSquares(x.data.shape[1], y.data.shape[1],
                                 learn_rate=0.1, batch_size=10)
    lsq.set_data(x, y)
    lsq.run(max_steps=1000)

    result = lsq.apply_model(x)
    errors = np.sum(np.abs(result.data - y.data), axis=1)

    # plot
    plt.figure()
    plt.plot(np.arange(100), errors[:100])
    plt.xlabel("sample")
    plt.ylabel("error")
    plt.show()