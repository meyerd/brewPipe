#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from brewPipe.data.winton import WintonStockData
from brewPipe.data import BrewPipeDataFrame
from brewPipe.output.winton import WintonStockDataOutput
from brewPipe.models.numpy_mean_variance import RandomMeanVariance


__author__ = 'Dominik Meyer <meyerd@mytum.de>'


if __name__ == '__main__':
    winton_training = WintonStockData(data_directory=os.path.join("data", "winton"),
                                      intermediate_directory="intermediates/",
                                      data_source="train")

    x1 = winton_training.intraday_120_180()
    x2 = winton_training.returns_next_days()

    n_samples = x1.data.shape[0]
    sample_size = x1.data.shape[1] + x2.data.shape[1]
    whole_x = np.hstack([x1.data, x2.data])

    w_x_df = BrewPipeDataFrame("Whole X Input")
    w_x_df.data = whole_x

    rmv = RandomMeanVariance(sample_size)
    rmv.set_data(w_x_df)
    rmv.run()

    # apply model to generate 121 - 180 data and next day data
    y = rmv.apply_model(n_samples=120000)

    out = WintonStockDataOutput(output_file="submission_winton_rand.csv",
                                overwrite=True)
    out.set_data(y)
    out.write()