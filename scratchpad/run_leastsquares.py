#!/usr/bin/env python
# -*- coding: utf-8 -*-

from brewPipe.models.tf_leastsquares import TensorflowLeastSquares

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


if __name__ == '__main__':
    lsq = TensorflowLeastSquares(60, 60, learn_rate=0.1, batch_size=10)

    lsq.run(max_steps=1000)