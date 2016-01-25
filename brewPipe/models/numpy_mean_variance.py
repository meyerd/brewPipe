#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ..pipelineState import PipelineStateInterface
from ..data import BrewPipeDataFrame

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class RandomMeanVariance(PipelineStateInterface):
    """
    This is an implementation that records the mean and variance
    of each feature separately. Afterwards, new data can be sampled
    from normal distributions with that means and variances.
    """

    def __init__(self, input_dimension, silent=True):
        super(RandomMeanVariance, self).__init__()

        self._input_dimension = input_dimension
        self._x_data = None
        self._n_samples = 0
        self._sample_size = 0
        self._means = None
        self._variances = None
        self._silent = silent

    def set_data(self, x):
        self._x_data = x.data
        x_samples = self._x_data.shape[0]
        x_sample_size = self._x_data.shape[1]
        self._means = np.zeros(x_sample_size)
        self._variances = np.zeros(x_sample_size)
        self._n_samples = x_samples
        self._sample_size = x_sample_size

    def apply_model(self, n_samples):
        """
        Apply the model and return `n_samples` of random
        data with previously estimated parameters.
        :param n_samples: number of samples to draw
        """
        rdata = np.zeros((n_samples, self._sample_size))

        for column in xrange(self._sample_size):
            rdata[:, column] = np.random.normal(loc=self._means[column],
                                                scale=self._variances[column],
                                                size=n_samples)

        ret = BrewPipeDataFrame('RandomMeanVarianceSamples')
        ret.data = rdata
        return ret

    def run(self):
        x = self._x_data

        self._means = np.mean(x, axis=1)
        self._variances = np.std(x, axis=1)

        print "Results: "
        print " Means: ", self._means
        print " Variances: ", self._variances
