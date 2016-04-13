#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from ..pipelineState import PipelineStateInterface
from ..data import BrewPipeDataFrame

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class TensorflowLSTMIntraday(PipelineStateInterface):
    """
    This class aims at predicting the intraday values
    of the ``winton'' dataset using the supplied values
    from 2 to 120 and by shifting by one predicting one
    by one future value.
    """

    def __init__(self, input_dimension, output_dimension, learn_rate=0.1,
                 batch_size=1, silent=True):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._hidden_size = 100
        self._num_layers = 2
        self._is_training = True
        self._keep_prob = 0.95
        self._learn_rate = learn_rate
        self._batch_size = batch_size
        self._x_data = None
        self._y_data = None
        self._n_samples = 0
        self._silent = silent

        self._data_ptr = 0

        self._lstm_cell = tf.models.rnn.rnn_cell.BasicLSTMCell(
            self._hidden_size,
            forget_bigas=1.0
        )
        if self._is_training and self._keep_prob < 1.0:
            self._lstm_cell = tf.models.rnn.rnn_cell.DropoutWrapper(
                self._lstm_cell, output_keep_prob=self._keep_prob
            )
        self._cell = tf.models.rnn.rnn_cell.MultiRNNCell([
            self._lstm_cell * self._num_layers
        ])
        self._initial_state = self._cell.zero_state(self._batch_size, tf.float32)



        self._graph = tf.Graph()
        with self._graph.as_default():
            self._x = tf.placeholder(tf.float32, shape=[None, self._input_dimension])
            self._y_ = tf.placeholder(tf.float32, shape=[None, self._output_dimension])

            # linear least squares, fit for every element of the
            # output vector one line
            self._b = tf.Variable(tf.zeros_like(self._result_b, dtype=tf.float32))
            self._W = tf.Variable(tf.random_uniform([self._result_W.shape[0],
                                                     self._result_W.shape[1]], -1.0, 1.0,
                                                    dtype=tf.float32))

            self._y = tf.matmul(self._x, self._W) + self._b

            self._loss = tf.reduce_mean(tf.square(self._y - self._y_))
            self._optimizer = tf.train.GradientDescentOptimizer(self._learn_rate).minimize(self._loss)

    def _generate_batch(self):
        if self._n_samples <= 0:
            raise RuntimeError("Data has to be set first.")

        # TODO: make this more robust and shuffle data
        if self._data_ptr >= self._n_samples:
            self._data_ptr = 0
        data_from = self._data_ptr
        data_to = self._data_ptr + self._batch_size
        self._data_ptr = data_to

        return (self._x_data[data_from:data_to, :],
                self._y_data[data_from:data_to, :])

    def set_data(self, x, y):
        self._x_data = x.data
        self._y_data = y.data
        x_samples = self._x_data.shape[0]
        y_samples = self._y_data.shape[0]
        if x_samples != y_samples:
            raise RuntimeError("There have to be the same number of samples")
        self._n_samples = x_samples

    def apply_model(self, x):
        x = x.data

        tmp = np.zeros((1,1))
        with tf.Session(graph=self._graph) as sess:
            tf.initialize_all_tables().run()

            feed_dict = {self._x: x,
                         self._W: self._result_W,
                         self._b: self._result_b}

            tmp = sess.run(self._y, feed_dict=feed_dict)

        ret = BrewPipeDataFrame('y')
        ret.data = tmp
        return ret

    def run(self, max_steps=1000):
        with tf.Session(graph=self._graph) as sess:
            tf.initialize_all_variables().run()

            average_loss = 0.0
            for step in xrange(max_steps):
                batch_x, batch_y = self._generate_batch()
                feed_dict = {self._x: batch_x, self._y_: batch_y}

                _, loss_val = sess.run([self._optimizer, self._loss], feed_dict=feed_dict)

                if not self._silent:
                    average_loss += loss_val
                    printinterval = 20
                    if step % printinterval == 0:
                        if step > 0:
                            average_loss /= printinterval
                        print "Average loss at step ", step, ": ", average_loss
                        average_loss = 0
                        print sess.run(self._W), sess.run(self._b)

            self._result_W = sess.run(self._W)
            self._result_b = sess.run(self._b)

        print "Result: "
        print self._result_W, self._result_b
