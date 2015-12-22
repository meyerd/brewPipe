#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from ..pipelineState import PipelineStateInterface

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class TensorflowLeastSquares(PipelineStateInterface):
    """
    This is a demo class that does least squares regression
    to figure out interfaces and integrate tensorflow.
    """

    def __init__(self, input_dimension, output_dimension, learn_rate=0.1, batch_size=1):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._learn_rate = learn_rate
        self._batch_size = batch_size

        self._result_W = np.zeros((self._output_dimension, self._input_dimension))
        self._result_b = np.zeros((self._output_dimension))

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._x = tf.placeholder(tf.float32, shape=[None, self._input_dimension])
            self._y_ = tf.placeholder(tf.float32, shape=[None, self._output_dimension])

            # linear least squares, fit for every element of the
            # output vector one line
            self._b = tf.Variable(tf.zeros_like(self._result_b))
            self._W = tf.Variable(tf.random_uniform([self._result_W.shape[0],
                                                     self._result_W.shape[1]], -1.0, 1.0))

            self._y = tf.matmul(self._W, self._x) + self._b

            self._loss = tf.reduce_mean(tf.square(self._y - self._y_))
            self._optimizer = tf.train.GradientDescentOptimizer(self._learn_rate).minimize(self._loss)

    def _generate_batch(self):
        # (num_examples, example dimension)
        return (np.zeros((self._batch_size, self._input_dimension)),
                np.zeros((self._batch_size, self._input_dimension)))

    def run(self, max_steps=1000):
        with tf.Session(graph=self._graph) as sess:
            tf.initialize_all_variables().run()

            average_loss = 0.0
            for step in xrange(max_steps):
                batch_x, batch_y = self._generate_batch()
                feed_dict = {self._x: batch_x, self._y_: batch_y}

                _, loss_val = sess.run([self._optimizer, self._loss], feed_dict=feed_dict)

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
