#!/usr/bin/env python
# -*- coding: utf-8 -*-

from state import PipelineState

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class PipelineStateInterface(object):
    """
    The actual pipeline state interface definition. If you
    implement any stage object, then inherit from this class
    and you can easily save and load pipeline parameters.
    """

    def __init__(self):
        self._ps = PipelineState()

    def put(self, descriptor, value):
        """
        Put a value in the pipeline state description.
        The descriptor can be defined by each class to identify
        sub-items. It will be pre-pended with the class name
        in order to distinguish different pipeline stages.
        """
        stage_descriptor = str(self.__class__.__name__) + \
            '##' + descriptor
        self._ps[stage_descriptor] = value

    def get(self, descriptor):
        stage_descriptor = str(self.__class__.__name__) + \
            '##' + descriptor
        return self._ps[stage_descriptor]
