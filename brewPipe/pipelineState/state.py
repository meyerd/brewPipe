#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import os
import threading

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class Singleton(type):
    """
    Singleton base class to create singletons.
    Taken from:
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PipelineState(object):
    """
    Store and load the state of the whole pipeline to/from disk.
    This is essential in order to not pre-process the input data
    data every time the pipeline is run.

    The idea is, that at every time a specific part e.g. the data
    loader, which reads the input files and stores the data into
    pandas dataframes, finishes and persists the data to the disk,
    the location and type of data is added to the pipeline state.
    Subsequent runs then can skip the parts that are already finished.
    """
    __metaclass__ = Singleton

    def __init__(self, state_file='pipelinestate.pickle'):
        self._state_file = state_file
        self._state = self._load_state()
        self._lock = threading.Lock()

    def _load_state(self):
        if not os.path.isfile(self._state_file):
            return {}
        with open(self._state_file, 'r') as f:
            return pickle.load(f)

    def _save_state(self):
        with open(self._state_file, 'w') as f:
            pickle.dump(self._state, f)

    def __setitem__(self, stage, value):
        """
        Put a stage descriptor into the pipeline state file.
        The pickle file is written right after the element is
        set. Since PipelineState is a singleton object, we have
        to take care for parallelism and locking.
        """
        with self._lock:
            self._state[stage] = value
            self._save_state()

    def __getitem__(self, stage):
        with self._lock:
            rval = None
            try:
                rval = self._state[stage]
            except KeyError:
                rval = self.__missing__(stage)
            return rval

    @staticmethod
    def __missing__(stage):
        return {}

    def __delitem__(self, stage):
        with self._lock:
            try:
                del self._state[stage]
            except KeyError:
                pass

    def __len__(self):
        with self._lock:
            return len(self._state)

    def __contains__(self, stage):
        with self._lock:
            rval = False
            if stage in self._state:
                rval = True
            return rval
