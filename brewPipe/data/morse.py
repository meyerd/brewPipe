#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from ..pipelineState import PipelineStateInterface
from ..data import BrewPipeDataFrame

__author__ = 'Dominik Meyer <meyerd@mytum.de>'


class MorseData(PipelineStateInterface):
    """
    Load the morse code learning challenge
    data into pandas dataframe.
    """
    def __init__(self, data_directory=os.path.join("data", "morse"),
                 intermediate_directory="intermediates",
                 data_source="audio_fixed"):
        """
        Initialize the MorseData source.
        :param data_directory: Directory, where train.csv and
            test.csv reside.
        :param intermediate_directory: Directory, where the
            intermediate pandas dataframe should be persisted
            to.
        :param data_source: Can be only 'audio_fixed' for now
            to load .wav files from the 'audio_fixed' directory.
        """
        super(MorseData, self).__init__()

        self._data_directory = data_directory
        self._intermediate_directory = intermediate_directory
        if not os.path.isdir(self._intermediate_directory):
            os.makedirs(self._intermediate_directory)
        if not (data_source == 'audio_fixed'):
            raise RuntimeError("incorrect data_source given: %s" % (data_source))
        self._data_source = data_source
        self._train_df = None
        self._dfptr = None

    def _build_df_name(self, name):
        return "df##" + name

    def _persist_df(self, dfptr, dfpath):
        dfptr.to_hdf(dfpath, 'w', complevel=9,
                     complib='bzip2')
        return True

    def _load_df(self, dfpath):
        try:
            dfptr = pd.read_hdf(dfpath)
        except IOError:
            return None
        return dfptr

    def _check_and_load_df(self, name):
        dfpath = self.get(self._build_df_name(name))
        if not dfpath:
            if self._data_source == 'audio_fixed':
                dfpath = os.path.join(self._intermediate_directory, 'audio_fixed.hdf5')
            else:
                raise RuntimeError("invalid data source")
            self.put(self._build_df_name(name), dfpath)
        return self._load_df(dfpath)

    def _load_csv_if_no_df(self):
        """
        Set self._dfptr to the corresponding dfptr for
        pandas.DataFrame and load data if there is no
        persisted data. Otherwise load that.
        """
        if self._data_source == 'audio_fixed':
            datapath = os.path.join(self._data_directory, 'audio_fixed')
            self._dfptr = self._train_df
        else:
            raise RuntimeError("invalid data source")

        self._dfptr = self._check_and_load_df(self._data_source)
        if self._dfptr is None:
            #self._dfptr = pd.read_csv(filename, sep=",")
            tmpdfptr = pd.read_csv(os.path.join(self._data_directory, 'sampleSubmission.csv'))
            tmpdfptr = tmpdfptr.fillna('')
            d = {'id': [],
                 'prediction': [],
                 'samplerate': [],
                 'data': []}
            for f in os.listdir(os.path.join(self._data_directory, 'audio_fixed')):
                if f.endswith('.wav'):
                    cw_n = int(f[2:5])
                    prediction = tmpdfptr['Prediction'][cw_n-1]
                    id = tmpdfptr['ID'][cw_n-1]
                    samplerate, data = wavfile.read(os.path.join(self._data_directory,
                                                                 'audio_fixed', f))
                    d['id'].append(id)
                    d['prediction'].append(prediction)
                    d['samplerate'].append(samplerate)
                    d['data'].append(data)

            self._dfptr = pd.DataFrame(d)
            self._persist_df(self._dfptr, self.get(self._build_df_name(self._data_source)))

    @property
    def _input_hash(self):
        """
        :return: The hash of the input data. In this case
            we use the hash of the modification date of the input
            files to detect if the files have changed and have to
            be reread.
        """
        hstr = ''
        hash_over = []
        if self._data_source == 'audio_fixed':
            for f in os.listdir(os.path.join(self._data_directory, 'audio_fixed')):
                if f.endswith('.wav'):
                    hash_over.append(os.path.join(self._data_directory, 'audio_fixed', f))
            hash_over.append(os.path.join(self._data_directory, 'sampleSubmission.csv'))
        for ds in hash_over:
            filename = os.path.join(ds)
            mtime = os.path.getmtime(filename)
            hstr += str(mtime)
        return hash(hstr)

    def training(self):
        """
        :return: training data
        """
        dataname = 'morse##' + self._data_source + '##training'
        def cb(name):
            obj = self
            obj._load_csv_if_no_df()
            tmp = np.zeros(0)
            return tmp

        h = self._input_hash
        r = BrewPipeDataFrame(dataname, lazy_frame=True, hash=h, callback=cb)
        return r
