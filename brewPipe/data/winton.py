#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
from ..pipelineState import PipelineStateInterface

__author__ = 'Martin Kiechle <martin.kiechle@gmail.com>'


class WintonStockData(PipelineStateInterface):
    """
    Load the winton stock market challenge data
    into the pandas dataframe.
    """
    def __init__(self, data_directory="data",
                 intermediate_directory="intermediates",
                 data_source="training"):
        """
        Initialize the WintonStockData source.
        :param data_directory: Directory, where train.csv and
            test.csv reside.
        :param intermediate_directory: Directory, where the
            intermediate pandas dataframe should be persisted
            to.
        :param data_source: Can be either 'train' or
            'test' to load from either train.csv or test.csv.
        """
        super(WintonStockData, self).__init__()

        self._data_directory = data_directory
        self._intermediate_directory = intermediate_directory
        if not os.path.isdir(self._intermediate_directory):
            os.makedirs(self._intermediate_directory)
        if not (data_source == 'train' or data_source == 'test'):
            raise RuntimeError("incorrect data_source given: %s" % (data_source))
        self._data_source = data_source
        self._train_df = None
        self._test_df = None
        self._dfptr = None

    def _build_df_name(self, name):
        return "df##" + name

    def _persist_df(self, dfptr, dfpath):
        dfptr.to_hdf(dfpath, 'w', complevel=9,
                     complib='bzip2')
        return True

    def _load_df(self, dfptr, dfpath):
        dfptr = pd.read_hdf(dfpath)
        return True

    def _check_and_load_df(self, dfptr, name):
        dfpath = self.get(self._build_df_name(name))
        if not dfpath:
            dfpath = os.path.join(self._intermediate_directory, 'winton_train.hdf5')
            self.put(self._build_df_name(name), dfpath)
        return self._load_df(dfptr, dfpath)

    def _load_csv_if_no_df(self):
        """
        Set self._dfptr to the corresponding dfptr for
        pandas.DataFrame and load data if there is no
        persisted data. Otherwise load that.
        """
        if self._data_source == 'train':
            filename = os.path.join(self._data_directory, 'train.csv')
            self._dfptr = self._train_df
        else:
            filename = os.path.join(self._data_directory, 'test.csv')
            self._dfptr = self._test_df

        if not self._check_and_load_df(self._dfptr, self._data_source):
            self._dfptr = pd.read_csv(filename, sep=",")
            self._persist_df(self._dfptr, self.get(self._build_df_name(self._data_source)))

    def _fail_if_testmode(self):
        if self._data_source == 'test':
            raise RuntimeError("This data is not available if data_source is 'test'")

    def features(self):
        """
        :return: external features
        """
        self._load_csv_if_no_df()
        return self._dfptr.ix[:, 1:26]

    def intraday_2_120(self):
        """
        :return: return day D, minutes 2 - 120
        """
        self._load_csv_if_no_df()
        return self._dfptr.ix[:, 28:147]

    def intraday_120_180(self):
        """
        :return: return day D, minutes 121 - 180
        """
        self._fail_if_testmode()
        self._load_csv_if_no_df()
        return self._dfptr.ix[:, 147:207]

    def returns_last_days(self):
        """
        :return: return day D-2 and day D-1
        """
        self._load_csv_if_no_df()
        return self._dfptr.ix[:, 26:27]

    def returns_next_days(self):
        """
        :return: return day D+1 and day D+2
        """
        self._fail_if_testmode()
        self._load_csv_if_no_df()
        return self._dfptr.ix[:, 207:208]

    def weights(self):
        """
        :return: weights for scoring
        """
        self._fail_if_testmode()
        self._load_csv_if_no_df()
        return self._dfptr.ix[:, 209:210]