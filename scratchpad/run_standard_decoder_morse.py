#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from brewPipe.data.morse import MorseData
from brewPipe.data import BrewPipeDataFrame


__author__ = 'Dominik Meyer <meyerd@mytum.de>'


if __name__ == '__main__':
    morse_data = MorseData(data_directory=os.path.join("data", "morse"), intermediate_directory="intermediates/",
                           data_source="audio_fixed")

    t = morse_data.training()
    print t.data.shape[0]
