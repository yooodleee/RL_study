from __future__ import print_function
from __future__ import absolute_import
import os
import glob
import tarfile
import gzip
import shutil
import numpy as np
import multiprocessing
import os
import sys

from keras import utils     # to_categorical

from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.data.index_processor import KGSIndex
from dlgo.data.sampling import Sampler
from dlgo.data.generator import DataGenerator
from dlgo.encoders.base import get_encoder_by_name


def worker(jobinfo):
    try:
        clazz, encoder, zip_file, data_file_name, game_list = jobinfo
        clazz(encoder=encoder).process_zip(
            zip_file,
            data_file_name,
            game_list,
        )
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child proceess.')


