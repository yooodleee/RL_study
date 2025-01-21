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


class GoDataProcessor:
    def __init__(
            self,
            encoder='simple',
            data_directory='data'):
        
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory
    
    def load_go_data(
            self,
            data_type='train',
            num_samples=1000,
            use_generator=False):
        
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()

        sampler = Sampler(data_dir=self.data_dir)
        data = sampler.draw_data(data_type, num_samples)

        self.map_to_workers(data_type, data)
        if use_generator:
            generator = DataGenerator(self.data_dir, data)
            return generator
        else:
            features_and_labels = self.consolidate_games(data_type, data)
            return features_and_labels
    
    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(
            self.data_dir + '/' + zip_file_name
        )
        tar_file = zip_file_name[0:-3]
        this_tar = open(
            self.data_dir + '/' + tar_file, 'wb'
        )

        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()
        return tar_file
    
    def process_zip(
            self,
            zip_file_name,
            data_file_name,
            game_list):
        
        tar_file = self.unzip_data(zip_file_name)