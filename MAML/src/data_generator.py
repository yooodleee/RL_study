
"""Code for loadding data."""
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images

FLAGS = flags.FLAGS


class DataGenerator(object):
    """Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid
    function.
    """

    def __init__(
            self,
            num_samples_per_class,
            batch_size,
            config=()
    ):
        """
        Args
        ----------------
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)

        """

        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1    # by default 1 (only relevant for classification prob)

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        
        elif 'omniglot' in FLAGS.datasource:
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (28, 28))
            self.dim_input = np.prod(self.img_size)
            self.dim_output = self.num_classes
            
            # data that is pre-resized using PIL with lanczos filter
            data_folder = config.get('data_folder', './data/omniglot_resized')

            character_folders = [
                os.path.join(data_folder, family, character)
                for family in os.listdir(data_folder)
                if os.path.isdir(os.path.join(data_folder, family))
                for character in os.listdir(os.path.join(data_folder, family))
            ]

            random.seed(1)
            random.shuffle(character_folders)
            num_val = 100
            num_train = config.get('num_train', 1200) - num_val
            self.metatrain_character_folders = character_folders[:num_train]

            if FLAGS.test_set:
                self.metaval_character_folders = character_folders[num_train + num_val:]
            else:
                self.metaval_character_folders = character_folders[num_train:num_train + num_val]
            self.rotations = config.get('rotations', [0, 90, 180, 270])
        
        elif FLAGS.datasource == 'miniimagenet':
            self.num_classes = config.get('num_classes', FLAGS.num_classes)
            self.img_size = config.get('img_size', (84, 84))
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output = self.num_classes
            metatrain_folder = config.get('metatrain_folder', './data/minImagenet/train')

            if FLAGS.test_set:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/test')
            else:
                metaval_folder = config.get('metaval_folder', './data/miniImagenet/val')
            
            metatrain_folder = [
                os.path.join(metatrain_folder, label)
                for label in os.listdir(metatrain_folder)
                if os.path.isdir(os.path.join(metatrain_folder, label))
            ]
            metaval_folder = [
                os.path.join(metaval_folder, label)
                for label in os.listdir(metaval_folder)
                if os.path.isdir(os.path.join(metaval_folder, label))
            ]

            self.metatrain_character_folders = metatrain_folder
            self.metaval_character_folders = metaval_folder
            self.rotations = config.get('rotations', [0])
        
        else:
            raise ValueError('Unrecognized data source')
    

    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # num of tasks, not num of meta-iterations. (devide by metabatch size to measure)
            num_total_batches = 200000
        else:
            folders = self.metaval_character_folders
            num_total_batches = 600
        
        
        # make list of files
        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)