import ftplib
import hashlib
import json
import os
from logging import getLogger

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from chess_zero.agent.api_chess import ChessModelAPI
from chess_zero.config import Config

# noinspection PyPep8Naming

logger = getLogger(__name__)



class ChessModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None   # type: Model
        self.digest = None
        self.api = None
    
    def get_pipes(self, num=1):
        if self.api is None:
            self.api = ChessModelAPI(self.config, self)
            self.api.start()
        return [self.api.get_pipe() for _ in range(num)]
    
    def build(self):
        mc = self.config.model
        in_x = x = Input((18, 8, 8))

        # (batch, channels, height, width)
        x = Conv2D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_first_filter_size,
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)
        
        res_out = x

        # for policy output
        x = Conv2D(
            filters=2,
            kernel_size=1,
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(
            self.config.n_labels,
            kernel_regularizer=l2(mc.l2_reg),
            activation="softmax",
            name="policy_out")(x)