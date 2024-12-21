"""Utilities for DMALab-30."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf


LEVEL_MAPPING = collections.OrderedDict([
    ('rooms_collec_good_objects_train', 'room_collect_good_objects_test'),
    ('room_exploit_deferred_effects_train', 'rooms_exploit_deferred_effects_test'),
    ('')
])