import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
from collections import defaultdict


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class KVWriter(object):

    def writekvs(self, kvs):
        raise NotImplementedError
    

