"""
Represent SGF games.

This is inteded for use with SGF FF[4]; see http://www.red-bean.com/sgf/

Adapted from gomill by Matthew Woodcraft,
    https://github.com/mattheww/gomill
"""

from __future__ import absolute_import
import datetime

import six

from . import sgf_grammer
from . import sgf_properties


__all__ = [
    'Node',
    'Sgf_game',
    'Tree_node',
]


