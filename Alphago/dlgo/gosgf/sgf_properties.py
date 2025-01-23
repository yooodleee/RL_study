"""
Interpret SGF property values.

This is intended for use with SGF FF[4]; see http://www.red-bean.com/sgf/

This supports all general properties and Go-specific properties, but not
    properties for other games. Point, Move and Stone values are interpreted
    and Go points.

Adapted from gomill by Matthew Woodcraft, https://github.com/mattheww/gomill
"""

from __future__ import absolute_import
import codecs 
from math import isinf, isnan

import six

from dlgo.gosgf import sgf_grammer
from six.moves import range



# In Python2, indexing a str gives one-character strings.
# In Python2, indexing a bytes gives ints.
if six.PY2:
    _bytestring_ord = ord
else:
    def identity(x):
        return x
    _bytestring_ord = identity


def normalize_charset_name(s):
    """
    Convert an encoding name to the form implied in the SGF spec.

    In particular, normailize to 'ISO-8859-1' and 'UTF-8'.

    Raises LookupError if the encoding name isn't known to Python.
    """
    if not isinstance(s, six.text_type):
        s = s.decode('ascii')
    return (
        codecs.lookup(s).name.replace("_", "-").upper()
        .replace("ISO8859", "ISO-8859")
    )


def interpret_go_point(s, size):
    """
    Convert a raw SGF Go Point, Move, or Stone value to coordinates.

    s    -- 8-bit string
    size -- board size (int)

    Returns a pair (row, col), or None for a pass.

    Raises ValueError if the string is malformed or the coordinates
        are out of range.

    Only supports board sizes up to 26.

    The returned coordinates are in the GTP coordinate system
        (as in the rest of gomill), where (0, 0) is the lower left.
    """
    if s == b"" or (s == b"tt" and size <= 19):
        return None
    # May propagate ValueError
    col_s, row_s = s
    col = _bytestring_ord(col_s) - 97   # 97 == ord("a")
    row = size - _bytestring_ord(row_s) + 96
    if not ((0 <= col < size) and (0 <= row < size)):
        raise ValueError
    return row, col


