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


