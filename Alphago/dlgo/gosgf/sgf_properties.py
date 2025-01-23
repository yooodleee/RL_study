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


def serailize_go_point(move, size):
    """
    Serialize a GO Point, Move, or Stone value.

    move -- pair (row, col), or None for a pass

    Returns an 8-bit string.

    Only supports board size up to 26.

    The move coordinates are in the GTP coordinate system
        (as in the rest of gomill), where (0, 0) is the lower left.
    """
    if not 1 <= size <= 26:
        raise ValueError
    if move is None:
        # Prefer 'tt' where possible, for the sake of older code
        if size <= 19:
            return b"tt"
        else:
            return b""
    row, col = move
    if not ((0 <= col < size) and (0 <= row < size)):
        raise ValueError
    col_s = "abcdefghijklmnopqrstuvwxy"[col].encode('ascii')
    row_s = "abcdefghijklmnopqrstuvwxy"[size - row - 1].encode('ascii')
    return col_s, row_s


class _Context:
    def __init__(self, size, encoding):
        self.size = size
        self.encoding = encoding


def interpret_none(s, context=None):
    """
    Convert a raw None value to a boolean.

    That is, unconditionally returns True.
    """
    return True


def serialize_none(b, context=None):
    """
    Serialize a None value.

    Ignores its parameter.
    """
    return b""


def interpret_number(s, context=None):
    """
    Convert a raw Number value to the integer it represents.

    This is a little more lenient than the SGF spec: it permits leading
        and training spaces, and spaces between the sign and the numerals.
    """
    return int(s, 10)


def serialize_number(i, context=None):
    """
    Serialize a Number value.

    i -- integer
    """
    return ("%d" % i).encode('ascii')


def interpret_real(s, context=None):
    """
    Convert a raw Real value to the float it represents.

    This is more lenient than the SGF spec: it accepts strings accepted
        as a float by the platform libc. It rejects infinities and NaNs.
    """
    result = float(s)
    if isinf(result):
        raise ValueError("infinite")
    if isnan(result):
        raise ValueError("not a number")
    return result


def serialize_real(f, context=None):
    """
    Serialize a Real value.

    f -- real number (int or float)

    If the absolute value is too small to conveniently express as a decimal,
        returns "0" (this currently happens if abs(f) is less than 0.0001).
    """
    f = float(f)
    try:
        i = int(f)
    except OverflowError:
        # infinity
        raise ValueError
    if f == i:
        # avoid trailing '.0';
        # also avoid scientific notation for large numbers
        return str(i).encode('ascii')
    s = repr(f)
    if 'e-' in s:
        return "0".encode('ascii')
    return s.encode('ascii')


