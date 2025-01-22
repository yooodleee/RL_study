"""
Parse and serialize SGF data.

This is intended for use with SGF FF[4]; see http://www.red-bean.com/sgf/

Nothing in this module is Go-specific.

This module is encoding-agnostic: it works with 8-bit strings in an arbitrary
    'ascii-compatible' encoding.

In the documentation below, a _property map_ is a dict mapping a PropIdent to a 
    nonempty list of raw property values.

A raw property value is an 8-bit string containing a PropValue without its
    enclosing brackets, but with blackslashes and line endings left untouched.

So a property map's keys should pass is_valid_property_identifier(), 
    and its values should pass is_valid_property_value().

Adapted from gomill by Matthew Woodcraft, https://github.com/mattheww/gomill
"""

from __future__ import absolute_import
import re
import string

import six

_propident_re = re.compile(
    r"\A[A-Z]{1, 8}\Z".encode('ascii')
)
_propvalue_re = re.compile(
    r"\A [^\\\]]* (?: \\. [^\\\]]* )* \Z".encode('ascii'),
    re.VERBOSE | re.DOTALL,
)
_find_start_re = re.compile(
    r"\(\s*;".encode('ascii')
)
_tokenize_re = re.compile(
    r"""
\s*
(?:
    \[ (?P<V> [^\\\]]* (?: \\. [^\\\]]* )* \]     # PropValue
    |
    (?P<I> [A-Z]{1, 8} )                        # PropIdent
    |
    (?P<D> [;()] )                              # delimiter
)
""".encode('ascii'), re.VERBOSE | re.DOTALL
)


def is_valid_property_identifier(s):
    """
    Check whether 's' is a well-formed PropIdent.

    s -- 8-bit string

    This accepts the same values as the tokenizer.

    Details:
        - it doesn't permit lower-case letters (these are allowed in some ancient
            SGF variants)
        - it accepts at most 8 letters (there is no limit in the spec; no standard
            property has more than 2)
    """
    return bool(_propident_re.search(s))


