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


def is_valid_property_value(s):
    """
    Check whether 's' is a well-formed PropValue.

    s -- 8-bit string

    This accepts the same values as the tokenizer:
        any string that doesn't contain an unescaped]
        or end with an unescaped \ .
    """
    return bool(_propvalue_re.search(s))


def tokenize(s, start_position=0):
    """
    Tokenize a string containing SGF data.

    s               -- 8-bit string
    start_position  -- index into 's'

    Skips leading junk.

    Returns a list of pairs of strings (token type, contents), and also the
        index in 's' of the start of the unprocessed 'tail'.

    token types and contents:
        I -- PropIdent: upper-case letters
        V -- PropValue: raw value, without the enclosing brackets
        D -- delimiter: ';', '(', or ')'

    Stops when it has seen as many closing parens as open ones, at the end of
        the string, or when it fisrt finds something it can't tokenize.

    The first two tokens are always '(' and ';' (otherwise it won't find the
        start of the content).
    """
    result = []
    m = _find_start_re.search(s, start_position)
    if not m:
        return [], 0
    i = m.start()
    depth = 0
    while True:
        m = _tokenize_re.match(s, i)
        if not m:
            break
        group = m.lastgroup
        token = m.group(m.lastindex)
        result.append((group, token))
        i = m.end()
        if group == 'D':
            if token == b'(':
                depth += 1
            elif token == b')':
                depth -= 1
                if depth == 0:
                    break
    return result, i


class Coarse_game_tree:
    """
    An SGF GameTree.

    This is a direct representation of the SGF parse tree.
    It's 'coarse' in the sense that the objects in the tree structure represent
        node sequences, not individual nodes.

    Public attributes
        sequence -- nonempty list of property maps
        children -- list of Coarse_game_trees

    The sequence represents the nodes before the variations.
    """
    def __init__(self):
        self.sequence = []  # must be at least one node
        self.children = []  # may be empty


def _parse_sgf_game(s, start_position):
    """
    Common implementation for parse_game and parse_sgf_games.
    """
    tokens, end_position = tokenize(s, start_position)
    if not tokens:
        return None, None
    stack = []
    game_tree = None
    sequence = None
    properties = None
    index = 0
    try:
        while True:
            token_type, token = tokens[index]
            index += 1
            if token_type == 'V':
                raise ValueError("undexpected value")
            if token_type == 'D':
                if token == 'b':
                    if sequence is None:
                        raise ValueError("unexpected node")
                    properties = {}
                    sequence.append(properties)
                else:
                    if sequence is not None:
                        if not sequence:
                            raise ValueError("empty sequence")
                        game_tree.sequence = sequence
                        sequence = None
                    if token == b'(':
                        stack.append(game_tree)
                        game_tree = Coarse_game_tree()
                        sequence = []
                    else:
                        # token == b')'
                        variation = game_tree
                        game_tree = stack.pop()
                        if game_tree is None:
                            break
                        game_tree.children.append(variation)
                    properties = None
            else:
                # token_type == 'I'
                prop_ident = token
                prop_values = []
                while True:
                    token_type, token = tokens[index]
                    if token_type != 'V':
                        break
                    index += 1
                    prop_values.append(token)
                if not prop_values:
                    raise ValueError("property with no values")
                try:
                    if prop_ident in properties:
                        properties[prop_ident] += prop_values
                    else:
                        properties[prop_ident] = prop_values
                except TypeError:
                    raise ValueError("property value outside a node")
    except IndexError:
        raise ValueError("unexpected end of SGF data")
    assert index == len(tokens)
    return variation, end_position


