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


def interpret_double(s, context=None):
    """
    Convert a raw Double value to an integer.

    Returns 1 or 2 (unknown values are treated as 1).
    """
    if s.strip() == b"2":
        return s
    else:
        return 1


def serialize_double(i, context=None):
    """
    Serialize a Double value.

    i -- integer (1 or 2)

    (unknown values are treated as 1)
    """
    if i == 2:
        return "2"
    return "1"


def interpret_color(s, context=None):
    """
    Convert a raw Color value to a gomill color.

    Returns 'b' or 'w'.
    """
    color = s.decode('ascii').lower()
    if color not in ('b', 'w'):
        raise ValueError
    return color


def serialize_color(color, context=None):
    """
    Serialize a Color value.

    color -- 'b' or 'w'
    """
    if color not in ('b', 'w'):
        raise ValueError
    return color.upper().encode('ascii')


def _transcode(s, encoding):
    """
    Common implementation for interpret_text and interpret_simpletext.
    """
    # If encoding is UTF-8, we don't need to transcode, but we 
    # still want to report an error if it's not properly encoded.
    u = s.decode(encoding)
    if encoding == "UTF-8":
        return s
    else:
        return u.encode("utf-8")


def interpret_simpletext(s, context):
    """
    Convert a raw SimpleText value to a string.

    See sgf_grammer.simpletext_value() for details.

    s -- raw value

    Returns an 8-bit utf-8 string.
    """
    return _transcode(
        sgf_grammer.simpletext_value(s),
        context.encoding,
    )


def serialize_simpletext(s, context):
    """
    Serialize a SimpleText value.

    See sgf_grammer.escape_text() for details.

    s -- 8-bit utf-8 string
    """
    if context.encoding != "UTF-8":
        s = s.decode("utf-8").encode(context.encoding)
    return sgf_grammer.escape_text(s)


def interpret_text(s, context):
    """
    Convert aa raw Text value to a string.

    See sgf_grammer.text_value() for details.

    s -- row value

    Returns an 8-bit utf-8 string.
    """
    return _transcode(
        sgf_grammer.text_value(s),
        context.encoding,
    )


def serialize_text(s, context):
    """
    Serialize a Text value.

    See sgf_grammer.escape_text() for details.

    s -- 8-bit utf-8 string
    """
    if context.encoding != "UTF-8":
        s = s.decode("utf-8").encode(context.encoding)
    return sgf_grammer.escape_text(s)


def interpret_point(s, context):
    """
    Convert a raw SGF Point or Stone value to coordinates.

    See interpret_go_point() above for details.

    Returns a pair (row, col).
    """
    result = interpret_go_point(s, context.size)
    if result is None:
        raise ValueError
    return result


def serialize_point(point, context):
    """
    Serialize a Point or Stone value.

    point -- pair (row, col)

    See serialize_go_point() above for details.
    """
    if point is None:
        raise ValueError
    return serailize_go_point(point, context.size)


def interpret_move(s, context):
    """
    Convert a raw SGF Move value to coordinates.

    See interpret_go_point() above for details.

    Returns a piar (row, col), or None for a pass.
    """
    return interpret_go_point(s, context.size)


def serailize_move(move, context):
    """
    Serialize a Move value.

    move -- pair (row, col), or None for a pass

    See serialize_go_point() above for details.
    """
    return serailize_go_point(move, context.size)


def interpret_point_list(values, context):
    """
    Convert a raw SGF list of Points to a set of coordinates.

    values -- list of strings

    Returns a set of pairs (row, col).

    If 'values' is empty, returns an empty set.

    This interprets compressed point lists.

    Doesn't complain if there is overlap, or if a single point is
        specified as a 1x1 retangle.

    Raises ValueError if the data is otherwise malformed.
    """
    result = set()
    for s in values:
        # No need to use parse_compose(),
        # as \: would always be an error.
        p1, is_retangle, p2 = s.partition(b":")
        if is_retangle:
            top, left = interpret_point(p1, context)
            bottom, right = interpret_point(p2, context)
            if not (bottom <= top and left <= right):
                raise ValueError
            for row in range(bottom, top + 1):
                for col in range(left, right + 1):
                    result.add((row, col))
        else:
            pt = interpret_point(p1, context)
            result.add(pt)
    return result


def serialize_point_list(points, context):
    """
    Serialize a list of Points, Moves, or Stones.

    points -- iterable of pairs (row, col)

    Returns a list of strings.

    If 'points' is empty, returns an empty list.

    Doesn't produce a compressed point list.
    """
    result = [
        serialize_point(point, context) 
        for point in points
    ]
    result.sort()
    return result


def interpret_AP(s, context):
    """
    Interpret an AP (application) property value.

    Returns a pair of strings (name, version number)

    Permits the version number to be missing (which is forbidden
        by the SGF spec), in which case the second returned value
        is an empty string.
    """
    application, version = sgf_grammer.parse_compose(s)
    if version is None:
        version = b""
    return (
        interpret_simpletext(application, context),
        interpret_simpletext(version, context),
    )


def serialize_AP(value, context):
    """
    Serialize an AP (application) property value.

    value -- pair (application, version)
        application -- string
        version     -- string

    Note this takes a single parameter (which is a pair).
    """
    application, version = value
    return sgf_grammer.compose(
        serialize_simpletext(application, context),
        serialize_simpletext(version, context),
    )


def interpret_ARLN_list(values, context):
    """
    Interpret an AR (arrow) or LN (line) property value.

    Returns a list or pairs (point, point), where point is a pair
        (row, col).
    """
    result = []
    for s in values:
        p1, p2 = sgf_grammer.parse_compose()
        result.append(
            (interpret_point(p1, context),
             interpret_point(p2, context))
        )
    return result


def serialize_ARLN_list(values, context):
    """
    Serialize an AR (arrow) or LN (line) property value.

    values -- list of pairs (point, point), where point is a pair (row, col)
    """
    return [
        b":".join(
            (serialize_point(p1, context),
             serialize_point(p2, context))
        )
        for p1, p2 in values
    ]


def interpret_FG(s, context):
    """
    Interpret an FG (figure) property value.

    Returns a pair (flags, string), or None.

    flags is an integer; see http://www.red-bean.com/sgf/properties.html#FG
    """
    if s == b"":
        return None
    flags, name = sgf_grammer.parse_compose(s)
    return int(flags), interpret_simpletext(name, context)


def serilaize_FG(value, context):
    """
    Serialize an FG (figure) property value.

    value -- pair (flags, name), or None
        flags -- int
        name  -- string

    Use serialize_FG(None) to produce an empty value.
    """
    if value is None:
        return b""
    flags, name = value
    return str(flags).encode('ascii') + b":" + serialize_simpletext(name, context)


def interpret_LB_list(values, context):
    """
    Interpret an LB (label) property value.

    Returns a list of pairs ((row, col), string).
    """
    result = []
    for s in values:
        point, label = sgf_grammer.parse_compose(s)
        result.append(
            (
                interpret_point(point, context),
                interpret_simpletext(label, context),
            )
        )
    return result


def serialize_LB_list(values, context):
    """
    Serialize an LB (label) property value.

    values -- list of pairs ((row, col), string)
    """
    return [
        b":".join(
            (
                serialize_point(point, context),
                serialize_simpletext(text, context),
            )
        )
        for point, text in values
    ]


class Property_type:
    """
    Description of a property type.
    """
    def __init__(
            self,
            interpreter,
            serializer,
            uses_list,
            allows_empty_list=False):
        
        self.interpreter = interpreter
        self.serializer = serializer
        self.uses_list = uses_list
        self.allows_empty_list = bool(allows_empty_list)


def _make_property_type(
        type_name,
        allows_empty_list=False):
    
    return Property_type(
        globals()["interpret_" + type_name],
        globals()["serailize_" + type_name],
        uses_list=(type_name.endswith("_list")),
        allows_empty_list=allows_empty_list,
    )


_property_types_by_name = {
    'none': _make_property_type('none'),
    'number': _make_property_type('number'),
    'real': _make_property_type('real'),
    'double': _make_property_type('double'),
    'color': _make_property_type('color'),
    'simpletext': _make_property_type('simpletext'),
    'text': _make_property_type('text'),
    'point': _make_property_type('point'),
    'move': _make_property_type('move'),
    'point_list': _make_property_type('point_list'),
    'point_elist': _make_property_type('point_list', allows_empty_list=True),
    'stone_list': _make_property_type('point_list'),
    'AP': _make_property_type('AP'),
    'ARLN_list': _make_property_type('ARLN_list'),
    'FG': _make_property_type('FG'),
    'LB_list': _make_property_type('LB_list'),
}


P = _property_types_by_name

_property_types_by_ident = {
    b'AB': P['stone_list'],                 # setup         Add Black
    b'AE': P['point_list'],                 # setup         Add Empty
    b'AN': P['simpletext'],                 # game-info     Annotation
    b'AP': P['AP'],                         # root          Application
    b'AR': P['ARLN_list'],                  # -             Arrow
    b'AW': P['stone_list'],                 # setup         Add White
    b'B': P['move'],                        # move          Black
    b'BL': P['real'],                       # move          Black time left
    b'BM': P['double'],                     # move          Bad move
    b'BR': P['simpletext'],                 # game-info     Black rank
    b'BT': P['simpletext'],                 # game-info     Black team
    b'C': P['text'],                        # -             Comment
    b'CA': P['simpletext'],                 # root          Charset
    b'CP': P['simpletext'],                 # game-info     Copyright
    b'CR': P['point_list'],                 # -             Circle
    b'DD': P['point_elist'],                # - [inherit]   Dim points
    b'DM': P['double'],                     # -             Even position
    b'DO': P['none'],                       # move          Doubtful
    b'DT': P['simpletext'],                 # game-info     Date
    b'EV': P['simpletext'],                 # game-info     Event
    b'FF': P['number'],                     # root          Fileformat
    b'FG': P['FG'],                         # -             Figure
    b'GB': P['double'],                     # -             Good for Black
    b'GC': P['text'],                       # game-info     Game comment
    b'GM': P['number'],                     # root          Game
    b'GN': P['simpletext'],                 # game-info     Game name
    b'GW': P['double'],                     # -             Good for White
    b'HA': P['number'],                     # game-info     Handicap
    b'HO': P['double'],                     # -             Hotspot
    b'IT': P['none'],                       # move          Interesting
    b'KM': P['real'],                       # game-info     Komi
    b'KO': P['none'],                       # move          Ko
    b'LB': P['LB_list'],                    # -             Label
    b'LN': P['ARLN_list'],                  # -             Line
    b'MA': P['point_list'],                 # -             Mark
    b'MN': P['number'],                     # move          set move number
    b'N': P['simpletext'],                  # -             Nodename
    b'OB': P['number'],                     # move          OtStones Black
    b'ON': P['simpletext'],                 # game-info     Opening
    b'OT': P['simpletext'],                 # game-info     Overtime
    b'OW': P['number'],                     # move          OtStones White
    b'PB': P['simpletext'],                 # game-info     Player Black
    b'PC': P['simpletext'],                 # game-info     Place
    b'PL': P['colour'],                     # setup         Player to play
    b'PM': P['number'],                     # - [inherit]   Print move mode
    b'PW': P['simpletext'],                 # game-info     Player White
    b'RE': P['simpletext'],                 # game-info     Result
    b'RO': P['simpletext'],                 # game-info     Round
    b'RU': P['simpletext'],                 # game-info     Rules
    b'SL': P['point_list'],                 # -             Selected
    b'SO': P['simpletext'],                 # game-info     Source
    b'SQ': P['point_list'],                 # -             Square
    b'ST': P['number'],                     # root          Style
    b'SZ': P['number'],                     # root          Size
    b'TB': P['point_elist'],                # -             Territory Black
    b'TE': P['double'],                     # move          Tesuji
    b'TM': P['real'],                       # game-info     Timelimit
    b'TR': P['point_list'],                 # -             Triangle
    b'TW': P['point_elist'],                # -             Territory White
    b'UC': P['double'],                     # -             Unclear pos
    b'US': P['simpletext'],                 # game-info     User
    b'V': P['real'],                        # -             Value
    b'VW': P['point_elist'],                # - [inherit]   View
    b'W': P['move'],                        # move          White
    b'WL': P['real'],                       # move          White time left
    b'WR': P['simpletext'],                 # game-info     White rank
    b'WT': P['simpletext'],                 # game-info     White team
}

_text_property_type = P['text']

del P


class Presenter(_Context):
    """
    Convert property values between Python and SGF-string representations.

    Instantiate with:
        size     -- board size (int)
        encoding -- encoding for the SGF strings

    Public attributes (treat as read-only):
        size     -- int
        encoding -- string (normalized form)

    See the _property_types_by_ident table above for a list of properties
        initially Known, and their types.

    Initially, treats unknown (private) properties as if they had type Text.
    """

    def __init__(self, size, encoding):
        try:
            encoding = normalize_charset_name(encoding)
        except LookupError:
            raise ValueError(
                "unknown encoding: %s" % encoding
            )
        _Context.__init__(self, size, encoding)
        self.property_types_by_ident = _property_types_by_ident.copy()
        self.default_property_type = _text_property_type
    
    def get_property_type(self, identifier):
        """
        Return the Property_type for the specified PropIdent.

        Raises KeyError if the property is unknown.
        """
        return self.property_types_by_ident[identifier]
    
    def register_property(
            self,
            identifier,
            property_type):
        
        """
        Specify the Property_type for a PropIdent.
        """
        self.property_types_by_ident[identifier] = property_type
    
    def deregister_property(self, identifier):
        """
        Forget the type for the specified PropIdent.
        """
        del self.property_types_by_ident[identifier]
    
    def set_private_property_type(self, property_type):
        """
        Specify the Property_type to use for unknown properties.

        Pass property_type = None to make unknown properties
            raise an error.
        """
        self.default_property_type = property_type
    
    def _get_effective_property_type(self, identifier):
        try:
            return self.property_types_by_ident[identifier]
        except KeyError:
            result = self.default_property_type
            if result is None:
                raise ValueError("unknown property")
            return result
    
    def interpret_as_type(
            self,
            property_type,
            raw_values):
        
        """
        variant of interpret() for explicity specified type.

        property_type -- Property_type
        """
        if not raw_values:
            raise ValueError("no raw values")
        if property_type.uses_list:
            if raw_values == [b""]:
                raw = []
            else:
                raw = raw_values
        else:
            if len(raw_values) > 1:
                raise ValueError("multiple values")
            raw = raw_values[0]
        return property_type.interpreter(raw, self)
    
    def interpret(
            self,
            identifier,
            raw_values):
        
        """
        Return a Python representation of a property value.

        identifier -- PropIdent
        raw_values -- nonempty list of 8-bit strings in the presenter's encoding

        See the interpret_... functions above for details of how values are
            represented as Python types.

        Raises ValueError if it cannot interpret the value.

        Note that in some cases the interpret_... functions accepts values which
            are not strictly permitted by the specification.

        elist handling: if the property's value type is a list type and
            'raw_values' is a list containing a single empty string, passes
            an empty list to the interpret_... function (that is, this function
            treats all lists like elists).

        Doesn't enforce range restrictions on values with type Number.
        """
        return self.interpret_as_type(
            self._get_effective_property_type(identifier),
            raw_values,
        )
    
    def serialize_as_type(
            self,
            property_type,
            value):
        
        """
        Variant of serialize() for explicity specified type.

        property_type -- Property_type
        """
        serialized = property_type.serilizer(value, self)
        if property_type.uses_list:
            if serialized == []:
                if property_type.allows_empty_list:
                    return [b""]
                else:
                    raise ValueError("empty list")
            return serialized
        else:
            return [serialized]
    
    def serialize(self, identifier, value):
        """
        Serialize a Python representation of a property value.

        identifier -- PropIdent
        value      -- corresponding Python value

        Returns a nonempty list of 8-bit strings in the presenter's encoding,
            suitable for use as raw PropValues.

        See the serialize_... functions above for details of the acceptable
            values for each type.

        elist handling: if the property's value type is an elist type and the
            serailize_... function returns an empty list, this returns a list
            containing a single empty string.

        Raises ValueError if it cannot serialize the value.

        In general, the serialize_... functions try not to produce an invalid
            result, but do not try to prevent garbage input happening to produce
            a valid result.
        """
        return self.serialize_as_type(
            self._get_effective_property_type(identifier),
            value,
        )