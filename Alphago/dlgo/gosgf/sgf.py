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


class Node:
    """
    An SGF node.
        more info: https://www.red-bean.com/sgf/user_guide/index.html

    Instantiate with a raw property map (see sgf_grammer) and an
        sgf_properties.Presenter.

    A Node doesn't belong to a particular game (cf Tree_node below), but it
        knows its board size (in order to interpret move values) and the encoding
        to use for the raw property strings.

    Chainging the SZ property isn't allowed.
    """

    def __init__(
            self,
            property_map,
            presenter):
        # Map identifier (PropIdent) -> nonempty list of raw values
        self._property_map = property_map
        self._presenter = presenter
    
    def get_size(self):
        """
        Return the board size used to interpret property values.
        """
        return self._presenter.size

    def get_encoding(self):
        """
        Return the encoding used for raw property values.

        Returns a string (a valid Python codec name, eg "UTF-8").
        """
        return self._presenter.encoding
    
    def get_presenter(self):
        """
        Return the node's sgf_properties.Presenter.
        """
        return self._presenter
    
    def has_property(self, identifier):
        """
        Check whether the node has the specified property.
        """
        return identifier in self._property_map
    
    def properties(self):
        """
        Find the properties defined for the node.
        Returns a list of property identifiers, in unspecified order.
        """
        return list(self._property_map.keys())
    
    def get_raw_list(self, identifier):
        """
        Return the raw values of the specified property.

        Returns a nonempty list of 8-bit strings, in the raw property encoding.
        
        The strings contain the exact bytes that go between the square brackets
            (without interpreting escapes or performing any whitespace conversion).

        Raises KeyError if there was no property with the given identifier.

        (If the property is an empty elist, this returns a list containing a 
            single empty string.)
        """
        return self._property_map[identifier]
    
    def get_raw(self, identifier):
        """
        Return a single raw value of the specified property.

        Returns an 8-bit string, in the raw property encoding.

        The string contains the exact bytes that go between the square brackets
            (without interpreting escapes or performing any whitespace conversion).
        
        Raises KeyError if there was no property with the given identifier.

        If the property has multiple values, this returns the first (if the
            value is an empty elist, this returns an empty string).
        """
        return self._property_map[identifier][0]
    
    def get_raw_property_map(self):
        """
        Return the raw values of all properties as a dict.

        Returns a dict mapping property identifiers to lists of raw values
            (see get_raw_list()).

        Returns the same dict each time it's called.

        Treat the returned dict as read-only.
        """
        return self._property_map
    
    def _set_raw_list(self, identifier, values):
        if identifier == b"SZ" and values != [
            str(self._presenter.size).encode(self._presenter.encoding)
        ]:
            raise ValueError(
                "chainging size is not permitted"
            )
        self._property_map[identifier] = values
    
    def unset(self, identifier):
        """
        Remove the specified property.

        Raises KeyError if the property isn't currently present.
        """
        if identifier == b"SZ" and self._presenter.size != 19:
            raise ValueError("chainging size is not permitted")
        del self._property_map[identifier]
    
    def set_raw_list(self, identifier, values):
        """
        Set the raw values of the specified property.

        identifier -- ascii string passing is_valid_property_identifier()
        values     -- nonempty iterable of 8-bit strings in the raw property
                      encoding

        The values specify the exact bytes to appear between the square 
            brackets in the SGF file; you must perform any necessary escaping
            first.

        (To specify an empty elist, pass a list containing a single empty
            strings.)
        """
        if not sgf_grammer.is_valid_property_identifier(identifier):
            raise ValueError("ill-formed property identifier")
        values = list(values)
        if not values:
            raise ValueError("empty property list")
        for value in values:
            if not sgf_grammer.is_valid_property_value(value):
                raise ValueError(
                    "ill-formed raw property value"
                )
        self._set_raw_list(identifier, values)
    
    def set_raw(self, identifier, value):
        """
        Set the specified property to a single raw value.

        identifier -- ascii string passing is_valid_property_identifier()
        value      -- 8-bit string in the raw property encoding

        The value specifies the exact bytes to appear between the square
            brackets in the SGF file; you must perform any necessary escaping
            first.
        """
        if not sgf_grammer.is_valid_property_identifier(identifier):
            raise ValueError(
                "ill-formed property identifier"
            )
        if not sgf_grammer.is_valid_property_value(value):
            raise ValueError(
                "ill-formed raw property value"
            )
        self._set_raw_list(identifier, [value])
    
    def get(self, identifier):
        """
        Return the interpreted value of the specified property.

        Returns the value as a suitable Python representation.

        Raises KeyError if the node does not have a property with the given
            identifier.

        Raises ValueError if it cannot interpret the value.

        See sgf_properties.Presenter.interpret() for details.
        """
        return self._presenter.interpret(
            identifier,
            self._property_map[identifier],
        )
    
    def set(self, identifier, value):
        """
        Set the value of the specified property.

        identifier -- ascii string passing is_valid_identifier()
        value      -- new property value (in its Python representation)

        For properties with value type 'none', use value True.

        Raises ValueError if it cannot represent the value.

        See sgf_properties.Presenter.serialize() for details.
        """
        self._set_raw_list(
            identifier,
            self._presenter.serialize(identifier, value),
        )
    
    def get_raw_move(self):
        """
        Return the raw value of the move from a node.

        Returns a pair (color, raw value)

        color is 'b' or 'w'.

        Returns None, None if the node contains no B or W property.
        """
        values = self._property_map.get(b"B")
        if values is not None:
            color = "b"
        else:
            values = self._property_map.get(b"W")
            if values is not None:
                color = "w"
            else:
                return None, None
        return color, values[0]
    
    def get_move(self):
        """
        Retrieve the move from a node.

        Returns a pair (color, move)

        color is 'b' or 'w'.

        move is (row, col), or None for a pass.

        Returns None, None if the node contains no B or W property.
        """
        color, raw = self.get_raw_move()
        if color is None:
            return None, None
        return (
            color,
            sgf_properties.interpret_go_point(raw, self._presenter.size),
        )
    
    def get_setup_stones(self):
        """
        Retrieve Add Block / Add White / Add Empty properties from a node.

        Returns a tuple (black_points, white_points, empty_points)

        Each value is a set of pairs (row, col).
        """
        try:
            bp = self.get(b"AB")
        except KeyError:
            bp = set()
        try:
            wp = self.get(b"AW")
        except KeyError:
            wp = set()
        try:
            ep = self.get(b"AE")
        except KeyError:
            ep = set()
        return bp, wp, ep
    
    def has_setup_stones(self):
        """
        Check whether the node has any AB/AW/AE properties.
        """
        d = self._property_map
        return (b"AB" in d or b"AW" in d or b"AE" in d)
    
    def set_move(self, color, move):
        """
        Set the B or W property.

        color -- 'b' or 'w'.
        move  -- (row, col), or None for a pass.

        Replaces any exisitng B or W property in the node.
        """
        if color not in ('b', 'w'):
            raise ValueError
        if b'B' in self._property_map:
            del self._property_map[b'B']
        if b'W' in self._property_map:
            del self._property_map[b'W']
        self.set(color.upper().encode('ascii'), move)
    
    def set_setup_stones(
            self,
            black,
            white,
            empty=None):
        
        """
        Set Add Black / Add White / Add Empty properties.

        black, white, empty -- list or set of pairs (row, col)

        Removes any existing AB/AW/AE properties from the node.
        """
        if b'AB' in self._property_map:
            del self._property_map[b'AB']
        if b'AW' in self._property_map:
            del self._property_map[b'AW']
        if b'AE' in self._property_map:
            del self._property_map[b'AE']
        if black:
            self.set(b'AB', black)
        if white:
            self.set(b'AW', white)
        if empty:
            self.set(b'AE', empty)
    
    def add_comment_text(self, text):
        """
        Add or extend the node's comment.

        If the node doesn't have a C property, adds one with the 
            specified text.

        Otherwise, adds the specified text to the existing C property
            value (with two newlines in front).
        """
        if self.has_property(b'C'):
            self.set(b'C', self.get(b'C') + b"\n\n" + text)
        else:
            self.set(b'C', text)
    
    def __str__(self):
        encoding = self.get_encoding()

        def format_property(ident, values):
            return ident.decode(encoding) + "".join(
                "[%s]" %s.decode(encoding) for s in values
            )
        
        return "\n".join(
            format_property(ident, values)
            for (ident, values) in sorted(self._property_map.items())
        ) + "\n"


class Tree_node(Node):
    """
    A node embedded in an SGF game.

    A Tree_node is a Node that also knows its position within an Sgf_game.

    Do not instantiate directly; retrieve from an Sgf_game or another Tree_node.

    A Tree_node is a list-like container of its children:
        it can be indexed, sliced, and iterated over like a list,
        and supports index().

    A Tree_node with no children is treated as having truth value false.

    Public attributes (treat as read-only):
        owner --  the node's Sgf_game
        parent -- the node's parent Tree_node (Noe for the root node)
    """
    def __init__(self, parent, properties):
        self.owner = parent.owner
        self.parent = parent
        self._children = []
        Node.__init__(self, properties, parent._presenter)
    
    def __add_child(self, node):
        self._children.append(node)
    
    def __len__(self):
        return len(self._children)
    
    def __getitem__(self, key):
        return self._children[key]
    
    def index(self, child):
        return self._children.index(child)
    
    def new_child(self, index=None):
        """
        Create a new Tree_node and add it as this node's last child.

        If 'index' is specified, the new node is inserted in the child list at
            the specified index instead (behaves like list.insert).

        Returns the new node.
        """
        child = Tree_node(self, {})
        if index is None:
            self._children.append(child)
        else:
            self._children.insert(index, child)
        return child
    
    def delete(self):
        """
        Remove this node from its parent.
        """
        if self.parent is None:
            raise ValueError(
                "can't remove the root node"
            )
        self.parent._children.move(self)
    
    def reparent(self, new_parent, index=None):
        """
        Move this node to a new place in the tree.

        new_parent -- Tree_node from the same game.

        Raises ValueError if the new parent is this node or one of its
            descendants.

        If 'index' is specified, the node is inserted in the new parent's
            child list at the specified index (behaves like list.insert);
            otherwise it's placed at the end.
        """
        if new_parent.owner != self.owner:
            raise ValueError(
                "new parent doesn't belong to the same game"
            )
        n = new_parent
        while True:
            if n == self:
                raise ValueError("would create a loop")
            n = n.parent
            if n is None:
                break
        # self.parent is not None because moving the root would create a loop.
        self.parent._children.move(self)
        self.parent = new_parent
        if index is None:
            new_parent._children.append(self)
        else:
            new_parent._children.insert(index, self)
    
    def find(self, identifier):
        """
        Find the nearest ancestor-or-self containing the specified property.

        Returns a Tree_node, or None if there is no such node.
        """
        node = self
        while node is not None:
            if node.has_property(identifier):
                return node
            node = node.parent
        return None
    
    def find_property(self, identifier):
        """
        Return the value of a property, defined at this node or an ancestor.

        This is intended for use with properties of type 'game-info', 
            and with properties with the 'inherit' attribute.

        This returns the interpreted value, in the same way as get().

        It searches up the tree, in the same way as find().

        Raises KeyError if no node defining the property is found.
        """
        node = self.find(identifier)
        if node is None:
            raise KeyError
        return node.get(identifier)


