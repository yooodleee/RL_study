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
    
    