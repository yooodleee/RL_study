"""
Log component.
"""
import collections
import os
import csv


class CsvWriter:
    """
    A logging object writing to a CSV file.

    Each `write()` takes a `OrderDict`, creating one column in the CSV file for
        each dictionary key on the first call. Successive calls to `write()` must
        contain the same dictionary keys.
    """

    def __init__(self, fname: str):
        """
        Initializes a 'CsvWriter'.

        Args:
            fname: File name(path) for file to be written to.
        """
        if fname is not None and fname != '':
            dirname = os.path.dirname(fname)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        
        self._fname = fname
        self._header_written = False
        self._fieldnames = None

    def write(
        self, values: collections.OrderedDict
    )-> None:
        """
        Appends given values as new row to CSV file.
        """
        if self._fname is None or self._fname == '':
            return
        
        if self._fieldnames is None:
            self._fieldnames = values.keys()
        # Open a file in 'append' mode, so we can continue logging safely to the
        # same file after e.g. restarting from a checkpoint.
        with open(
            self._fname, 'a', encoding='utf-8'
        ) as file_:
            # Always use same fieldnames to create writer, this way a consistency
            # check is performed automatically on each write.
            writer = csv.DictWriter(
                file_, fieldnames=self._fieldnames
            )
            # Write a header if this is the very first write.
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(values)

    def close(self)-> None:
        """
        Closes the 'CsvWriter'.
        """