from __future__ import print_function
from contextlib import contextmanager
import numpy as np
import time 
import shlex
import subprocess


# ===============================================================
# Misc 
# ===============================================================

def fmt_row(width, row, header=False):
    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-" * len(out)
    return out


