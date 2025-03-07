
import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile

LOG_OUTPUT_FORMATS = ['stdout', 'log', 'json']

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class OutputFormat(object):

    def writekvs(self, kvs):
        """Write key-value pairs"""
        raise NotImplementedError
    
    def writeseq(self, args):
        """Write a sequence of other data (e.g. a logging message)"""
        pass

    def close(self):
        return


class HumanOutputFormat(OutputFormat):

    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)
        
        # Find max widths
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items()):
            lines.append(
                '| %s%s | %s%s |' % (
                    key,
                    ' ' * (keywidth - len(key)),
                    val,
                    ' ' * (valwidth - len(val)),
                )
            )
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()
    

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s
    

    def writeseq(self, args):
        for arg in args:
            self.file.write(arg)
        self.file.write('\n')
        self.file.flush()


class JSONOutputFormat(OutputFormat):

    def __init__(self, file):
        self.file = file
    

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()


class TensorBoardOutputFormat(OutputFormat):
    """Dumps key/value pairs into TensorBoard's numeric format."""

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = 'events'
        path = osp.join(osp.abspath(dir), prefix)

        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))
    

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {'tag': k, 'simple_value': float(v)}
            return self.tf.compat.v1.Summary.Value(**kwargs)
        
        summary = self.tf.compat.v1.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_item=time.time(), summary=summary)
        event.step = self.step  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1
    

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


def make_output_format(format, ev_dir):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    
    elif format == "log":
        log_file = open(osp.join(ev_dir, 'log.txt'), 'wt')
        return HumanOutputFormat(log_file)
    
    elif format == "json":
        json_file = open(osp.join(ev_dir, 'progress.json'), 'wt')
        return JSONOutputFormat(json_file)
    
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, 'tb'))
    
    else:
        raise ValueError('Unknown format specified: %s' % (format,))


