"""PyProcess.

This file includes utilities for running code in separate  Python processes as
part of a TensorFlow graph. It is similar to tf.py_func, but the code is run in
separate processes to avoid the GIL.

Example:

    class Zeros(object):
    
        def __init__(self, dim0):
            self._dim0 = dim0

        def comput(self, dim1):
            return np.zeros([self._dim0, dim1], dtype=np.int32)
        
        @staticmethod
        def _tensor_specs(method_name, kwargs, constructor_kwargs):
            dim0 = constructor_kwargs['dim0']
            dim1 = kwargs['dim1']
            if method_name == 'compute':
                return tf.contrib.framework.TensorSpec([dim0, dim1], tf.int32)
    
    with tf.Graph().as_default():
        p = py_process.PyProcess(Zeros, 1)
        result = p.proxy.compute(2)

        with tf.train.SingularMonitoredSession(
            hooks=[py_process.PyProcessHook()]) as session:
        print(session.run(result))  # Prints[[0, 0]].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import tensorflow as tf

from tensorflow.python.util import function_utils


# nest = tf.contrib.framework.nest
nest = tf.nest


class _TFProxy(object):
    """A proxy that creates TensorFlow operations for each method call to a
    separate process."""

    def __init__(self, type_, constructor_kwargs):
        self._type = type_
        self._constructor_kwargs = constructor_kwargs

    def __getattr__(self, name):
        def call(*args):
            kwargs = dict(
                zip(function_utils.fn_args(getattr(self._type, name))[1:], args))
            specs = self._type._tensor_specs(name, kwargs, self._constructor_kwargs)

            if specs is None:
                raise ValueError(
                    'No tensor specifications were provided for: %s' % name)
            
            flat_dtypes = nest.flatten(nest.map_structure(lambda s: s.dtype, specs))
            flat_shapes = nest.flatten(nest.map_structure(lambda s: s.dtype, specs))

            def py_call(*args):
                try:
                    self._out.send(args)
                    result = self._out.recv()
                    if isinstance(result, Exception):
                        raise result
                    if result is not None:
                        return result
                except Exception as e:
                    if isinstance(e, IOError):
                        raise StopIteration()   # Clean exit.
                    else:
                        raise
            
            result = tf.py_function(
                py_call(), (name,) + tuple(args), flat_dtypes, name=name)
            
            if isinstance(result, tf.Operation):
                return result
            
            for t, shape in zip(result, flat_shapes):
                t.set_shape(shape)
            return nest.pack_sequence_as(specs, result)
        return call
    

    def _start(self):
        self._out, in_ = multiprocessing.Pipe()
        self._process = multiprocessing.Process(
            target=self._worler_fn,
            args = (self._type, self._constructor_kwargs, in_))
        self._process.start()
        result = self._out.recv()

        if isinstance(result, Exception):
            raise result
    
    def _close(self, session):
        try:
            self._out.send(None)
            self._out.close()
        except IOError:
            pass
        self._process.join()
    
    def _worker_fn(self, type_, constructor_kwargs, in_):
        try:
            o = type_(**constructor_kwargs)

            in_.send(None)  # Ready.

            while True:
                # Recieve request.
                serialized = in_.recv()