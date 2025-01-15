"""PyProcess.

This file includes utilities for running code in separate  
    Python processes as part of a TensorFlow graph. 
It is similar to tf.py_func, but the code is run in
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
import multiprocessing.pool
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
                zip(function_utils.\
                    fn_args(getattr(self._type, name))[1:], args))
            specs = self._type._tensor_specs(name, 
                                             kwargs, 
                                             self._constructor_kwargs)

            if specs is None:
                raise ValueError(
                    'No tensor specifications were provided for: %s' \
                        % name)
            
            flat_dtypes = nest.flatten(nest.map_structure(lambda s: s.dtype, 
                                                          specs))
            flat_shapes = nest.flatten(nest.map_structure(lambda s: s.dtype, 
                                                          specs))

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
            
            result = tf.py_function(py_call(), 
                                    (name,) + tuple(args), 
                                    flat_dtypes, 
                                    name=name)
            
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
                            args = (self._type, 
                                    self._constructor_kwargs, 
                                    in_))
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

                if serialized is None:
                    if hasattr(o, 'close'):
                        o.close()
                    in_.close()
                    return
                
                method_name = str(serialized[0])
                inputs = serialized[1:]

                # Compute result.
                results = getattr(0, method_name)(*inputs)
                if results is not None:
                    results = nest.flatten(results)
                
                # Respond.
                in_.send(results)
        except Exception as e:
            if 'o' in locals() and hasattr(o, 'close'):
                try:
                    o.close()
                except:
                    pass
            in_.send(e)



class PyProcess(object):
    COLLECTION = 'py_process_processes'

    def __init__(self, type_, *constructor_args, **constructor_kwargs):
        self._type = type_
        self._constructor_kwargs = dict(
            zip(function_utils.fn_args(type_.__init__)[1:], constructor_args))
        self._constructor_kwargs.update(constructor_kwargs)

        tf.compat.v1.add_to_collection(PyProcess.COLLECTION, self)

        self._proxy = _TFProxy(type_, self._constructor_kwargs)
    
    @property
    def proxy(self):
        """A proxy that creates TensorFlow operations for each method call."""
        return self._proxy
    
    def close(self, session):
        self._proxy._close(session)
    
    def start(self):
        self._proxy._start()



class PyProcessHook(tf.compat.v1.train.SessionRunHook):
    """A MonitoredSession hook that starts and stops PyProcess instances."""

    def begin(self):
        tf.compat.v1.logging.info('Starting all processes.')
        tp = multiprocessing.pool.ThreadPool()
        tp.map(lambda p: p.start(), 
               tf.compat.v1.get_collection(PyProcess.COLLECTION))
        tp.close()
        tp.join()
        tf.compat.v1.logging.info('All process started.')
    
    def end(self, session):
        tf.compat.v1.logging.info('Closing all processes.')
        tp = multiprocessing.pool.ThreadPool()
        tp.map(lambda p: p.close(session), 
               tf.compat.v1.get_collection(PyProcess.COLLECTION))
        tp.close()
        tp.join()
        tf.compat.v1.logging.info('All processes closed.')