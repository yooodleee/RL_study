def set_session_config(
    per_process_gpu_memory_fraction=None,
    allow_growth=None):
    """
    Params:
        allow_growth: When necessary, reserve memory
        (float) per_process_gpu_memory_fraction: specify GPU memory usage as 
            0 to 1
    """

    import tensorflow as tf
    import keras.backend as k

    config = tf.compat.v1.ConfigProto(
                gpu_options = tf.compat.v1.GPUOptions(
                                per_process_gpu_memory_fraction = \
                                    per_process_gpu_memory_fraction,
                                allow_growth = allow_growth))
    sess = tf.compat.v1.Session(config=config)
    k.set_session(sess)