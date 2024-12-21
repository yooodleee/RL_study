#!/usr/bin/env python
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
import os
from a3c import A3C
from envs import create_env
from constants import constants
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.compat.v1.VERSION) >= distutils.version.LooseVersion('0.12.0')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastServer(tf.compat.v1.train.Saver):
    def save(
        self,
        sess,
        save_path,
        global_step=None,
        latest_filename=None,
        meta_graph_suffix="meta",
        write_meta_graph=True
    ):
        super(FastServer, self).save(sess, save_path, global_step, latest_filename, meta_graph_suffix, False)
    
    def run(args, server):
        env = create_env(
            args.env_id,
            client_id=str(args.task),
            remotes=args.remotes,
            envWrap=args.envWrap,
            designHead=args.designHead,
            noLifeRewaard=args.noLifeReward
        )
        trainer = A3C(
            env,
            args.task,
            args.visualise,
            args.unsup,
            args.designHead,
            args.noReward
        )

        # logging
        if args.task == 0:
            with open(args.log_dir + '/log.txt', 'w') as fid:
                for key, val in constants.items():
                    fid.write('%s: %s\n'%(str(key), str(val)))
                fid.write('designHead: %s\n'%args.designHead)
                fid.write('input observation: %s\n'%str(env.observation_space.shape))
                fid.write('env name: %s\n'%str(env.spec.id))
                fid.write('unsup method type: %s\n'%str(args.unsup))
            
            # Variable names that start with "local" are not saved in checkpoints.
            if use_tf12_api:
                variables_to_save = [v for v in tf.compat.v1.global_variables() if not v.name.startswith("local")]
                init_op = tf.compat.v1.variables_initializer(variables_to_save)
                init_all_op = tf.compat.v1.global_variables_initializer()
            else:
                variables_to_save = [v for v in tf.compat.v1.all_variables() if not v.name.startswith("local")]
                init_op = tf.compat.v1.initialize_variables(variables_to_save)
                init_all_op = tf.compat.v1.initialize_all_variables()
            saver = FastServer(variables_to_save)
            if args.pretrain is not None:
                variables_to_save = [v for v in tf.compat.v1.trainable_variables() if not v.name.startswith("local")]
                pretrain_server = FastServer(variables_to_save)
            
            var_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, tf.compat.v1.get_variable_scope().name)
            logger.info('Trainable vars:')
            for v in var_list:
                logger.info('   %s  %s', v.name, v.get_shape())
            
            def init_fn(ses):
                logger.info("Initializing all parameters.")
                ses.run(init_all_op)
                if args.pretrain is not None:
                    pretrain = tf.compat.v1.train.latest_checkpoint(args.pretrain)
                    logger.info("==> Restoring from given pretrained checkpoint.")
                    logger.info("   Pretraining address: %s", pretrain)
                    pretrain_server.restore(ses, pretrain)
                    logger.info("==> Done restoring model! Restored %d variables.", len(variables_to_save))
            
            config = tf.compat.v1.ConfigProto(device_filters=["/"])