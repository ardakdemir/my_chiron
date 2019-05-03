#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Mon Mar 27 14:04:57 2017
# from rnn import rnn_layers

# 20190116
# UNet for processing the same amout of data as chiron, revised by Yao-zhong@imsut

from __future__ import absolute_import
from __future__ import print_function
import argparse, sys, os, time, json

from distutils.dir_util import copy_tree

import tensorflow as tf
import chiron.chiron_model as model

from chiron_input import read_raw_data_sets
from chiron.chiron_input import read_tfrecord
from chiron.chiron_input import read_cache_dataset
from chiron.cnn import getcnnfeature
from chiron.cnn import getcnnlogit
from six.moves import range

from keras import Input, models, layers, regularizers
from keras.optimizers import RMSprop, SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras.layers import Lambda

import h5py
import numpy as np

CB = [callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="auto", restore_best_weights=True)]

def loading_data():

    print(FLAGS)
    ds = read_raw_data_sets(FLAGS.data_dir, FLAGS.train_cache,FLAGS.sequence_len, FLAGS.k_mer)

    # loading into memory
    X, seq_len, label = ds.next_batch(ds._reads_n)

    return X, seq_len, label

def ctc_loss(input_length, label_length):

    def define_loss(labels, logits):
    
        #return(tf.nn.ctc_loss(label, logits, seq_len, ctc_merge_repeated=True, time_major=False,\
        #       ignore_longer_outputs_than_inputs=True) )

        return(K.sum(logits))
        #return(K.ctc_batch_cost(labels, logits, input_length, label_length))


    return define_loss


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    #y_pred = y_pred[:,:,:]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



# UNet train for the model, non-generator version
def train():
    # loading the training data

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    trainX, seqLen, label = loading_data()

    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    trainY = K.eval(tf.SparseTensor(label[0], label[1], label[2]))
    print(trainX.shape)
    print(trainY.shape)

    # caching the training data
    """
    with h5py.File("./unet_train_cache.h5", 'w') as hf:
        hf.create_dataset("x_data", data=trainX)
        hf.create_dataset("y_data", data=trainY)
    """


    # loading keras model and train
    kernel_size =[128, 128]
    maxpooling_len = [5, 5, 2, 2, 5, 5]
    conv_window_len = [7, 7]
    dropoutRate = 0.2
    lr = 0.001

    signals = Input(name='input', shape=[FLAGS.sequence_len, 1], dtype='float32')

    conv1 = layers.Conv1D(kernel_size[0], conv_window_len[0], activation="relu")(signals)
    conv2 = layers.Conv1D(kernel_size[1], conv_window_len[1], activation="relu")(conv1)
    pool = layers.MaxPooling1D(2)(conv2)
    fc = layers.Flatten()(pool)
    fc = layers.Dense(256, activation="relu")(fc)
    #output =  [ layers.Dense(5, activation='softmax',name='c%d'%(i+1))(fc) for i in range(trainY.shape[1]) ]
    output = layers.Dense(trainY.shape[1])(fc)

    base_model = models.Model(signals, output)
    
    #y_pred = layers.Activation('Softmax')(output)
    y_pred = output
    # base_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # base_model.fit(trainX, trainY, epochs=10, batch_size=64, verbose=1, callbacks=CB, validation_split=0.2)


    labels = Input(name="labels", shape=[trainY.shape[1]],dtype='int64')
    input_length = Input(name="input_length", shape=[1], dtype='int64')
    label_length = Input(name="label_length", shape=[1],dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = models.Model(input=[signals, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc':lambda y_true, y_pred: y_pred}, optimizer='adam')
    
    #model = UNet_networkstructure_basic(rd_input, conv_window_len, maxpooling_len, True, dropoutRate)
    n_train = trainX.shape[0]
    model.summary()

    hisotry = model.fit([trainX, trainY, np.array(np.ones(n_train)*trainY.shape[1]), np.array(np.ones(n_train)*trainY.shape[1])],\
            trainY, epochs=10, batch_size=64, verbose=1, callbacks=CB, validation_split=0.2)

    
def chiron_train():
    training = tf.placeholder(tf.bool)
    global_step = tf.get_variable('global_step', trainable=False, shape=(),
                                  dtype=tf.int32,
                                  initializer=tf.zeros_initializer())
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    y_indexs = tf.placeholder(tf.int64)
    y_values = tf.placeholder(tf.int32)
    y_shape = tf.placeholder(tf.int64)
    y = tf.SparseTensor(y_indexs, y_values, y_shape)
    default_config = os.path.join(FLAGS.log_dir,FLAGS.model_name,'model.json')
    
    if FLAGS.retrain:
        if os.path.isfile(default_config):
            config_file = default_config
        else:
            raise ValueError("Model Json file has not been found in model log directory")
    else:
        config_file = FLAGS.configure   
    
    config = model.read_config(config_file)
    
    
    logits, ratio = model.inference(x, seq_length, training,FLAGS.sequence_len,configure = config)
    ctc_loss = model.loss(logits, seq_length, y)
    opt = model.train_opt(FLAGS.step_rate,
                          FLAGS.max_steps, 
                          global_step=global_step,
                          opt_name = config['opt_method'])
    step = opt.minimize(ctc_loss,global_step = global_step)
    error = model.prediction(logits, seq_length, y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    if FLAGS.retrain == False:
        sess.run(init)
        print("Model init finished, begin loading data. \n")
    else:
        saver.restore(sess, tf.train.latest_checkpoint(
            FLAGS.log_dir + FLAGS.model_name))
        print("Model loaded finished, begin loading data. \n")
    summary_writer = tf.summary.FileWriter(
        FLAGS.log_dir + FLAGS.model_name + '/summary/', sess.graph)
    model.save_model(default_config,config)
    
    train_ds,valid_ds = generate_train_valid_datasets()
    start = time.time()
    
    for i in range(FLAGS.max_steps):
        batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size)
        indxs, values, shape = batch_y
        feed_dict = {x: batch_x, seq_length: seq_len / ratio, y_indexs: indxs,
                     y_values: values, y_shape: shape,
                     training: True}
        loss_val, _ = sess.run([ctc_loss, step], feed_dict=feed_dict)
        if i % 10 == 0:
            global_step_val = tf.train.global_step(sess, global_step)
            valid_x, valid_len, valid_y = valid_ds.next_batch(FLAGS.batch_size)
            indxs, values, shape = valid_y
            feed_dict = {x: valid_x, seq_length: valid_len / ratio,
                         y_indexs: indxs, y_values: values, y_shape: shape,
                         training: True}
            error_val = sess.run(error, feed_dict=feed_dict)
            end = time.time()
            print(
            "Step %d/%d Epoch %d, batch number %d, train_loss: %5.3f validate_edit_distance: %5.3f Elapsed Time/step: %5.3f" \
            % (i, FLAGS.max_steps, train_ds.epochs_completed,
               train_ds.index_in_epoch, loss_val, error_val,
               (end - start) / (i + 1)))
            saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/model.ckpt',
                       global_step=global_step_val)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step_val)
            summary_writer.flush()
    global_step_val = tf.train.global_step(sess, global_step)
    print("Model %s saved." % (FLAGS.log_dir + FLAGS.model_name))
    print("Reads number %d" % (train_ds.reads_n))
    saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/final.ckpt',
               global_step=global_step_val)

# note the data, note this only support tf data

def generate_train_valid_datasets():
    
    if FLAGS.read_cache:
        train_ds = read_cache_dataset(FLAGS.train_cache)
        if FLAGS.validation is not None:
            valid_ds = read_cache_dataset(FLAGS.valid_cache)
        else:
            valid_ds = train_ds
        if train_ds.event.shape[1]!=FLAGS.sequence_len:
            raise ValueError("The event length of training cached dataset %d is inconsistent with given sequene_len %d"%(train_ds.event.shape()[1],FLAGS.sequence_len))
        if valid_ds.event.shape[1]!=FLAGS.sequence_len:
            raise ValueError("The event length of training cached dataset %d is inconsistent with given sequene_len %d"%(valid_ds.event.shape()[1],FLAGS.sequence_len))
        return train_ds,valid_ds
    sys.stdout.write("Begin reading training dataset.\n")
    
    train_ds = read_raw_data_sets(FLAGS.data_dir, FLAGS.train_cache,\
            FLAGS.sequence_len, FLAGS.k_mer)

    """
    train_ds = read_tfrecord(FLAGS.data_dir, 
                             FLAGS.tfrecord, 
                             FLAGS.train_cache,
                             FLAGS.sequence_len, 
                             k_mer=FLAGS.k_mer,
                             max_segments_num=FLAGS.segments_num)
    """
    sys.stdout.write("Begin reading validation dataset.\n")
    
    if FLAGS.validation is not None:
        valid_ds = read_tfrecord(FLAGS.data_dir, 
                                 FLAGS.validation,
                                 FLAGS.valid_cache,
                                 FLAGS.sequence_len, 
                                 k_mer=FLAGS.k_mer,
                                 max_segments_num=FLAGS.segments_num)
    else:
        valid_ds = train_ds
    
    return train_ds,valid_ds


def run(args):
    global FLAGS
    FLAGS = args
    FLAGS.data_dir = FLAGS.data_dir + os.path.sep
    FLAGS.log_dir = FLAGS.log_dir + os.path.sep
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')

    parser.add_argument('-i', '--data_dir', default="/data/workspace/nanopore/data/chiron_data/code_dev_sample" ,required= False,
                        help="Directory that store the tfrecord files.")
    parser.add_argument('-o', '--log_dir', default="model/", required = False  ,
                        help="log directory that store the training model.")
    parser.add_argument('-m', '--model_name', default="devTest", required = False,
                        help='model_name')
    parser.add_argument('-v', '--validation', default = None, 
                        help="validation tfrecord file, default is None, which conduct no validation")
    parser.add_argument('-f', '--tfrecord', default="train.tfrecords",
                        help='tfrecord file')
    parser.add_argument('--train_cache', default=None, help="Cache file for training dataset.")
    parser.add_argument('--valid_cache', default=None, help="Cache file for validation dataset.")
    parser.add_argument('-s', '--sequence_len', type=int, default=300,
                        help='the length of sequence')
    parser.add_argument('-b', '--batch_size', type=int, default=400,
                        help='Batch size')
    parser.add_argument('-t', '--step_rate', type=float, default=1e-2,
                        help='Step rate')
    parser.add_argument('-x', '--max_steps', type=int, default=10000,
                        help='Maximum step')
    parser.add_argument('-n', '--segments_num', type = int, default = None,
                        help='Maximum number of segments read into the training queue, default(None) read all segments.')
    parser.add_argument('--configure', default = None,
                        help="Model structure configure json file.")
    parser.add_argument('-k', '--k_mer', default=1, help='Output k-mer size')
    parser.add_argument('--retrain', dest='retrain', action='store_true',
                        help='Set retrain to true')
    parser.add_argument('--read_cache',dest='read_cache',action='store_true',
                        help="Read from cached hdf5 file.")
    parser.set_defaults(retrain=False)
    args = parser.parse_args(sys.argv[1:])
    if args.train_cache is None:
        args.train_cache = args.data_dir + '/train_cache.hdf5'
    if (args.valid_cache is None) and (args.validation is not None):
        args.valid_cache = args.data_dir + '/valid_cache.hdf5'
    run(args)


