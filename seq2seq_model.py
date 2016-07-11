"""
Simple sequence-to-sequence model 
"""

import math
import os
import random
import sys
import time
import numpy as np

import cPickle

import tensorflow as tf


class SimpleSeq2SeqModel:
	"""A simple sequence-to-sequence model. It implements a 
	(multi-layer) RNN encoder followed by a (multi-layer) 
	RNN  decoder. Encoder and decoder use the same RNN cell type,
	but don't share parameters
	"""

	def __init__(self, num_units, num_layers, batch_size, learning_rate, learning_rate_decay_factor):
		"""Create the model

		Args:
			num_units: number of units in each layer of the model.
			num_layers: number of layers in the model.
			batch_size: the size of the batches used during training.
			learning_rate: starting learning rate. 
			learning_rate_decay_factor: decay factor for the learning rate, when needed.

		"""
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)

		# Create the internal multi-layer cell for the RNN.
		single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
		cell = single_cell
		if num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
		
		# Feeds for inputs.  TODO: sequence size? right now I think it is only for a sequence of size 1...
		self.encoder_inputs = tf.placeholder(tf.int32, shape=[None], name="encoder")
		self.decoder_inputs = tf.placeholder(tf.int32, shape=[None], name="decoder")
		self.weights = tf.ones_like(self.decoder_inputs, dtype=tf.float32)

		self.output, self.state = tf.nn.seq2seq.basic_rnn_seq2seq(self.encoder_inputs, self.decoder_inputs,cell)
		
		# Build a standard sequence loss function: mean cross-entropy over each item of each sequence
		self.loss = seq2seq.sequence_loss(self.output, self.decoder_inputs, self.weights)

		# Gradients and SGD update operation for training the model.

		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

		train_op = optimizer.minimize(loss)

		self.saver = tf.train.Saver(tf.all_variables())

	def step(self, session, encoder_inputs, decoder_inputs, forward_only=False):
		"""Run a step of the model feeding the given inputs. 
			Args:
				session: tensorflow session to use.
				encoder_inputs: list of numpy int vectors to feed as encoder inputs.
				decoder_inputs: list of numpy int vectorss to feed as decoder inputs.
				forward_only: whether to do the backward step or only forward, defaut=False

			Returns:
				(gradient norm, output)	
		"""
		# TODO

		sess.run(tf.initialize_all_variables())

		


