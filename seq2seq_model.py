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
import seq2seqs 

import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils


class Seq2SeqsModel:
	"""A simple sequence-to-sequence model. It implements a 
	(multi-layer) RNN encoder followed by a (multi-layer) 
	RNN  decoder. Encoder and decoder use the same RNN cell type,
	but don't share parameters.

	EDIT: now using buckets
	"""

	def __init__(self, vocab_size,buckets, num_units, num_layers, 
		max_gradient_norm, num_decoders, batch_size, 
		learning_rate, learning_rate_decay_factor):
		"""Create the model

		Args:
			num_units: number of units in each layer of the model.
			num_layers: number of layers in the model.
			batch_size: the size of the batches used during training.
			learning_rate: starting learning rate. 
			learning_rate_decay_factor: decay factor for the learning rate, when needed.

		"""
		self.vocab_size = vocab_size
		self.buckets = buckets
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
			self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)

		self.encoder_inputs = []
		self.target_weights = []
		for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
			self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], 
				name="encoder{0}".format(i)))

		self.all_targets_weights = [] 
		self.all_decoder_inputs = []
		all_targets = []
		for j in xrange(num_decoders):
			decoder_inputs = []
			target_weights = []
			for i in xrange(buckets[-1][1] + 1):
				decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], 
					name="decoder{0}{1}".format(j,i)))
				target_weights.append(tf.placeholder(tf.float32, shape=[None], 
					name="weight{0}{1}".format(j,i)))
			
			# Our targets are decoder inputs shifted by one (remove <GO>).
			targets = [decoder_inputs[i + 1] for i in xrange(len(decoder_inputs) - 1)]
			all_targets.append(targets)
			self.all_decoder_inputs.append(decoder_inputs)
			self.all_targets_weights.append(target_weights)


		# Create the internal multi-layer cell for the RNN.
		single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
		cell = single_cell
		if num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers,state_is_tuple=True)
		

	
		self.outputs, self.losses = seq2seqs.model_with_buckets(self.encoder_inputs, 
			num_decoders,self.all_decoder_inputs, all_targets, self.all_targets_weights, 
			buckets, lambda x,y: seq2seqs.embedding_rnn_seq2seqs(x,num_decoders,y,cell,
				num_encoder_symbols=vocab_size, num_decoder_symbols=vocab_size,
				embedding_size=num_units))
	

		# Gradients and GD update operation for training the model
		params = tf.trainable_variables() 
		self.gradient_norms = []
		self.updates = []
		optimizer =tf.train.GradientDescentOptimizer(self.learning_rate)
		for b in xrange(len(buckets)):
			gradients = tf.gradients(self.losses[b],params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

			self.gradient_norms.append(norm)
			self.updates.append(optimizer.apply_gradients(zip(clipped_gradients,params),
				global_step=self.global_step))

		self.saver = tf.train.Saver(tf.all_variables())


	def step(self, session, encoder_inputs, num_decoders, decoder_inputs, target_weights,
		bucket_id, forward_only=False):
		"""Run a step of the model feeding the given inputs. 
			Args:
				session: tensorflow session to use.
				encoder_inputs: list of numpy int vectors to feed as encoder inputs.
				decoder_inputs: list of numpy int vectors to feed as decoder inputs.
				forward_only: whether to do the backward step or only forward, 
				defaut=False

			Returns:
				(gradient norm, output)	
		"""
		# TODO
    	# Check if the sizes match.
		encoder_size, decoder_size = self.buckets[bucket_id]
		if len(encoder_inputs) != encoder_size:
			raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
		if len(decoder_inputs[0]) != decoder_size:
			raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
		if len(target_weights[0]) != decoder_size:
			raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))
		

		feed_dict = {}
		for l in xrange(encoder_size):
			feed_dict[self.encoder_inputs[l].name] = encoder_inputs[l]
		for m in xrange(num_decoders):	
			for l in xrange(decoder_size):
				feed_dict[self.all_decoder_inputs[m][l].name] = decoder_inputs[m][l]
				feed_dict[self.all_targets_weights[m][l].name] = target_weights[m][l]
			# Since our targets are decoder inputs shifted by one, we need one more.
			last_target = self.all_decoder_inputs[m][decoder_size].name
			feed_dict[last_target] = np.zeros([self.batch_size], dtype=np.int32)
		
		if not forward_only:
			output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
		else:
			output_feed = [self.losses[bucket_id]]  # Loss for this batch.
			for l in xrange(decoder_size):  # Output logits.
				output_feed.append(self.outputs[bucket_id][l])

		outputs = session.run(output_feed, feed_dict)
		if not forward_only:
			return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1:]


	def get_batch(self, data, num_decoders, bucket_id):
		# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
		"""Get a random batch of data from the specified bucket, prepare for step.

		To feed data in step(..) it must be a list of batch-major vectors, while
		data here contains single length-major cases. So the main logic of this
		function is to re-index data cases to be in the proper format for feeding.

		Args:
			data: a tuple of size len(self.buckets) in which each element contains
			lists of pairs of input and output data that we use to create a batch.
			bucket_id: integer, which bucket to get the batch for.

		Returns:
		The triple (encoder_inputs, decoder_inputs, target_weights) for
		the constructed batch that has the proper format to call step(...) later.
	    """
		encoder_size, decoder_size = self.buckets[bucket_id]
		encoder_inputs, all_decoders_inputs = [], []

	    # Get a random batch of encoder and decoder inputs from data,
	    # pad them if needed, reverse encoder inputs and add GO to decoder.
		for _ in xrange(self.batch_size):
			encoder_input, decoders_inputs = random.choice(data[bucket_id]) 
			# Encoder inputs are padded and then reversed.
			encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
			encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

			# Decoder inputs get an extra "GO" symbol, and are padded then.
			decoders_input = []
			for decoder_input in decoders_inputs:
				decoder_pad_size = decoder_size - len(decoder_input) - 1
				decoders_input.append([data_utils.GO_ID] + decoder_input +
	                            [data_utils.PAD_ID] * decoder_pad_size)
			all_decoders_inputs.append(decoders_input)

		# Now we create batch-major vectors from the data selected above.
		batch_encoder_inputs, batch_all_decoders_inputs, batch_all_weights = [], [], []

	    # Batch encoder inputs are just re-indexed encoder_inputs.
		for length_idx in xrange(encoder_size):
			batch_encoder_inputs.append(
				np.array([encoder_inputs[batch_idx][length_idx]
	                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

		def get_one_decoder_inputs(column, decoders_list):
			return map(list,zip(*map(list, zip(*decoders_list))[column:column+1]))

		# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
		for column in xrange(num_decoders):
			batch_decoder_inputs, batch_weights = [], []
			decoder_inputs = get_one_decoder_inputs(column,all_decoders_inputs)
			
			for length_idx in xrange(decoder_size):
				batch_decoder_inputs.append(
	          	np.array([decoder_inputs[batch_idx][0][length_idx]
	                    	for batch_idx in xrange(self.batch_size)], dtype=np.int32))
				
				 # Create target_weights to be 0 for targets that are padding.
				batch_weight = np.ones(self.batch_size, dtype=np.float32)
				for batch_idx in xrange(self.batch_size):
					# We set weight to 0 if the corresponding target is a PAD symbol.
					# The corresponding target is decoder_input shifted by 1 forward.
					if length_idx < decoder_size - 1:
						target = decoder_inputs[batch_idx][0][length_idx + 1]
					if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
						batch_weight[batch_idx] = 0.0
				batch_weights.append(batch_weight)


			batch_all_decoders_inputs.append(batch_decoder_inputs)
			batch_all_weights.append(batch_weights)
		return batch_encoder_inputs, batch_all_decoders_inputs, batch_all_weights

