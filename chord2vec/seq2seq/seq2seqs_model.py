"""
Simple sequence-to-sequence model 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import zip     # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

import math
import os
import random
import sys
import time
import numpy as np
from six.moves import xrange


from tensorflow.models.rnn.translate import data_utils


class Seq2SeqsModel:
	"""A sequence-to-sequence model. It implements a
	(multi-layer) RNN encoder followed by a (multi-layer) 
	RNN  decoder. Encoder and decoder use the same RNN cell type,
	but don't share parameters.

	EDIT: now using buckets
	"""

	def __init__(self, vocab_size,buckets, num_units, num_layers,
		max_gradient_norm, num_decoders, batch_size,
		learning_rate, learning_rate_decay_factor, forward_only=False):
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
		single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
		cell = single_cell
		if num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)



		self.outputs, self.losses = model_with_buckets(self.encoder_inputs,
			num_decoders,self.all_decoder_inputs, all_targets, self.all_targets_weights,
			buckets, lambda x,y: embedding_rnn_seq2seqs(x,num_decoders,y,cell,
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
				(gradient norm, loss, output)
		"""
		# TODO  Check if the sizes match.
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


	def get_batch(self, data, num_decoders, bucket_id,batch_id):
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
		for data_point in data[bucket_id][(batch_id*self.batch_size-self.batch_size):(batch_id*self.batch_size)]:
			encoder_input, decoders_inputs = data_point
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
			return list( map(list,zip(* list(map(list, zip(*decoders_list)))[column:column+1])) )

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

	def get_test_batch(self, data, num_decoders, bucket_id):
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
		test_size = len(data[bucket_id])

	    # Get a random batch of encoder and decoder inputs from data,
	    # pad them if needed, reverse encoder inputs and add GO to decoder.
		for data_point in data[bucket_id]:
			encoder_input, decoders_inputs = data_point
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
	                    for batch_idx in xrange(test_size)], dtype=np.int32))

		def get_one_decoder_inputs(column, decoders_list):
			return map(list,zip(*map(list, zip(*decoders_list))[column:column+1]))

		# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
		for column in xrange(num_decoders):
			batch_decoder_inputs, batch_weights = [], []
			decoder_inputs = get_one_decoder_inputs(column,all_decoders_inputs)

			for length_idx in xrange(decoder_size):
				batch_decoder_inputs.append(
	          	np.array([decoder_inputs[batch_idx][0][length_idx]
	                    	for batch_idx in xrange(test_size)], dtype=np.int32))

				 # Create target_weights to be 0 for targets that are padding.
				batch_weight = np.ones(test_size, dtype=np.float32)
				for batch_idx in xrange(test_size):
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


def embedding_rnn_decoders(num_decoders, all_decoder_inputs, initial_state, cell, num_symbols,
						   embedding_size, output_projection=None,
						   feed_previous=False,
						   update_embedding_for_previous=True, scope=None):
	"""RNN decoder with embedding and a pure-decoding option.

	Args:
	num_decoders: Integer; number of sequences to output
	all_decoder_inputs: A list num_decoders lists of 1D batch-sized int32 Tensors (decoder inputs).
	initial_state: 2D Tensor [batch_size x cell.state_size].
	cell: rnn_cell.RNNCell defining the cell function.
	num_symbols: Integer, how many symbols come into the embedding.
	embedding_size: Integer, the length of the embedding vector for each symbol.
	output_projection: None or a pair (W, B) of output projection weights and
	biases; W has shape [output_size x num_symbols] and B has
	shape [num_symbols]; if provided and feed_previous=True, each fed
	previous output will first be multiplied by W and added B.
	feed_previous: Boolean; if True, only the first of decoder_inputs will be
	used (the "GO" symbol), and all other decoder inputs will be generated by:
	next = embedding_lookup(embedding, argmax(previous_output)),
	In effect, this implements a greedy decoder. It can also be used
	during training to emulate http://arxiv.org/abs/1506.03099.
	If False, decoder_inputs are used as given (the standard decoder case).
	update_embedding_for_previous: Boolean; if False and feed_previous=True,
	only the embedding for the first symbol of decoder_inputs (the "GO"
		symbol) will be updated by back propagation. Embeddings for the symbols
	generated from the decoder itself remain unchanged. This parameter has
	no effect if feed_previous=False.
	scope: VariableScope for the created subgraph; defaults to
	"embedding_rnn_decoder".

	Returns:
	A tuple of the form (outputs, state), where:
	outputs: A list of the same length as decoder_inputs of 2D Tensors with
	shape [batch_size x output_size] containing the generated outputs.
	state: The state of each decoder cell in each time-step. This is a list
	with length len(decoder_inputs) -- one item for each time-step.
	It is a 2D Tensor of shape [batch_size x cell.state_size].

	Raises:
	ValueError: When output_projection has the wrong shape.
	"""
	if output_projection is not None:
		proj_weights = ops.convert_to_tensor(output_projection[0],
											 dtype=dtypes.float32)
		proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
		proj_biases = ops.convert_to_tensor(
			output_projection[1], dtype=dtypes.float32)
		proj_biases.get_shape().assert_is_compatible_with([num_symbols])

	with variable_scope.variable_scope(scope or "embedding_rnn_decoder"):
		embedding = variable_scope.get_variable("embedding",
												[num_symbols, embedding_size])
		loop_function = _extract_argmax_and_embed(embedding, output_projection,
												  update_embedding_for_previous) if feed_previous else None
		all_outputs = []
		all_states = []
		d = 0
		for decoder_inputs in all_decoder_inputs:
			emb_inp = (embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs)
			outputs, states = tf.nn.seq2seq.rnn_decoder(emb_inp, initial_state, cell,
														loop_function=loop_function, scope="rnn_decoder{0}".format(d))
			all_outputs.append(outputs)
			all_states.append(states)
			d += 1
		return all_outputs, all_states


def embedding_rnn_seq2seqs(encoder_inputs, num_decoders, all_decoder_inputs, cell,
						   num_encoder_symbols, num_decoder_symbols,
						   embedding_size, output_projection=None,
						   feed_previous=False, dtype=dtypes.float32,
						   scope=None):
	"""Embedding RNN sequence-to-sequences model.

	This model first embeds encoder_inputs by a newly created embedding (of shape
		[num_encoder_symbols x input_size]). Then it runs an RNN to encode
	embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
	by another newly created embedding (of shape [num_decoder_symbols x
		input_size]). Then it runs RNN decoder, initialized with the last
	encoder state, on embedded decoder_inputs.

	Args:
	encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
	num_decoders: Integer; number of sequences to output
	decoder_inputs: A list num_decoder lists of 1D int32 Tensors of shape [batch_size].
	cell: rnn_cell.RNNCell defining the cell function and size.
	num_encoder_symbols: Integer; number of symbols on the encoder side.
	num_decoder_symbols: Integer; number of symbols on the decoder side.
	embedding_size: Integer, the length of the embedding vector for each symbol.
	output_projection: None or a pair (W, B) of output projection weights and
	biases; W has shape [output_size x num_decoder_symbols] and B has
	shape [num_decoder_symbols]; if provided and feed_previous=True, each
	fed previous output will first be multiplied by W and added B.
	feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
	of decoder_inputs will be used (the "GO" symbol), and all other decoder
	inputs will be taken from previous outputs (as in embedding_rnn_decoder).
	If False, decoder_inputs are used as given (the standard decoder case).
	dtype: The dtype of the initial state for both the encoder and encoder
	rnn cells (default: tf.float32).
	scope: VariableScope for the created subgraph; defaults to
	"embedding_rnn_seq2seq"

	Returns:
	Tuples of the form (outputs, state), where:
	outputs: A list of the same length as decoder_inputs of 2D Tensors with
	shape [batch_size x num_decoder_symbols] containing the generated
	outputs.
	state: The state of each decoder cell in each time-step. This is a list
	with length len(decoder_inputs) -- one item for each time-step.
	It is a 2D Tensor of shape [batch_size x cell.state_size].
	"""
	with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq"):
		# Encoder.
		encoder_cell = rnn_cell.EmbeddingWrapper(
			cell, embedding_classes=num_encoder_symbols,
			embedding_size=embedding_size)
		_, encoder_state = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtype)

		# Decoder.
		if output_projection is None:
			cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

			if isinstance(feed_previous, bool):
				return embedding_rnn_decoders(num_decoders,
											  all_decoder_inputs, encoder_state, cell, num_decoder_symbols,
											  embedding_size, output_projection=output_projection,
											  feed_previous=feed_previous)

		# If feed_previous is a Tensor, we construct 2 graphs and use cond.
		def decoders(feed_previous_bool):
			reuse = None if feed_previous_bool else True
			with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=reuse):
				all_outputs, all_states = embedding_rnn_decoders(num_decoders,
																 all_decoder_inputs, encoder_state, cell,
																 num_decoder_symbols,
																 embedding_size, output_projection=output_projection,
																 feed_previous=feed_previous_bool,
																 update_embedding_for_previous=False)
				return all_outputs + [all_states]

		all_outputs_and_states = control_flow_ops.cond(feed_previous, lambda: decoders(True), lambda: decoders(False))
		return all_outputs_and_states[:-1], all_outputs_and_states[-1]


def sequences_loss(logits, targets, weights, num_decoders,
				   average_across_timesteps=True, average_across_batch=True,
				   softmax_loss_function=None, name=None):
	"""Product of weighted cross-entropy loss for sequences of logits, batch-collapsed.

	Args:
	logits: Lists of 2D Tensors of shape [batch_size x num_decoder_symbols] of size num_decoders.
	targets: Lists of 1D batch-sized int32 Tensors of the same lengths as logits.
	weights: List of 1D batch-sized float-Tensors of the same length as logits.
	average_across_timesteps: If set, divide the returned cost by the total
	label weight.
	average_across_batch: If set, divide the returned cost by the batch size.
	softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
	to be used instead of the standard softmax (the default if this is None).
	name: Optional name for this operation, defaults to "sequence_loss".

	Returns:
	A scalar float Tensor: The products of average log-perplexities per symbol (weighted).

	Raises:
	ValueError: If len(logits) is different from len(targets) or len(weights).
	"""
	if len(targets) != len(logits) or num_decoders != len(logits):
		raise ValueError("Lengths of logits and targets must be %d, not "
						 "%d, %d." % (num_decoders, len(logits), len(targets)))
	losses = []
	for i in xrange(num_decoders):
		temp = tf.nn.seq2seq.sequence_loss(logits[i], targets[i], weights[i],
												  average_across_timesteps, average_across_batch, softmax_loss_function,
												  name)
		temp=tf.reshape(temp,[1,1])
		losses.append(temp)
	losses = tf.reshape(tf.concat(1, losses), [-1, num_decoders])
	return math_ops.reduce_prod(losses)


def model_with_buckets(encoder_inputs, num_decoders, all_decoders_inputs, all_targets, weights,
					   buckets, seq2seq, softmax_loss_function=None,
					   per_example_loss=False, name=None):
	"""Create a sequence-to-sequence model with support for bucketing.

	The seq2seq argument is a function that defines a sequence-to-sequence model,
	e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

	Args:
	encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
	decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
	targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
	weights: List of 1D batch-sized float-Tensors to weight the targets.
	buckets: A list of pairs of (input size, output size) for each bucket.
	seq2seq: A sequence-to-sequence model function; it takes 2 input that
	agree with encoder_inputs and decoder_inputs, and returns a pair
	consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
	softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
	to be used instead of the standard softmax (the default if this is None).
	per_example_loss: Boolean. If set, the returned loss will be a batch-sized
	tensor of losses for each sequence in the batch. If unset, it will be
	a scalar with the averaged loss from all examples.
	name: Optional name for this operation, defaults to "model_with_buckets".

	Returns:
	A tuple of the form (outputs, losses), where:
	outputs: The outputs for each bucket. Its j'th element consists of a list
	of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
	losses: List of scalar Tensors, representing losses for each bucket, or,
	if per_example_loss is set, a list of 1D batch-sized float Tensors.

	Raises:
	ValueError: If length of encoder_inputsut, targets, or weights is smaller
	than the largest (last) bucket.
	"""
	if len(encoder_inputs) < buckets[-1][0]:
		raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
						 "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
	# TODO: must check all target and weiths, not just first elem
	if len(all_targets[0]) < buckets[-1][1]:
		raise ValueError("Length of targets (%d) must be at least that of last"
						 "bucket (%d)." % (len(targets), buckets[-1][1]))
	if len(weights[0]) < buckets[-1][1]:
		raise ValueError("Length of weights (%d) must be at least that of last"
						 "bucket (%d)." % (len(weights), buckets[-1][1]))

	def bucket_decoders_inputs(b, decoders_list):
		return list( map(list, zip(*list(map(list, zip(*decoders_list)))[:b])))

	all_inputs = encoder_inputs + all_decoders_inputs + all_targets + weights
	losses = []
	outputs = []
	with ops.op_scope(all_inputs, name, "model_with_buckets"):
		for j, bucket in enumerate(buckets):
			with variable_scope.variable_scope(variable_scope.get_variable_scope(),
											   reuse=True if j > 0 else None):
				decoders_inputs = bucket_decoders_inputs(bucket[1], all_decoders_inputs)
				bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
											decoders_inputs)
				outputs.append(bucket_outputs)

				bucket_targets = bucket_decoders_inputs(bucket[1], all_targets)
				bucket_weights = bucket_decoders_inputs(bucket[1], weights)
				if per_example_loss:

					losses.append(sequence_loss_by_example(
						outputs[-1], bucket_targets, bucket_weights, num_decoders,
						softmax_loss_function=softmax_loss_function))
				else:
					losses.append(sequences_loss(
						outputs[-1], bucket_targets, bucket_weights, num_decoders,
						softmax_loss_function=softmax_loss_function))
	return outputs[0], losses