"""
Train a simple sequence2sequence model to learn the notes in the context
of an input chord
"""

import math
import os
import random
import sys
import time
import numpy as np
from operator import add
import copy

import cPickle
import tensorflow as tf

from tensorflow.models.rnn.translate import data_utils
import seq2seq_model
import seq2seqs_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("notes_range", 109, "Number of notes in the vocabulary.")
tf.app.flags.DEFINE_integer("num_decoders", 2,"Number of decoders, i.e. number of context chords")
tf.app.flags.DEFINE_string("data_dir", "save_network", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "save_network", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5,5)]

def read_data(file_name='JSB_Chorales.pickle', context_size=1, full_context=False, training_data=True, 
	valid_data=False, test_data = False):
	""""Load pickled piano-roll file from file_name and build
		(inputs, targets) pairs

		Args:
			file_name: path to the pickled piano-roll file
			context_size: the size of the context (number of preceeding and succeeding chords)
			full_context: if true, a training pair has the format (input, [output1,...,output_d]) where d=context_size*2
				if false, return multiple traing pairs with the same input : (input, output1),...,(input, output_d)
			training_data: true if we need to read the training data.
			valid_data: true if we need to read the validation data.
			test_data: true if we need to read the test data. Only one of training_data, valid_data and test_data can be true
			

		Returns:
			data_set: a list of pairs (input, output) = (chord, context chord(s)) 

	"""
	if not training_data ^ valid_data ^ test_data or training_data & valid_data & test_data:
		raise ValueError("Only one of training_data, valid_data and test_data can be True")

	dataset = cPickle.load(file(file_name))
	train_data = dataset['train']
	valid_data = dataset['valid']
	test_data = dataset['test']
 
	def get_full_context(chords_seq):
		"""Gives the context of each chord in the list chords_seq

		Args: 
			chords_seq: a list of sequences of chords 
		Returns:
			chord_and_context: a list of pairs (chord, [contexts]) for each chord in chords_seq
		"""
		chord_and_context = []
		m_before = context_size
		empty_before = 0 
		m_after = context_size
		empty_after = 0

		size = len(chords_seq)
		
		for i in range(size):
			# the neighborhood of chords at the beginning or at the end of a sequence is smaller
			if i < m_before:
				m_before = i
				empty_before = context_size-m_before
			elif size-i <= m_after:
				m_after = size-i-1
				empty_after = context_size-m_after

			neighborhood = []

			for j in range(empty_before):
					neighborhood.append([])
			if(m_before > 0):
				neighborhood.extend(map(list, chords_seq[(i-m_before):i]))
			if(m_after > 0):
				neighborhood.extend(map(list, chords_seq[(i+1):(i+m_after+1)])) 
			for j in range(empty_after):
					neighborhood.append([])

			chord_and_context.append((list(chords_seq[i]),neighborhood))

			m_before = context_size
			m_after = context_size	
			empty_after = 0
			empty_before =0

		return chord_and_context

	def get_contexts(chords_seq):
		"""Gives the context of each chord in the list chords_seq

		Args: 
			chords_seq: a list of sequences of chords 
		Returns:
			chord_and_context: a list of pairs (chord, context_1),...,(chord, context_d) for each 
			chord in chords_seq. d in -context_size, ..., -1, 1, ... context_size
		"""
		chord_and_context = []
		m_before = context_size
		m_after = context_size

		size = len(chords_seq)
		
		for i in range(size):
			# the neighborhood of chords at the beginning or at the end of a sequence is smaller
			if i < m_before:
				m_before = i
			elif size-i <= m_after:
				m_after = size-i-1

			if(m_before > 0):
				for context in map(list, chords_seq[(i-m_before):i]):
					c_j = list(context)
					c_j.append(data_utils.EOS_ID)
					chord_and_context.append( (list(chords_seq[i]), c_j) )
			if(m_after > 0):
				for context in map(list, chords_seq[(i+1):(i+m_after+1)]):
					c_j = list(context)
					c_j.append(data_utils.EOS_ID)
					chord_and_context.append( (list(chords_seq[i]), c_j) )

			m_before = context_size
			m_after = context_size	

		return chord_and_context

	chords_and_contexts = []

	if training_data:
		data = train_data
	elif valid_data:
		data = valid_data
	else:
		data = test_data

	def augment_data(data, theta):
		"""Augment the data by applying to each data point d the transformation 
		d + theta_i for each theta_i in theta
		"""
		augmented_data = copy.deepcopy(data)

		for s in data:
			for t in theta:
				augmented_data.append([])
				for chord in s:
					if chord:
						if min(chord)+t >= 21 and max(chord)+t <= 108:
							augmented_data[-1].append(map(add,chord, [t]*len(chord)))

		return augmented_data

	theta = range(-6,0)
	theta.extend(range(1,6))
	augmented_data = augment_data(data, theta)

	for seq in augmented_data:
		if full_context:
			chords_and_contexts.extend(get_full_context(seq))
		else:
			chords_and_contexts.extend(get_contexts(seq))

	return [chords_and_contexts],augmented_data,data

def _get_max_seqLength(chords):
	"""Gives the maximum chord length 
	"""
	max_len = 0
	for chord_seq in chords:
		for note_seq in chord_seq:
			if max_len < len(note_seq):
				max_len = len(note_seq)
	return max_len


def create_seq2seqs_model(session,forward_only):
	"""Create the model or load parameters in session """
	model = seq2seqs_model.Seq2SeqsModel(FLAGS.notes_range, _buckets, FLAGS.num_units, 
		FLAGS.num_layers, FLAGS.max_gradient_norm,FLAGS.num_decoders,FLAGS.batch_size, FLAGS.learning_rate,
		FLAGS.learning_rate_decay_factor)

	checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir+"2")
	if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
		print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
		model.saver.restore(session, checkpoint.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
	return model


def create_seq2seq_model(session,forward_only):
	"""Create the model or load parameters in session """
	model = seq2seq_model.Seq2SeqModel(FLAGS.notes_range,FLAGS.notes_range, _buckets, FLAGS.num_units, 
		FLAGS.num_layers, FLAGS.max_gradient_norm,FLAGS.batch_size, FLAGS.learning_rate,
		FLAGS.learning_rate_decay_factor,forward_only=forward_only)


	checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir+"1")
	if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
		print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
		model.saver.restore(session, checkpoint.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
	return model


def train(multiple_decoders = False):
	"""Train a model 
	Args:
		multiple_decoders: if true, trian the seq2seqs model, if false train original seq2seq,
							default is false
	"""
	
	with tf.Session() as sess:
		# Create model.
		print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.num_units))
		
		if multiple_decoders:
			model = create_seq2seqs_model(sess, False)
			print("Reading test and raining data." )
			test_set,_,_ = read_data(full_context=True, training_data=False, test_data=True)
			valid_set,_,_ = read_data(full_context=True, training_data=False, valid_data=True)
			train_set,_,_ = read_data(full_context=True, training_data=True)
		else:
			model = create_seq2seq_model(sess,False)
			print("Reading test and raining data." )
			test_set,_,_ = read_data(training_data=False, test_data=True)
			valid_set,_,_ = read_data(training_data=False, valid_data=True)
			train_set,_,_ = read_data(training_data=True)
		
		
		
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
		train_total_size = float(sum(train_bucket_sizes))

		# Training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		previous_eval_losses = []
		consecutive_loss_increase=0
		stop_training = False

		while not stop_training:
			# currently using only one bucket of size (max_seq_length, max_seq_length).
			bucket_id=0

			# Get a batch and make a step.
			start_time = time.time()

			step_loss,_ = _get_batch_make_step(sess, model, multiple_decoders, train_set, FLAGS.num_decoders,bucket_id,False)

			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step +=1

			# Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step % FLAGS.steps_per_checkpoint == 0:
		        # Print statistics for the previous epoch.
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				print ("global step %d learning rate %.4f step-time %.2f loss %.2f  perplexity %.2f"
               		 % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, loss, perplexity))
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				
				previous_losses.append(loss)
				# Save checkpoint and zero timer and loss.
				if multiple_decoders:
					checkpoint_path =  FLAGS.train_dir + "2/chords2vec.ckpt"
				else:
					checkpoint_path =  FLAGS.train_dir + "1/chords2vec.ckpt"
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
				# Run evals on development set and print their perplexity.
				for bucket_id in xrange(len(_buckets)):
					if len(valid_set[bucket_id]) == 0:
						print("  eval: empty bucket %d" % (bucket_id))
					 	continue
					eval_loss, eval_ppx = _get_batch_make_step(sess, model, multiple_decoders, 
						valid_set, FLAGS.num_decoders,bucket_id,True)
					print("  eval:  loss %.2f perplexity %.2f" % ( eval_loss, eval_ppx))
					previous_eval_losses.append(eval_loss)
					if len(previous_eval_losses)>1 and previous_eval_losses[-1] > previous_eval_losses[-2]:
						consecutive_loss_increase +=1
					else:
						consecutive_loss_increase=0
				# Test if overfitting : if the evaluation loss kept increasing the past 6  evaluation steps.	
				if consecutive_loss_increase > 6:
					stop_training = True 
				sys.stdout.flush()
		# Print testing error:
		print("END of training")
		print("Model evaluation on test data...")
		for bucket_id in xrange(len(_buckets)):
			if len(valid_set[bucket_id]) == 0:
				print("  eval: empty bucket %d" % (bucket_id))
				continue
			test_loss, test_ppx = _get_batch_make_step(sess,model, multiple_decoders, 
				test_set, FLAGS.num_decoders, bucket_id, True)
			print("		test: loss %.4f  perplexity %.4f " % (test_loss, test_ppx) )


def _get_batch_make_step(sess,model, multiple_decoders, data_set, num_decoders,bucket_id,forward_only):
	if multiple_decoders:
		encoder_inputs, decoder_inputs, target_weights = model.get_batch(
							data_set,num_decoders, bucket_id)
		_, loss, _ = model.step(sess, encoder_inputs,num_decoders, 
							decoder_inputs, target_weights, bucket_id, forward_only)
	else:
		encoder_inputs, decoder_inputs, target_weights = model.get_batch(
							data_set, bucket_id)
		_, loss, _ = model.step(sess, encoder_inputs, 
							decoder_inputs, target_weights, bucket_id, forward_only)
		ppx = math.exp(loss) if loss < 300 else float('inf')
	return loss, ppx

def self_test():
  """Test the sequence-to-sequences model."""
  with tf.Session() as sess:
    print("Self-test for sequence-to-sequences model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqsModel(88,[(3, 3), (6, 6)], 32, 2,
                                       5.0,2, 32, 0.3, 0.99)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.

    data_set = ([([1, 1], [[2, 2],[4,4]]), ([3, 3], [[4],[5]]), ([5], [[6],[6]])],
                [([1, 1, 1, 1, 1], [[2, 2, 2, 2, 2],[3,3,3,3,3]]), ([3, 3, 3], [[5, 6],[6,7]] )])

    num_decoders=2 

    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, 2,bucket_id)
      model.step(sess, encoder_inputs, num_decoders, decoder_inputs, target_weights,
                 bucket_id, False)

def main(_):
   self_test()


if __name__ == "__main__":
  tf.app.run()