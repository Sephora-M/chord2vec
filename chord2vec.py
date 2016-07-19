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

import cPickle
import tensorflow as tf

import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("notes_range", 200, "Number of notes in the vocabulary.")
tf.app.flags.DEFINE_integer("num_decoders", 2,"Number of decoders, i.e. number of context chords")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 20,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(12,12)]

def read_data(file_name='Piano-midi.de.pickle', context_size=1, training_data=True):
	""""Load pickled piano-roll file from file_name and build
		(inputs, targets) pairs

		Args:
			file_name: path to the pickled piano-roll file
			context_size: the size of the context (number of preceeding and succeeding chords)
			training_data: true if we need to read the training data. If false, the test data is used

		Returns:
			data_set: a list containing a pair (inputs, targets) where inputs is a list of chords and targets a
			list of corresponding context chords 

	"""
	dataset = cPickle.load(file(file_name))
	train_data = dataset['train']
	test_data = dataset['test']
 
	def get_contexts(chords_seq):
		"""Gives the context of each chord in the list chords_seq
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
				neighborhood.extend(chords_seq[(i-m_before):i])
			if(m_after > 0):
				neighborhood.extend(chords_seq[(i+1):(i+m_after+1)]) 
			for j in range(empty_after):
					neighborhood.append([])

			chord_and_context.append((chords_seq[i],neighborhood))

			m_before = context_size
			m_after = context_size	
			empty_after = 0
			empty_before =0

		return chord_and_context

	chords_and_contexts = []

	if training_data:
		data = train_data
	else:
		data = test_data

	for seq in data:
		chords_and_contexts.append(get_contexts(seq))

	return chords_and_contexts

def get_max_seqLength(chords):
	max_len = 0
	for chord_seq in chords:
		for note_seq in chord_seq:
			if max_len < len(note_seq):
				max_len = len(note_seq)
	return max_len


def create_model(session):
	"""Create the model or load parameters in session """
	model = seq2seq_model.SimpleSeq2SeqModel(FLAGS.notes_range, _buckets, FLAGS.num_units, 
		FLAGS.num_layers, FLAGS.max_gradient_norm,FLAGS.num_decoders,FLAGS.batch_size, FLAGS.learning_rate,
		FLAGS.learning_rate_decay_factor)

	checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
		print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
		model.saver.restore(session, checkpoint.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.initialize_all_variables())
	return model


def train():
	
	with tf.Session() as sess:
		# Create model.
		print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.num_units))
		model = create_model(sess)
		
		print("Reading test and raining data." )
		test_set = read_data(training_data=False)
		train_set = read_data(training_data=True)
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
		train_total_size = float(sum(train_bucket_sizes))

		# Training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []

		while True:
			# currently using only one bucket of size (max_seq_length, max_seq_length).
			bucket_id=0

			# Get a batch and make a step.
			start_time = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, 
				FLAGS.num_decoders,bucket_id)
			_, step_loss, _ = model.step(sess, encoder_inputs, FLAGS.num_decoders, decoder_inputs, 
				target_weights, bucket_id, False)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step +=1

			# Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step % FLAGS.steps_per_checkpoint == 0:
		        # Print statistics for the previous epoch.
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				print ("global step %d learning rate %.4f step-time %.2f perplexity "
               		"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				
				previous_losses.append(loss)
				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(FLAGS.train_dir, "chords2vec.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
				# Run evals on development set and print their perplexity.
				for bucket_id in xrange(len(_buckets)):
					if len(test_set[bucket_id]) == 0:
						print("  eval: empty bucket %d" % (bucket_id))
						continue
					encoder_inputs, decoder_inputs, target_weights = model.get_batch(
						test_set,FLAGS.num_decoders, bucket_id)
					_, eval_loss, _ = model.step(sess, encoder_inputs,FLAGS.num_decoders, 
						decoder_inputs, target_weights, bucket_id, True)
					eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
					print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
				sys.stdout.flush()

def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for sequence-to-sequences model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.SimpleSeq2SeqModel(88,[(3, 3), (6, 6)], 32, 2,
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