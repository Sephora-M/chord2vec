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

def read_data(file_name='Piano-midi.de.pickle', context_size=2):
	""""Load pickled piano-roll file from file_name and build
		(inputs, targets) pairs

		Args:
			file_name: path to the pickled piano-roll file
			context_size: the size of the context (number of preceeding and succeeding chords)

		Returns:
			data_set: two lists (inputs, targets) where inputs is a list of chords and targets a
			list of corresponding context chords 

	"""
	dataset = cPickle.load(file(file_name))
	training_data = dataset['train']

	def get_contexts(chords_seq):
		"""Gives the context of each chord in the list chords_seq
		"""
		chord = []
		context = []
		m_before = context_size
		m_after = context_size
		size = len(chords_seq)
		
		for i in range(size):
			# the neighborhood of chords at the beginning or at the end of a sequence is smaller
			if i < m_before:
				m_before = i
			elif size-i <= m_after:
				m_after = size-i-1

			neighborhood = []
			if(m_before > 0):
				neighborhood.append(chords_seq[(i-m_before):i])
			if(m_after > 0):
				neighborhood.append(chords_seq[(i+1):(i+m_after+1)]) 

			chord.append(chords_seq[i])
			context.append(neighborhood)

			m_before = context_size
			m_after = context_size	

		return (chord,context)

	inputs = []
	targets = []

	for seq in training_data:
		chords,contexts = get_contexts(seq)
		inputs.append(chords)
		targets.append(contexts) 

	return inputs,targets

def get_max_seqLength(chords):
	max_len = 0
	for chord_seq in chords:
		for note_seq in chord_seq:
			if max_len < len(note_seq):
				max_len = len(note_seq)
	return max_len


def create_model():
	# TODO
	return 0

def train():
	# TODO
	return 0

def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for sequence-to-sequences model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.SimpleSeq2SeqModel(88,[(3, 3), (6, 6)], 32, 2,
                                       5.0,2, 32, 0.3, 0.99)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [[2, 2], [1,2]]), ([3, 3], [[4],[4]]), ([5], [[6],[5]])],
                [([1, 1, 1, 1, 1], [[2, 2, 2, 2, 2],[2, 2, 2, 3, 3]]), ([3, 3, 3], [[5, 6],[7,6]])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, 2,bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)

def main(_):
   self_test()

if __name__ == "__main__":
  tf.app.run()