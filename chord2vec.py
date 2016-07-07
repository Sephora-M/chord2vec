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
from tensorflow.models.rnn import rnn_cell, seq2seq

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

def create_model():
	# TODO
	return 0

def train():
	# TODO
	return 0

def main(_):
   train()

if __name__ == "__main__":
  tf.app.run()