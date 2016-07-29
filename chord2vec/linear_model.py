import tensorflow as tf

import math
NUM_NOTES = 88
class LinearModel(object):
    """Linear Chord2vec model
    """
    def __init__(self, inputs, hidden_units):
        """Create the model

        """
        with tf.name_scope('hidden'):
            weights = tf.Variable(tf.truncated_normal([NUM_NOTES, hidden_units],
                                                      stddv=1.0/math.sqrt(float(NUM_NOTES))),
                                  name='weights');
            hidden = tf.nn.matmul(inputs, weights)
    def step(self, inputs, targets, forward_only = False):
        """

        Args:
            inputs:
            targets:
            forward_only:

        Returns:

        """