"""
Train a simple sequence2sequence model to learn the notes in the context
of an input chord
"""

import math
import os
import random
import sys
import time
import datetime
import numpy as np
from operator import add
import copy

import pickle
from six.moves import xrange
import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs


from tensorflow.models.rnn.translate import data_utils

from chord2vec import seq2seq_model
from chord2vec import seq2seqs_model

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("adam_epsilon", 1e-6,
                          "Epsilon used for numerical stability in Adam optimizer.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_units", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("notes_range", 109, "Number of notes in the vocabulary.")

tf.app.flags.DEFINE_boolean("attention", True, "Build sequence-to-sequence model with attention mechanism")
tf.app.flags.DEFINE_boolean("multiple_decoders", False, "Build sequence-to-sequences model")
tf.app.flags.DEFINE_integer("num_decoders", 2, "Number of decoders, i.e. number of context chords")

tf.app.flags.DEFINE_string("data_file", "reduced_JSB.pickle", "Data file name")
tf.app.flags.DEFINE_boolean("all_data_sets", False, "Uses all 4 data sets for training")

tf.app.flags.DEFINE_boolean("GD", False, "Uses Gradient Descent with adaptive learning rate")
tf.app.flags.DEFINE_string("train_dir", "unit1024layer2", "Training directory.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_valid_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Limit on the size of validation data (0: no limit).")
tf.app.flags.DEFINE_integer("max_epochs", 50,
                            "Maximium number of epochs for trainig.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("test_model", False,
                            "Evaluate an existing model on test data")

FLAGS = tf.app.flags.FLAGS

_buckets = [(4, 6)]


def read_data(file_name, context_size, full_context=False, training_data=True,
              valid_data=False, test_data=False):
    """"Load file_name and build (inputs, targets) pairs

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

    dataset = pickle.load(open(file_name,'rb'))
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
                empty_before = context_size - m_before
            elif size - i <= m_after:
                m_after = size - i - 1
                empty_after = context_size - m_after

            neighborhood = []

            for j in range(empty_before):
                neighborhood.append([])
            if (m_before > 0):
                neighborhood.extend(map(list, chords_seq[(i - m_before):i]))
            if (m_after > 0):
                neighborhood.extend(map(list, chords_seq[(i + 1):(i + m_after + 1)]))
            for j in range(empty_after):
                neighborhood.append([])

            for context_chord in neighborhood:
                context_chord.append(data_utils.EOS_ID)

            chord_and_context.append((list(chords_seq[i]), neighborhood))

            m_before = context_size
            m_after = context_size
            empty_after = 0
            empty_before = 0

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
            elif size - i <= m_after:
                m_after = size - i - 1

            if (m_before > 0):
                for context in map(list, chords_seq[(i - m_before):i]):
                    c_j = list(context)
                    c_j.append(data_utils.EOS_ID)
                    chord_and_context.append((list(chords_seq[i]), c_j))
            if (m_after > 0):

                for context in map(list, chords_seq[(i + 1):(i + m_after + 1)]):
                    c_j = list(context)
                    c_j.append(data_utils.EOS_ID)
                    chord_and_context.append((list(chords_seq[i]), c_j))

            m_before = context_size
            m_after = context_size

        return chord_and_context

    train_chords_and_contexts = []
    test_chords_and_contexts = []
    valid_chords_and_contexts = []

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
                        if min(chord) + t >= 21 and max(chord) + t <= 108:
                            augmented_data[-1].append(map(add, chord, [t] * len(chord)))

        return augmented_data

    theta = list(range(-6, 0))
    theta.extend(range(1, 6))
    augmented_data = augment_data(train_data, theta)

    for seq in augmented_data:
        if full_context:
            train_chords_and_contexts.extend(get_full_context(seq))
        else:
            train_chords_and_contexts.extend(get_contexts(seq))


    augmented_data = augment_data(valid_data, theta)

    for seq in augmented_data:
        if full_context:
            valid_chords_and_contexts.extend(get_full_context(seq))
        else:
            valid_chords_and_contexts.extend(get_contexts(seq))

    augmented_data = augment_data(test_data, theta)

    for seq in augmented_data:
        if full_context:
            test_chords_and_contexts.extend(get_full_context(seq))
        else:
            test_chords_and_contexts.extend(get_contexts(seq))

    return [train_chords_and_contexts], [valid_chords_and_contexts], [test_chords_and_contexts]

def read_all_data(context_size,full_context=False):
    files_dict = {}
    files_dict['jsb'] = 'JSB_Chorales.pickle'
    files_dict['piano'] = 'Piano-midi.de.pickle'
    files_dict['nottigham'] = 'Nottingham.pickle'
    files_dict['muse'] = 'MuseData.pickle'

    all_train, all_valid, all_test =  [],[],[]

    for d in files_dict:
        train, valid, test = read_data(files_dict[d], context_size,full_context=full_context)
        all_train.extend(train[0])
        all_valid.extend(valid[0])
        all_test.extend(test[0])

    return [all_train],[all_valid], [all_test]



def _get_max_seqLength(chords):
    """Gives the maximum chord length
	"""
    max_len = 0
    for chord_seq in chords:
        for note_seq in chord_seq:
            if max_len < len(note_seq):
                max_len = len(note_seq)
    return max_len


def create_seq2seqs_model(session, forward_only):
    """Create the model or load parameters in session """
    model = seq2seqs_model.Seq2SeqsModel(FLAGS.notes_range, _buckets, FLAGS.num_units,
                          FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.num_decoders, FLAGS.batch_size,
                          FLAGS.learning_rate,
                          FLAGS.learning_rate_decay_factor)

    checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
        print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
        model.saver.restore(session, checkpoint.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def create_seq2seq_model(session, forward_only, attention, result_file=None, batch_size=None, same_param=False):
    """Create the model or load parameters in session """
    if batch_size is None:
        batch_size = FLAGS.batch_size

    model = seq2seq_model.Seq2SeqModel(FLAGS.notes_range, FLAGS.notes_range, _buckets, FLAGS.num_units,
                         FLAGS.num_layers, FLAGS.max_gradient_norm, batch_size, FLAGS.learning_rate,
                         FLAGS.learning_rate_decay_factor,FLAGS.adam_epsilon,FLAGS.GD, forward_only=forward_only, attention=attention)

    if not same_param:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
            if result_file is not None:
                result_file.write("Continue training existing model! ")
            print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
            model.saver.restore(session, checkpoint.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
    return model

def _save_parameters(save):
    params_dic = {}

    params_dic["batch_size"] = FLAGS.batch_size
    params_dic["num_layers"] = FLAGS.num_layers
    params_dic["num_units"] = FLAGS.num_units
    params_dic["num_decoders"] = FLAGS.num_decoders
    params_dic["data_file"] = FLAGS.data_file
    params_dic["attention"] = FLAGS.attention
    params_dic["GD"] = FLAGS.GD
    params_dic["learning_rate"] = FLAGS.learning_rate
    params_dic["all_data_sets"] = FLAGS.all_data_sets
    params_dic["max_epochs"] = FLAGS.max_epochs
    params_dic["multiple_decoders"] = FLAGS.multiple_decoders
    params_dic["adam_epsilon"] = FLAGS.adam_epsilon
    params_dic["steps_per_checkpoint"] = FLAGS.steps_per_checkpoint
    params_dic["max_gradient_norm"] = FLAGS.max_gradient_norm
    params_dic["max_train_data_size"] = FLAGS.max_train_data_size
    params_dic["max_test_data_size"] = FLAGS.max_test_data_size
    params_dic["max_valid_data_size"] = FLAGS.max_valid_data_size
    params_dic["train_dir"] = FLAGS.train_dir

    if save:
        pickle.dump(params_dic, open(FLAGS.train_dir + '/params.pickle', 'wb'))
    return params_dic

def _check_dir(dir):
    """
    Checks if a directory exists; if not, creates a new directory and goes on.
     if it does exist; checks if it contains a trained model. If that is the case, print e message a quit.
     If it doesn't contain a trained model, either load existing parameters or
    Returns: True if the directory exists and contains a trained model, False otherwise.
    """

    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        if not os.path.isfile(dir+"/params.pickle"):
            _save_parameters(True)
        else:
            if _save_parameters(False) !=  pickle.load(open(dir+"/params.pickle",'rb')):
                raise ValueError("%s  directory contains a model trained with different parameters" % (dir))
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and os.path.isfile(dir + "/results.pickle"):
            raise IOError("A model trained with these parameters already exists!")

def train():
    """Train a model
	Args:
		multiple_decoders: if true, train the seq2seqs model, if false train original seq2seq,
							default is false
	"""

    with tf.Session() as sess:
        # Create model.
        with tf.variable_scope("model") as scope:
            _check_dir(FLAGS.train_dir)

            result_file = open(FLAGS.train_dir + "/results.txt", 'a+')

            if FLAGS.multiple_decoders:
                result_file.write("Creating sequence-to-sequences \n")
                print("Creating sequence-to-sequences ")
                print(" %d layers of %d units with %d decoders." % (FLAGS.num_layers, FLAGS.num_units, FLAGS.num_decoders))
                model = create_seq2seqs_model(sess, False)
                print("Reading test and raining data.")
                if FLAGS.all_data_sets:
                    train_set, valid_set, test_set = read_all_data(context_size=int(FLAGS.num_decoders / 2),
                                                               full_context=True)
                else:
                    train_set, valid_set, test_set = read_data(FLAGS.data_file, context_size=int(FLAGS.num_decoders/2), full_context=True)
            else:
                result_file.write("Creating sequence-to-sequences \n")
                print("Creating sequence-to-sequence ")
                if FLAGS.attention:
                    print("with attention mechanism")
                    result_file.write(("with attention mechanism \n"))
                print("Creating %d layers of %d units %d bach-size." % (FLAGS.num_layers, FLAGS.num_units, FLAGS.batch_size))


                model = create_seq2seq_model(sess, False, FLAGS.attention,result_file)
                print("Reading test and raining data.")
                if FLAGS.all_data_sets:
                    train_set, valid_set, test_set = read_all_data(context_size=int(FLAGS.num_decoders / 2))
                else:
                    train_set, valid_set, test_set = read_data(FLAGS.data_file, context_size=int(FLAGS.num_decoders/2))

            scope.reuse_variables()

            result_file.write("\n")
            result_file.write(str(datetime.datetime.now()))
            result_file.write("\n")

            random.shuffle(train_set[0])
            random.shuffle(test_set[0])
            random.shuffle(valid_set[0])

            if FLAGS.max_train_data_size:
                train_set[0] = train_set[0][:FLAGS.max_train_data_size]
            if FLAGS.max_valid_data_size:
                valid_set[0] = test_set[0][:FLAGS.max_valid_data_size]
            if FLAGS.max_test_data_size:
                test_set[0] = test_set[0][:FLAGS.max_test_data_size]

            result_file.write(
                " %d layers of %d units  %d context size %d bach-size. \n" %
                (FLAGS.num_layers, FLAGS.num_units, FLAGS.num_decoders, FLAGS.batch_size))


            # Training loop.
            MAX_STRIKES = 3
            step_time, ckpt_loss = 0.0, 0.0

            steps_per_epoch = int(len(train_set[0])/FLAGS.batch_size)
            current_step  = 0
            current_epoch = divmod(model.global_step.eval(),steps_per_epoch)[0]
            print("number of steps to complete one epoch %d" % steps_per_epoch)
            previous_losses,previous_train_ppx, previous_eval_ppx = [],[],[]
            best_train_loss, best_val_loss = np.inf, np.inf
            strikes = 0
            stop_training = False

            result_file.write(
                " %d batch size %d number of steps to complete one epoch \n" % (FLAGS.batch_size, steps_per_epoch))

            bucket_id = 0
            train_batch_id = 1
            num_valid_batches = int( len(valid_set[bucket_id]) / FLAGS.batch_size)

            checkpoint_path = FLAGS.train_dir + "/chords2vec.ckpt"

            result_dic = {}

            while (not stop_training) and int(model.global_step.eval()/steps_per_epoch) < FLAGS.max_epochs:
                # currently using only one bucket of size (max_seq_length, max_seq_length+2).

                # Get a batch and make a step.
                start_time = time.time()

                step_loss, _ = _get_batch_make_step(sess, model, FLAGS.multiple_decoders, train_set, FLAGS.num_decoders,
                                                    bucket_id, False)

                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                ckpt_loss += step_loss / FLAGS.steps_per_checkpoint

                current_step += 1
                train_batch_id +=1
                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(ckpt_loss) if ckpt_loss < 300 else float('inf')
                    print ("batch no %d epoch %d" % (train_batch_id,current_epoch))
                    print ("global step %d learning rate %.4f step-time %.2f loss %.2f  perplexity %.2f"
                           % (model.global_step.eval(), model.learning_rate.eval(),
                              step_time, ckpt_loss, perplexity))
                    result_file.write("global step %d learning rate %.4f step-time %.2f loss %.2f  perplexity %.2f \n"
                           % (model.global_step.eval(), model.learning_rate.eval(),
                              step_time, ckpt_loss, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if FLAGS.GD:
                        if len(previous_losses) > 2 and ckpt_loss > max(previous_losses[-3:]):
                            sess.run(model.learning_rate_decay_op)
                    previous_losses.append(ckpt_loss)
                    step_time, ckpt_loss = 0.0, 0.0


                if model.global_step.eval() % steps_per_epoch == 0:
                    print ("epoch  %d finished" % (current_epoch))
                    result_file.write("epoch  %d finished \n" % (current_epoch))
                    # Run evals on development set and print their perplexity.

                    _,train_loss, train_ppx = test_model("train_test",sess, train=True)
                    previous_train_ppx.append(train_ppx)
                    print("  train:  loss %.2f perplexity %.2f" % (train_loss, train_ppx))
                    result_file.write("  train:  loss %.4f perplexity %.4f \n" % (train_loss, train_ppx))

                    _,valid_loss, valid_ppx = test_model("eval_valid",sess, valid=True)

                    print("  eval:  loss %.2f perplexity %.2f" % (valid_loss, valid_ppx))
                    result_file.write("  eval:  loss %.4f perplexity %.4f \n" % (valid_loss, valid_ppx))

                    previous_eval_ppx.append(valid_ppx)

                    # Stopping criterion
                    margin = 0.001
                    improve_valid = previous_eval_ppx[-1] < best_val_loss + margin
                    if improve_valid:
                        strikes = 0
                        best_val_loss = previous_eval_ppx[-1]
                        # Save checkpoint.
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                    improve_train =  previous_train_ppx[-1] < best_train_loss + margin
                    if improve_train:
                        best_train_loss = previous_train_ppx[-1]

                    if improve_train and not improve_valid:
                        strikes +=1
                        print("strikes : %d" % strikes )
                        if strikes > MAX_STRIKES:
                            stop_training = True
                            print("Stopped training after %d epochs" % (int(model.global_step.eval()/steps_per_epoch)))
                            result_file.write("Stopped training after %d epochs\n" % int((model.global_step.eval() / steps_per_epoch)))

                    sys.stdout.flush()
                    train_batch_id = 1
                    current_epoch +=1
            if not stop_training:
                print("Reached the maximum number of epochs %d  stop trainig"  % (FLAGS.max_epochs))
                result_file.write("Reached the maximum number of epochs %d  stop trainig \n" % (FLAGS.max_epochs))

            print("best training  %.4f best validation %.4f \n" % (best_train_loss, best_val_loss) )
            result_file.write("best training loss %.4f best validation %.4f \n" % (best_train_loss, best_val_loss))

            # Print testing error:
            print("END of training")
            print("Model evaluation on test data...")

            _, test_loss, test_ppx = test_model("eval_test",sess, test=True)
            print("  test:  loss %.2f perplexity %.2f" % (test_loss, test_ppx))
            result_file.write("  test:  loss %.4f perplexity %.4f \n" % (test_loss, test_ppx))

            result_dic['test_ppx'] = test_ppx
            result_dic['train_losses'] = previous_losses
            result_dic['train_ppx'] = previous_train_ppx
            result_dic['valid_ppx'] = previous_eval_ppx
            result_dic['strikes'] = strikes
            result_dic['epoch'] = int(model.global_step.eval()/steps_per_epoch)

            pickle.dump(result_dic,open(FLAGS.train_dir + '/results.pickle','wb'))
            result_file.close()

def test_model(scope, sess, train=False, valid=False, test=False):
    train_set, valid_set, test_set = read_data(FLAGS.data_file, context_size=int(FLAGS.num_decoders / 2))
    data_set = []
    if train:
        data_set = train_set
    if valid:
        data_set = valid_set
    if test:
        data_set = test_set


    batch_size = len(data_set[0])
    model = create_seq2seq_model(sess, True,  attention=FLAGS.attention, result_file=None, batch_size=batch_size, same_param=True)

    for bucket_id in xrange(len(_buckets)):

        if len(valid_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
        encoder_final_state, loss, ppx = _get_test_batch_make_step(sess, model, FLAGS.multiple_decoders,
                                                   data_set, FLAGS.num_decoders, bucket_id, True,
                                                   batch_id=1)
    return encoder_final_state, loss,ppx


def get_vector_representation(data_point):
    with tf.Session() as sess:
        with tf.variable_scope("model") as scope:
            batch_size=1
            bucket_id =0
            model = create_seq2seq_model(sess, True, attention=FLAGS.attention, result_file=None, batch_size=batch_size)

            if len(data_point[bucket_id]) == 0:
                print("  eval: empty bucket %d" % (bucket_id))
            encoder_final_state, loss, ppx = _get_test_batch_make_step(sess, model, FLAGS.multiple_decoders,
                                                                       data_point, FLAGS.num_decoders, bucket_id, True,
                                                                       batch_id=1)
    print(encoder_final_state)
    return encoder_final_state



def _get_batch_make_step(sess, model, multiple_decoders, data_set, num_decoders, bucket_id, forward_only,
                         ):
    if multiple_decoders:

        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, num_decoders, bucket_id)

        encoder_final_state, loss, _ = model.step(sess, encoder_inputs, num_decoders,
                                  decoder_inputs, target_weights, bucket_id, forward_only)
    else:

        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, bucket_id)
        encoder_final_state, loss, _ = model.step(sess, encoder_inputs,
                                decoder_inputs, target_weights, bucket_id, forward_only)
    ppx = math.exp(loss) if loss < 300 else float('inf')

    if not forward_only:
        return loss, ppx
    else:
        return encoder_final_state, loss,ppx

def _get_test_batch_make_step(sess, model, multiple_decoders, data_set, num_decoders, bucket_id, forward_only,
                         batch_id=0):
    if multiple_decoders:

        encoder_inputs, decoder_inputs, target_weights = model.get_test_batch(
                data_set, num_decoders, bucket_id, batch_id)

        encoder_final_state, loss, _ = model.step(sess, encoder_inputs, num_decoders,
                                  decoder_inputs, target_weights, bucket_id, forward_only)
    else:

        encoder_inputs, decoder_inputs, target_weights = model.get_test_batch(
                data_set, bucket_id, batch_id)
        encoder_final_state, loss, _ = model.step(sess, encoder_inputs,
                                decoder_inputs, target_weights, bucket_id, forward_only)
    ppx = math.exp(loss) if loss < 300 else float('inf')
    if not forward_only:
        return loss, ppx
    else:
        return encoder_final_state, loss,ppx


def self_test():
    """Test the sequence-to-sequences model."""
    with tf.Session() as sess:
        print("Self-test for sequence-to-sequences model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seqs_model.Seq2SeqsModel(88, [(3, 3), (6, 6)], 32, 2,
                                            5.0, 2, 32, 0.3, 0.99)
        sess.run(tf.initialize_all_variables())

        # Fake data set for both the (3, 3) and (6, 6) bucket.

        data_set = ([([1, 1], [[2, 2], [4, 4]]), ([3, 3], [[4], [5]]), ([5], [[6], [6]])],
                    [([1, 1, 1, 1, 1], [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3]]), ([3, 3, 3], [[5, 6], [6, 7]])])

        num_decoders = 2

        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, 2, bucket_id)
            model.step(sess, encoder_inputs, num_decoders, decoder_inputs, target_weights,
                       bucket_id, False)


def main(_):
    if FLAGS.test_model:
        test_chord = [[([60, 72, 79, 88, 2],[72, 79, 88, 2])]]
        get_vector_representation(test_chord)
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
