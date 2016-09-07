'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from chord2vec.linear_models import data_processing as dp
import numpy as np
import random
import sys

# print('Loading data ...')
# dic=pickle.load(open('JSB_processed.pkl','rb'))
# train_chords = dic['t']
# test_chords = dic['te']
# valid_chords = dic['v']
#
# train_set = dp.generate_binary_vectors(train_chords)
# data_size = len(train_set[0])
# test_set = dp.generate_binary_vectors(test_chords)
# valid_set = dp.generate_binary_vectors(valid_chords)
# input_valid, target_valid = valid_set


def load_data(file_name = "JSB_Chorales.pickle"):
    print('Loading data ...')
    train_chords, test_chords , valid_chords = dp.read_data(file_name,1)

    train_set = dp.generate_binary_vectors(train_chords)
    # input_train, target_train = train_set
    test_set = dp.generate_binary_vectors(test_chords)
    valid_set = dp.generate_binary_vectors(valid_chords)
    # input_valid, target_valid = valid_set

    data_size = len(train_set[0])
    data_size_valid = len(valid_set[0])
    data_size_te = len(test_set[0])

    total_batch = int(data_size / batch_size)
    total_batch_valid = int(data_size_valid / batch_size)
    total_batch_test = int(data_size_te / batch_size)

    return train_set, test_set, valid_set, total_batch, total_batch_test, total_batch_valid

#train_set = [[[0, 0, 0, 1,1, 0, 1, 0], [1, 1, 1, 1,0, 0, 1, 1], [1, 1, 1, 0,0, 1, 0, 1], [0, 0, 0, 1,0, 0, 0, 1], [1, 0, 0, 0,1, 0, 0, 1]], \
#           [[1, 1, 0, 1,0, 1, 1, 0], [0, 0, 1, 1,1, 1, 1, 1], [0, 0, 1, 0,1, 0, 0, 1], [1, 1, 0, 1,1, 1, 0,1], [0, 1, 0, 0,0, 1, 0, 1]]]
#
#test_set = [[[0, 0, 0, 1,1, 0, 1, 0], [1, 1, 1, 1,0, 0, 1, 1], [1, 1, 1, 0,0, 1, 0, 1], [0, 0, 0, 1,0, 0, 0, 1], [1, 0, 0, 0,1, 0, 0, 1]], \
#           [[1, 1, 0, 1,0, 1, 1, 0], [0, 0, 1, 1,1, 1, 1, 1], [0, 0, 1, 0,1, 0, 0, 1], [1, 1, 0, 1,1, 1, 0,1], [0, 1, 0, 0,0, 1, 0, 1]]]
#data_size = len(train_set[0])


# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 1

# Network Parameters
D = 1024 # 1st layer number of features
NUM_NOTES = 88 # MNIST data input (img shape: 28*28)

# tf Graph input
input = tf.placeholder("float", [None, NUM_NOTES])
target = tf.placeholder("float", [None, NUM_NOTES])


# Create model
def linear(input, weights):
    hidden = tf.matmul( tf.truediv(input, tf.maximum(1.0,tf.reduce_sum(input, 1, keep_dims=True)) ) , weights['hidden']) #+ bias['hidden1']
    out_layer = tf.matmul(hidden, weights['out'])# + bias['out']
    return out_layer

# Store layers weight & bias
weights = {
    'hidden': tf.Variable(tf.random_normal([NUM_NOTES, D])),
    'out': tf.Variable(tf.random_normal([D, NUM_NOTES]))
}

bias = {
    'hidden': tf.Variable(tf.random_normal([D])),
    'out': tf.Variable(tf.random_normal([NUM_NOTES]))
}



def get_batch(data_set,id, stoch=False):
    if stoch:
        transpose_data_set = list(map(list, zip(*data_set)))
        batch = random.sample(transpose_data_set, batch_size)
        batch_input,batch_target = list(map(list, zip(*batch)))
        return batch_input,batch_target
    batch_id = id + 1
    input, target = data_set
    return input[(batch_id * batch_size - batch_size):(batch_id * batch_size)], target[(batch_id * batch_size - batch_size):(batch_id * batch_size)]


def train(file_name,checkpoint_path='save_models/linear/linear_D1024.ckpt',load_model=None,print_train=False):
    train_set, test_set, valid_set, total_batch, total_batch_test, total_batch_valid = load_data(file_name)
    data_size = len(train_set[0])

    # Construct model
    print('Create model ...')
    
    pred = linear(input, weights)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(pred, target),1))


    #optimizer = tf.train.AdamOptimizer(epsilon=1e-01,learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables(),max_to_keep=1)
    
    input_valid, target_valid = valid_set

    # Launch the graph
    print('Start training ...')
    with tf.Session() as sess:
        checkpoint = False #tf.train.get_checkpoint_state('save_models/linear')
        if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
            print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
            best_val_loss = sess.run(cost, feed_dict={input: input_valid, target: target_valid})
        else:
            sess.run(init)
            best_val_loss = np.inf

        # Training cycle
        previous_eval_loss = []
        best_val_epoch = -1
        strikes = 0
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(data_size/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = get_batch(train_set,i)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, out = sess.run([optimizer, cost, pred], feed_dict={input: batch_x,
                                                             target: batch_y})

                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            c_valid = sess.run(cost, feed_dict={input: input_valid, target: target_valid})
            print("Valid error %4f" % (c_valid))
            previous_eval_loss.append(c_valid)
            improve_valid = previous_eval_loss[-1] < best_val_loss

            if improve_valid:
                best_val_loss = previous_eval_loss[-1]
                best_val_epoch = epoch
                # Save checkpoint.
                saver.save(sess, checkpoint_path,global_step=epoch)
            else:
                strikes += 1
            if strikes > 3:
                break
        print("Optimization Finished!")

        input_test, target_test = test_set
        c_test = sess.run(cost, feed_dict={input: input_test, target: target_test})


        print("Test error %.9f" % (c_test))
        print("Best validation %.9f" % (best_val_loss))


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
