'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from chord2vec.linear_models import functions
import tensorflow as tf
from chord2vec.linear_models import data_processing as dp
import pickle

print('Loading data ...')
dic=pickle.load(open('JSB_processed.pkl','rb'))
train_chords = dic['t']
test_chords = dic['te']

train_set = dp.generate_binary_vectors(train_chords)
data_size = len(train_set[0])
test_set = dp.generate_binary_vectors(test_chords)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 128
display_step = 1

# Network Parameters
D = 512 # 1st layer number of features
NUM_NOTES = 88 # MNIST data input (img shape: 28*28)

# tf Graph input
input = tf.placeholder("float", [None, NUM_NOTES])
target = tf.placeholder("float", [None, NUM_NOTES])


# Create model
def multilayer_perceptron(input, weights):

    hidden = tf.matmul(functions.normalize(input,False), weights['hidden'])


    # Output layer with sigmoid activation
    out_layer = tf.sigmoid(tf.matmul(hidden, weights['out']))

    return out_layer

# Store layers weight & bias
weights = {
    'hidden': tf.Variable(tf.random_normal([NUM_NOTES, D])),
    'out': tf.Variable(tf.random_normal([D, NUM_NOTES]))
}


# Construct model
pred = multilayer_perceptron(input, weights)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, target))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(data_size/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(train_set,i)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={input: batch_x,
                                                          target: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({input: test_set[0], target: test_set[1]}))


def get_batch(data_set,id):
    batch_id = id+1
    input,target = data_set
    return input[(batch_id * batch_size - batch_size):(batch_id * batch_size)], target[(batch_id * batch_size - batch_size):(batch_id * batch_size)]