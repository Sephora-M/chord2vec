from collections import Counter
import pickle
from chord2vec.linear_models import data_processing as dp
import math
import numpy as np

NUM_NOTES=88

def normalized_density(train_data):

    alpha = 1.0
    data_size = len(train_data[0])
    prob_distr=[]
    c = Counter(sum(train_data[0],[]))
    for i in range(NUM_NOTES):
        prob_distr.append( (c[i]+alpha)/(data_size + alpha)) # additive smoothing

    prob_distr = np.array(prob_distr)

    return prob_distr

def get_data(dic=None):
    if dic is None:
        dic = pickle.load(open('JSB_processed.pkl', 'rb'))
    training_data = dic['t']


    test_data = dic['te']
    test_data = dp.generate_binary_vectors(test_data)[0]

    valid_data = dic['v']
    valid_data = dp.generate_binary_vectors(valid_data)[0]
    train_data = dp.generate_binary_vectors(training_data)[0]

    return  training_data, train_data, test_data, valid_data

def eval(data_file, epsilon = 1e-8):
    training_data, valid_data, test_data = dp.read_data(data_file,1)
    test_data = dp.generate_binary_vectors(test_data)[0]
    train_data = dp.generate_binary_vectors(training_data)[0]

    prob_distr = normalized_density([train_data])

    losses = 0.0
    num_notes = len(test_data[0])
    for data_point in test_data:
        loss = 0.0
        for note in range(num_notes):
            # avoiding overflow
            #output = np.clip(prob_distr[note], epsilon, 1 - epsilon)
            output = prob_distr[note]
            loss += np.log(output)* data_point[note]  +  np.log(1 - output)*(1-data_point[note])

        losses += loss
    losses /= len(test_data)

    print('Test loss : %.4f'  %losses)



def sigmoid_cross_entropy_cost(outputs, targets, derivative=False, epsilon=1e-8):
    """
    the sigmoid cross entropy cost
    """
    # preventing overflow
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    divisor = np.maximum(outputs * (1 - outputs), epsilon)
    if derivative:
        return (outputs - targets) / divisor
    else:
        return np.mean(-np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs), axis=1))
def run():

    training_data, train_data, test_data, valid_data = get_data()


    ppx, loss = eval(prob_distr, train_data)
    print(ppx)
    ppx, loss = eval(prob_distr, test_data)
    print(ppx)
    ppx, loss = eval(prob_distr, valid_data)
    print(ppx)