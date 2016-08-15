from collections import Counter
import pickle
from chord2vec.linear_models import data_processing as dp
import math
import numpy as np
def normalized_density(epsilon = 1e-8, dic=None):
    #dic=pickle.load(open('JSB_Chorales.pickle','rb'))
    if dic is None:
        dic = pickle.load(open('JSB_processed.pkl', 'rb'))
    train_data = dic['t']
    alpha = 1.0
    data_size = len(train_data[0])
    prob_distr=[]
    c = Counter(sum(train_data[0],[]))
    for i in range(88):
        prob_distr.append( (c[i]+alpha)/(data_size + alpha))

    prob_distr = np.array(prob_distr)
    print(prob_distr)
    test_data = dic['te']
    test_data = dp.generate_binary_vectors(test_data)[0]

    valid_data = dic['v']
    valid_data = dp.generate_binary_vectors(valid_data)[0]

    train_data = dp.generate_binary_vectors(train_data)[0]

    return prob_distr, train_data, test_data, valid_data

def eval(prob_distr, test_data,epsilon = 1e-8):
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

    return math.exp(-losses), -losses

def run():

    prob_distr, train_data, test_data, valid_data = normalized_density()

    ppx, loss = eval(prob_distr, train_data)
    print(ppx)
    ppx, loss = eval(prob_distr, test_data)
    print(ppx)
    ppx, loss = eval(prob_distr, valid_data)
    print(ppx)