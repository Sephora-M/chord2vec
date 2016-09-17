"""
Train a linear chord2vec (numpy implementation) using "L-BFGS-B" optimizer
"""

import sys
from chord2vec.linear_models.learning import optimize
from chord2vec.linear_models.linear_model import LinearModel1
from chord2vec.linear_models.functions import *
import pickle
from chord2vec.linear_models import data_processing as dp


NUM_NOTES=88
D=512
file_name=None
def load_data(file_name ='JSB_processed.pkl'):
    print('Loading data ...')
    dic=pickle.load(open(file_name,'rb'))
    train_chords = dic['t']
    test_chords = dic['te']
    train_set = dp.generate_binary_vectors(train_chords)
    test_set = dp.generate_binary_vectors(test_chords)
    return train_set,test_set



def create_model(NUM_NOTES=NUM_NOTES, D=D, load_model=None):
    if load_model is None:
        model = LinearModel1({
            "sigma"          : 0.1,      # weight range around zero
            "num_inputs": NUM_NOTES,
            "layers": [(D, linear_function), (NUM_NOTES,sigmoid_function ), (NUM_NOTES, sigmoid_function) ]#,(NUM_NOTES, linear_function)],
        })
    else:
        model = LinearModel1.load_model(load_model)
    return model

def train(model, train_set, test_set,cost_function = binary_cross_entropy_cost,save_file=None):
    optimize(model, train_set, test_set, cost_function, method="L-BFGS-B", save_file=save_file)


def check_grad():
 #for testing prupose
    train_set = [[[0, 0, 0, 1,1, 0, 1, 0], [1, 1, 1, 1,0, 0, 1, 1], [1, 1, 1, 0,0, 1, 0, 1], [0, 0, 0, 1,0, 0, 0, 1], [1, 0, 0, 0,1, 0, 0, 1]], \
            [[1, 1, 0, 1,0, 1, 1, 0], [0, 0, 1, 1,1, 1, 1, 1], [0, 0, 1, 0,1, 0, 0, 1], [1, 1, 0, 1,1, 1, 0,1], [0, 1, 0, 0,0, 1, 0, 1]]]

    test_set = [[[0, 0, 0, 1,1, 0, 1, 0], [1, 1, 1, 1,0, 0, 1, 1], [1, 1, 1, 0,0, 1, 0, 1], [0, 0, 0, 1,0, 0, 0, 1], [1, 0, 0, 0,1, 0, 0, 1]], \
            [[1, 1, 0, 1,0, 1, 1, 0], [0, 0, 1, 1,1, 1, 1, 1], [0, 0, 1, 0,1, 0, 0, 1], [1, 1, 0, 1,1, 1, 0,1], [0, 1, 0, 0,0, 1, 0, 1]]]

    #train_set, test_set = load_data()
    i, o = train_set
    train_set = [i[:2],o[:2]]

    model = create_model(8,16)
    model.check_gradient(train_set,binary_cross_entropy_cost)

def main():
    if len(sys.argv)>2:
        first_arg = sys.argv[1]
        second_arg = sys.argv[2]
        if first_arg == '-D' and isinstance(int(second_arg), int):
            D = int(second_arg)
    if len(sys.argv) > 4:
        third_arg = sys.argv[3]
        forth_arg = sys.argv[4]
        if third_arg == '-F':
            file_name = forth_arg

    if len(sys.argv) > 5:
        if sys.argv[5] == '-T':
            train_set = [
                [[0, 0, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 1]], \
                [[1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 1, 1, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1]]]

            test_set = [
                [[0, 0, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0, 1],[1, 0, 0, 0, 1, 0, 0, 1]], \
                [[1, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 1, 1, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1]]]
    else:
        train_set, test_set = load_data()

    model = create_model(D=D)

    train(model, train_set, test_set, binary_cross_entropy_cost,file_name)


    print("Test loss :")
    print(model.evaluate(train_set, binary_cross_entropy_cost))

if __name__ == "__main__":
    main()