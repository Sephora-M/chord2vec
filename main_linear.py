import sys
from chord2vec.linear_models.learning import scipyoptimize
from chord2vec.linear_models.learning import RMSprop
from chord2vec.linear_models.linear_model import LinearModel1
from chord2vec.linear_models.functions import *
import pickle
from chord2vec.linear_models import data_processing as dp

dic=pickle.load(open('JSP_processed','rb'))
train_chords = dic['t']
test_chords = dic['te']

train_set = dp.generate_binary_vectors(train_chords)
test_set = dp.generate_binary_vectors(test_chords)

#for testing prupose
#train_set = [[[0, 0, 0, 1,1, 0, 1, 0], [1, 1, 1, 1,0, 0, 1, 1], [1, 1, 1, 0,0, 1, 0, 1], [0, 0, 0, 1,0, 0, 0, 1], [1, 0, 0, 0,1, 0, 0, 1]], \
#            [[1, 1, 0, 1,0, 1, 1, 0], [0, 0, 1, 1,1, 1, 1, 1], [0, 0, 1, 0,1, 0, 0, 1], [1, 1, 0, 1,1, 1, 0,1], [0, 1, 0, 0,0, 1, 0, 1]]]

#test_set = [[[0, 0, 0, 1,1, 0, 1, 0], [1, 1, 1, 1,0, 0, 1, 1], [1, 1, 1, 0,0, 1, 0, 1], [0, 0, 0, 1,0, 0, 0, 1], [1, 0, 0, 0,1, 0, 0, 1]], \
#            [[1, 1, 0, 1,0, 1, 1, 0], [0, 0, 1, 1,1, 1, 1, 1], [0, 0, 1, 0,1, 0, 0, 1], [1, 1, 0, 1,1, 1, 0,1], [0, 1, 0, 0,0, 1, 0, 1]]]

cost_function = sigmoid_cross_entropy_cost

# initialize the neural model
model = LinearModel1()
#model.check_gradient(train_set, cost_function)

## load a stored model configuration
# model           = NeuralNet.load_model_from_file( "model0.pkl" )


# Train the model using LBFGS
scipyoptimize(
        model,
        train_set,                      # specify the training set
        test_set,                          # specify the test set
        cost_function,                      # specify the cost function to calculate error
        method               = "L-BFGS-B",
        save_trained_model = False        # Whether to write the trained weights to disk
    )

print(model.update(train_set),True)

# Train the model using backpropagation
# RMSprop(
#     model,  # the model to train
#     train_set,  # specify the training set
#     test_set,  # specify the test set
#     cost_function,  # specify the cost function to calculate error
#
#     ERROR_LIMIT=1e-2,  # define an acceptable error limit
#     # max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped
#
#     batch_size=0,  # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
#     print_rate=1000,  # print error status every `print_rate` epoch.
#     learning_rate=0.3,  # learning rate
#     momentum_factor=0.9,  # momentum
#     input_layer_dropout=0.0,  # dropout fraction of the input layer
#     hidden_layer_dropout=0.0,  # dropout fraction in all hidden layers
#     save_trained_model=False  # Whether to write the trained weights to disk
# )


def parse_args():
    if len(sys.argv)>1:
        first_arg = sys.argv[1]
        second_arg = sys.argv[2]
        if first_arg == '-D' and isinstance( int(second_arg), int ):
            D = int(second_arg)
            print(D)


if __name__ == "__main__":
    parse_args()