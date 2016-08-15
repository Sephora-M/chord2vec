from chord2vec.linear_models import data_processing
from scipy.optimize import minimize

import random, math
import numpy as np

default_configuration = {
    'ERROR_LIMIT'           : 0.001, 
    'learning_rate'         : 0.03, 
    'batch_size'            : 1, 
    'print_rate'            : 1000, 
    'save_trained_model'    : False,
    'input_layer_dropout'   : 0.0,
    'hidden_layer_dropout'  : 0.0, 
    'evaluation_function'   : None,
    'max_iterations'        : ()
}

def backpropagation(model, train_set, test_set, cost_function, compute_dW, evaluation_function=None,
                               ERROR_LIMIT=1e-3, max_iterations=(), batch_size=0, print_rate=1000, save_trained_model=False, **kwargs):
    

    train_inputs, train_targets = data_processing.check_data(model, train_set)
    test_inputs, test_targets = data_processing.check_data(model, test_set)

    # Whether to use another function for printing the dataset error than the cost function. 
    # This is useful if you train the model with the MSE cost function, but are going to 
    # classify rather than regress on your data.
    if evaluation_function != None:
        calculate_print_error = evaluation_function
    else:
        calculate_print_error = cost_function

    if batch_size != 0:
        batch_size = batch_size
    else:
        batch_size = train_inputs.shape[0]

    batch_train_inputs = np.array_split(train_inputs, math.ceil(1.0 * train_inputs.shape[0] / batch_size))
    batch_train_inputs = np.array_split(train_targets, math.ceil(1.0 * train_targets.shape[0] / batch_size))
    batch_indices = range(len(batch_train_inputs))  # fast reference to batches

    error = calculate_print_error(model.update(test_inputs), test_targets)
    reversed_layer_indices = range(len(model.layers))[::-1]

    epoch = 0
    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch += 1

        random.shuffle(batch_indices)  # Shuffle the order in which the batches are processed between the iterations

        for batch_index in batch_indices:
            batch_data = batch_train_inputs[batch_index]
            batch_targets = batch_train_inputs[batch_index]
            batch_size = float(batch_data.shape[0])

            input_signals, derivatives = model.update(batch_data, forward_only=True)
            out = input_signals[-1]
            cost_derivative = cost_function(out, batch_targets, derivative=True).T
            delta = cost_derivative * derivatives[-1]

            for i in reversed_layer_indices:
                # Loop over the weight layers in reversed order to calculate the deltas

                #  weight change
                dX = (np.dot(delta, input_signals[i]) / batch_size).T
                dW = compute_dW(i, dX)

                if i != 0:
                    weight_delta = np.dot(model.weights[i], delta)
                    delta = weight_delta * derivatives[i - 1]

                # Update the weights with Nestrov Momentum
                model.weights[i] += dW
                # end weight adjustment loop

        error = calculate_print_error(model.update(test_inputs), test_targets)

        if epoch % print_rate == 0:
            # Show the current training status
            print("[training] Current error:", error, "\tEpoch:", epoch)

    print("[training] Finished:")
    print("[training]   Converged to error bound (%.4g) with error %.4g." % (ERROR_LIMIT, error))
    print("[training]   Trained for %d epochs." % epoch)

    if save_trained_model:
        model.save_model_to_file()
        
def RMSprop(model, trainingset, testset, cost_function, decay_rate=0.99, epsilon=1e-8, **kwargs):
    configuration = dict(default_configuration)
    configuration.update(kwargs)

    learning_rate = configuration["learning_rate"]
    cache = [np.zeros(shape=weight_layer.shape) for weight_layer in model.weights]

    def calculate_dW(layer_index, dX):
        cache[layer_index] = decay_rate * cache[layer_index] + (1 - decay_rate) * dX ** 2
        return -learning_rate * dX / (np.sqrt(cache[layer_index]) + epsilon)

    # end

    return backpropagation(model, trainingset, testset, cost_function, calculate_dW, **configuration)


def scipyoptimize(model, train_set, test_set, cost_function, method="L-BFGS-B", save_trained_model=False):


    train_inputs, train_targets = data_processing.check_data(model, train_set)
    test_inputs, test_targets = data_processing.check_data(model, test_set)

    error_function_wrapper = lambda weights, train_inputs, train_targets, test_inputs, test_targets,\
                                    cost_function: model.error(weights, [test_inputs, test_targets], cost_function)
    gradient_function_wrapper = lambda weights, train_inputs, train_targets, test_inputs, test_targets,\
                                       cost_function: model.gradient(weights, [train_inputs, train_targets],
                                                                       cost_function)

    results = minimize(
        error_function_wrapper,  # The function we are minimizing
        model.get_weights(),  # The vector (parameters) we are minimizing
        method=method,  # The minimization strategy specified by the user
        jac=gradient_function_wrapper,  # The gradient calculating function
        args=(train_inputs, train_targets, test_inputs, test_targets, cost_function),
        # Additional arguments to the error and gradient function
    )

    model.set_weights(results.x)

    if not results.success:
        print("[training] WARNING:", results.message)
        print("[training]   Terminated with error %.4g." % results.fun)
    else:
        print("[training] Finished:")
        print("[training]   Completed with error %.4g." % results.fun)

        if save_trained_model:
            model.save_model_to_file()
