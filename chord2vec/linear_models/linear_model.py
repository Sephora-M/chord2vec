import os
import pickle
import re

import numpy as np
from six.moves import xrange

from chord2vec.linear_models import functions as fct
from chord2vec.linear_models import data_processing as dp

NUM_NOTES = 88
D = 512


default = {
    "sigma"          : 0.1,      # weight range around zero
    "num_inputs": NUM_NOTES,
    "layers": [(D, fct.linear_function), (NUM_NOTES, fct.sigmoid_function)],
    # [ (number_of_neurons, activation_function) ]
    # The last pair in the list dictate the number of output signals
}

class LinearModel1:
    def __init__(self, settings=None):
        self.__dict__.update(default)
        if settings is not None:
            self.__dict__.update(settings)


        # total number of weights
        self.num_weights = self.num_inputs * self.layers[0][0] + sum((self.layers[i][0] ) * layer[0] for i, layer in enumerate(self.layers[1:]))

        # Initialize the model with new randomized weights
        self.set_weights(np.random.uniform(-self.sigma, self.sigma, size=(self.num_weights,)))



    def set_weights(self, weight_list):
        """
        From a list of weights, constructs the weights matrices
        Args:
            weight_list: the list of all the weights

        Returns:

        """
        start, stop = 0, 0
        self.weights = []
        previous_shape = self.num_inputs

        for n_neurons, activation_function in self.layers:
            stop += previous_shape * n_neurons
            self.weights.append(weight_list[start:stop].reshape(previous_shape, n_neurons))

            previous_shape = n_neurons
            start = stop
        self.weights[2] = np.multiply(fct.zero_padding(self.weights[2]), self.weights[2])

    def force_zero_weights(self, id_layer):
        """
        forces some weights in the layer id_layer to be zero
        Args:
            id_layer:

        Returns:

        """

        return id_layer

    def get_weights(self, ):
        """
        Flattents the weight matrix and returns a vector of all weights in the model
        """
        return [w for l in self.weights for w in l.flat]


    def error(self, weight_vector, data_set, cost_function):
        """
        Computes the error on a dataset given a weights vector
        Args:
            weight_vector: a vector containing all weights
            data_set: a pair of (input,target)
            cost_function: the cost function to compute the error

        Returns: the error according to the cost_function

        """
        self.set_weights(np.array(weight_vector))
        inputs,targets = data_set
        output = self.update(inputs)
        return cost_function(output, targets)

    def gradient(self, weight_vector, data_set, cost_function):

        inputs,targets = data_set
        self.set_weights(np.array(weight_vector))

        input_signals, derivatives = self.update(inputs, forward_only=False)

        output = input_signals[-1]
        cost_derivative = cost_function(output, targets, derivative=True).T
        delta = cost_derivative * derivatives[-1]

        num_samples = float(inputs.shape[0])
        deltas = []

        for i in list(range(len(self.layers)))[::-1]:
            # Loop over the weight layers in reversed order to calculate the deltas
            deltas.append(list((np.dot(delta, input_signals[i]) / num_samples).T.flat))

            if i != 0:
                weight_delta = np.dot(self.weights[i], delta)

                delta = weight_delta * derivatives[i - 1]

        return np.hstack(reversed(deltas))


    def check_gradient(self, trainingset, cost_function, epsilon=1e-6):

        inputs, targets = dp.check_data(self, trainingset)


        initial_weights = np.array(self.get_weights())

        print("Checking gradients...")

        print("Number of weights %d" % self.num_weights)

        def numerical_grad(init_weights, input, target, cost_function, epsilon):

            df = np.zeros(init_weights.shape)
            p = np.zeros(init_weights.shape)
            for i in xrange(self.num_weights):
                p[i] = epsilon
                fx_right = self.error(init_weights + p, [input, target], cost_function)
                fx_left = self.error(init_weights - p, [input, target], cost_function)
                p[i] = 0
                df[i] = (fx_right - fx_left) / (2 * epsilon)

            return df
            # for i in xrange(self.num_weights):
            #     p[i] += epsilon
            #     fx_right = self.error(init_weights + p, [input[:1], target[:1]], cost_function)
            #     fx_left = self.error(init_weights , [input[:1], target[:1]], cost_function)
            #     p[i] = 0
            #     df[i] = (fx_right - fx_left) / (epsilon)
            # return df

        numeric_gradient = numerical_grad(initial_weights,inputs,targets,cost_function,epsilon)

        # reset weights
        self.set_weights(initial_weights)

        analytic_gradient = self.gradient(self.get_weights(), [inputs, targets], cost_function)

        # Compare the numeric and the analytic gradient
        ratio = np.linalg.norm(analytic_gradient - numeric_gradient) / np.linalg.norm(
            analytic_gradient + numeric_gradient)

        if not ratio < 1e-6:
            print("WARNING: The numeric gradient check failed! Analytical gradient differed by %g from the numerical." % ratio)
            print("analytical gradient : %.4f" % np.linalg.norm(analytic_gradient))
            print("numerical gradient : %.4f" % np.linalg.norm(numeric_gradient))

        else:
            print("Numeric gradient check passed :)")

        return ratio



    def update2(self, input_values, forward_only=True):
        """
        Forward pass. If forward_only is false, then return the output at each layer and the derivatives,
        otherwise returns the final output only.
        Returns:

        """
        output = fct.normalize_function(input_values)
        #output = input_values
        if not forward_only:
            derivatives = []  # collection of the derivatives of the act functions
            outputs = [output]  # passed through act. func.

        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            signal = np.dot(output, weight_layer)
            # if i >0:
            #     for j in range(signal.shape[1]):
            #         signal[:,j] = np.sum[signal[:,:(j+1)]]

            output = self.layers[i][1](signal)

            if not forward_only:
                outputs.append(output)
                derivatives.append(
                    self.layers[i][1](signal, derivative=True).T)  # the derivative used for weight update

        if not forward_only:
            return outputs, derivatives

        return output

    def update(self, input_values, forward_only=True):
        """
        Forward pass. If forward_only is false, then return the output at each layer and the derivatives,
        otherwise returns the final output only.
        Returns:

        """
        self.weights[2] = np.multiply(fct.zero_padding(self.weights[2]),self.weights[2])
        output = fct.normalize_function(input_values)

        if not forward_only:
            derivatives = []  # collection of the derivatives of the act functions
            outputs = [output]  # passed through act. func.

        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output

            #if i == 0:
                #signal = np.dot(output, np.multiply(fct.zero_padding(weight_layer),weight_layer))

            signal = np.dot(output, weight_layer)
            output = self.layers[i][1](signal)

            if not forward_only:
                outputs.append(output)
                derivatives.append(
                    self.layers[i][1](signal, derivative=True).T)  # the derivative used for weight update

        if not forward_only:
            return outputs, derivatives
        return output

    def evaluate(self, data_set, cost_function ):
        input,target = dp.check_data(self,data_set)

        print("Evaluate model")

        # perform a forward operation to calculate the output signal
        out = self.update(input)
        # calculate the mean error on the data classification
        mean_loss = cost_function( out, target ) #/ float(input.shape[0])
        return mean_loss


    def save_model(self, filename="linear2.pickle"):
        """
        Save the parameters of the model
        """

        if filename == "model0.pickle":
            while os.path.exists(os.path.join(os.getcwd(), filename)):
                filename = re.sub('\d(?!\d)', lambda x: str(int(x.group(0)) + 1), filename)

        with open(filename, 'wb') as file:
            params_dict = {
                "num_inputs": self.num_inputs,
                "layers": self.layers,
                "num_weights": self.num_weights,
                "weights": self.weights,
            }
            pickle.dump(params_dict, file, 4)


    @staticmethod
    def load_model(filename):
        """
        Load the complete configuration of a previously stored model.
        """
        model = LinearModel1()

        with open(filename, 'rb') as file:
            params_dict = pickle.load(file)

            model.num_inputs = params_dict["num_inputs"]
            model.num_weights = params_dict["num_weights"]
            model.layers = params_dict["layers"]
            model.weights = params_dict["weights"]

        return model

