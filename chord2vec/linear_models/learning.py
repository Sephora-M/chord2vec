from chord2vec.linear_models import data_processing
from scipy.optimize import minimize


def optimize(model, train_set, test_set, cost_function, method="L-BFGS-B", save_file=None):


    train_inputs, train_targets = data_processing.check_data(model, train_set)
    test_inputs, test_targets = data_processing.check_data(model, test_set)

    obj_function_wrapper = lambda weights, train_inputs, train_targets, test_inputs, test_targets,\
                                    cost_function: model.error(weights, [test_inputs, test_targets], cost_function)
    gradient_function_wrapper = lambda weights, train_inputs, train_targets, test_inputs, test_targets,\
                                       cost_function: model.gradient(weights, [train_inputs, train_targets],
                                                                       cost_function)

    results = minimize(obj_function_wrapper, model.get_weights(), method=method, jac=gradient_function_wrapper,
        args=(train_inputs, train_targets, test_inputs, test_targets, cost_function), options=dict(disp=True),
    )

    model.set_weights(results.x)

    if not results.success:
        print("[training] WARNING:", results.message)
        print("[training]   Terminated with error %.4g." % results.fun)
    else:
        print("[training] Finished:")
        print("[training]   Completed with error %.4g." % results.fun)

        if save_file is not None:
            model.save_model(save_file)
