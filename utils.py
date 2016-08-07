import matplotlib.pyplot as plt
import pickle

def plot_train_validation(pickle_file):
    result_dic = pickle.load(open(pickle_file,'rb'))


    train_ppx=result_dic['train_ppx']
    eval_ppx = result_dic['valid_ppx']

    plt.plot(train_ppx,label='train perplexity')
    plt.plot(eval_ppx,label='validation perplexity')
    plt.legend()
    plt.show()

def plot_train_losses(pickle_file):
    result_dic = pickle.load(open(pickle_file, 'rb'))

    training_losses = result_dic['train_losses']

    plt.plot(training_losses, label='train losses')
    plt.legend()
    plt.show()