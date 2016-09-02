import matplotlib.pyplot as plt
import pylab
import pickle
import os,math

def plot_train_validation(pickle_file, title,file_name,data_file,num_layers=2,num_units=512,attention=False,batch_size=128):

    result_dic = pickle.load(open(pickle_file,'rb'))
    result_file = open("finals_tex_table.txt", 'a+')

    train_ppx=result_dic['train_ppx']
    eval_ppx = result_dic['valid_ppx']
    test_ppx =  result_dic['test_ppx']

    print(title)
    print('best train : %.4f ' % (min(train_ppx)))
    print('best validation : %.4f ' % (min(eval_ppx)))
    print('test perplexity : %.4f  ' % (test_ppx))
    print('trained for %d epochs' % (1+eval_ppx.index(min(eval_ppx))))

    # result_file.write(title)
    # result_file.write("\n")
    # result_file.write('best train : %.4f \n' % (max(train_ppx)))
    # result_file.write('best validation : %.4f \n' % (max(eval_ppx)))
    # result_file.write('test perplexity : %.4f  \n' % (test_ppx))
    # result_file.write('trained for %d epochs \n' % result_dic['epoch'])

    if attention:
        print(attention)
        result_file.write("\\rowcolor[HTML]{EFEFEF}")
        result_file.write("\n")

    result_file.write('%s &   %.4f    &    %.4f        &    %.4f  &   %d     \\\\ \hline' %
                      (data_file, -math.log(min(train_ppx)),-math.log(min(eval_ppx)),-math.log(test_ppx), (1+eval_ppx.index(min(eval_ppx)))))
    result_file.write("\n")

    result_file.close()

    plt.plot(train_ppx,label='train perplexity')
    plt.plot(eval_ppx,label='validation perplexity')
    plt.title(title)
    plt.legend()
    pylab.savefig('plots/final_models/'+file_name+'_train_valid.png', bbox_inches='tight')
    plt.close()

def plot_train_losses(pickle_file, title,file_name):
    result_dic = pickle.load(open(pickle_file, 'rb'))

    training_losses = result_dic['train_losses']

    plt.plot(training_losses, label='train losses')
    plt.title(title)
    plt.legend()
    pylab.savefig('plots/final_models'+file_name + '_train_losses.png', bbox_inches='tight')
    plt.close()

def print_results(num_layers, num_units, attention, batch_size, reg_param=0, data_file='JSB_Chorales'):

    init_dir = 'bach'
    #if num_layers == 1:
    #    init_dir = init_dir+'/1layer_models'
    #else:
    #init_dir = init_dir+'/'+str(num_layers)+'layers_models/Addam'

    file_name = data_file+'_'+str(batch_size)+'batch_'+str(num_layers)+'layers_'+str(num_units)+'_units'#+str(reg_param)+'lamda'
    pickle_file =  init_dir+'/'+ file_name
    title = data_file #'num layers : ' + str(num_layers) + ' num units : ' + str(num_units) + 'batch size : ' + str(batch_size) #+ 'lambda : ' + str(reg_param)

    if attention:
        pickle_file += '_attention'
        file_name += '_attention'
        title += ' with attention'

    pickle_file += '/results.pickle'
    if os.path.exists(pickle_file):
        plot_train_losses(pickle_file,title,file_name)
        plot_train_validation(pickle_file,title,file_name,data_file,num_layers, num_units, attention,batch_size)
    else:
        print("%s does not exist!" % pickle_file)

def print_all():
    layers = 2
    units = 512
    attention = [False, True]
    batch_size = 128
    b = 128
    data_files = ['all_data']
    #reg_params = [0, 0.1, 0.3, 0.5, 1,2,3,5,10]
    # for num_layer in layers:
    #     for num_units in units:
    #         for att in attention:
    #             #for b in batch_size:
    #             for lamba in reg_params:
    #                 print_results(num_layer, num_units, att,b,lamba)
    for data_file in data_files:
            print_results(layers, units, False, b, data_file=data_file)

