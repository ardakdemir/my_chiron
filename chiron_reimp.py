import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D,Conv2D,Dense, Dropout,Flatten,Bidirectional, Activation,BatchNormalization
from keras.layers import TimeDistributed
from keras.optimizers import SGD
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.backend import ctc_decode, variable,get_value
import keras.backend as K
from keras.layers import Dense, Activation,Input,LSTM, Lambda
import pickle
import sys
import os
from random import shuffle

from evaluate import evaluate_preds

from read_data import read_h5,read_from_dict
n = 100000
class_num = 5
batch_size = 32
epoch_num = 10
seq_len = 300
max_nuc_len = 48
pickle_path = "toy_data.pk"

def create_model(input_shape=(300,1),class_num=5,max_nuc_len =48):
    inputs = Input(shape=input_shape)
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    outputs = chiron_cnn(inputs,256,1,3)
    outputs2 = chiron_rnn(outputs,200)
    dense =  TimeDistributed(Dense(class_num))(outputs2)
    preds = TimeDistributed(Activation('softmax',name = 'softmax'))(dense)
    labels = Input(name='the_labels', shape=[max_nuc_len], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([preds,labels,input_length,label_length])
    model3 = Model(inputs= [inputs,labels,input_length,label_length],outputs=loss_out)
    return inputs,input_length,outputs,outputs2,dense,preds,labels,input_length,label_length,loss_out,model3

def test_model(model_weight_path,h5_test_path, test_size = 100):
    x_data = np.array(pickle.load(open('x_data.pk',"rb")))
    shuffle(x_data)
    #y_data = pickle.load(open('y_data.pk',"rb"))
    x_data = x_data[:test_size].reshape(len(x_data[:test_size]),300,1)


    h5_dict = read_h5("",h5_test_path,example_num = n)
    x_tr,y_tr,y_categorical,y_labels,label_lengths = read_from_dict(h5_dict,example_num = n , class_num = 5 , seq_len = 300 ,padding = True)
    assert len(x_tr)== len(y_tr) == len(y_categorical )== len(y_labels) == len(label_lengths), "Dimension not matched"
    

    inputs,input_length,outputs,outputs2,dense,preds,labels,input_length,label_length,loss_out,model3 = create_model()
    flattened_input_x_width = keras.backend.squeeze(input_length, axis=-1)
    top_k_decoded, _ = K.ctc_decode(preds, flattened_input_x_width)
    decoder = K.function([inputs, flattened_input_x_width], [top_k_decoded[0]])
    model3.load_weights(model_weight_path)
    #model3.summary()
    inputs = x_data
    shapes = [len(x_data[0])for i in range(test_size)]
    print(inputs.shape)
    decoded = decoder([inputs, shapes])
    print(decoded[0])
    return 0

##read nucleotide sequence for each x,y pair and store in arrays
## pad the nucleotide sequences into max_length with 4 (denoting blank)
def read_pickle(pickle_path,example_num = 100 , class_num = 5 , seq_len = 300 ,padding = True):
    all_data = pickle.load(open(pickle_path,"rb"))
    keys = list(all_data.keys())
    x_tr = []
    y_tr = []
    labels = []
    print(example_num)
    label_lengths = []
    for key in keys:
        x_tr.append(all_data[key]['x_data'])
        y_tr.append(all_data[key]['y_vec'])
        labels.append(np.array(all_data[key]["nucleotides"])-1)
    x_train = np.array(x_tr[:example_num]).reshape(example_num,seq_len,1)
    y_train = np.array(y_tr[:example_num]).reshape(example_num,seq_len,1)
    y_labels =labels[:example_num]
    label_lengths = list(map(lambda x : len(x),y_labels))
    if padding:
        pad = 4
        max_length = max(list(map(lambda x : len(x),y_labels)))
        for i in range(len(y_labels)):
            leng = len(y_labels[i])
            y_labels[i] = np.pad(y_labels[i],(0,max_length-leng),'constant', constant_values=(4,4))
    #print(y_labels[0])
    y_train_class = keras.utils.to_categorical(y_train,num_classes = class_num)
    return x_train,y_train,y_train_class,y_labels,label_lengths


## conv-layer
def conv1D_layer(inputs,filternum,filtersize,activation='relu'):
    conv = Conv1D(filternum,filtersize,padding='same',input_shape = inputs.shape)
    x = inputs
    x = conv(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x
def chiron_cnn(inputs,filternum1,filtersize1,filtersize2,res_layers = 5,activation = 'relu'):
    x = inputs
    for i in range(res_layers):
        x = chiron_res_layer(x,filternum1,filtersize1,filtersize2,activation = activation)
    #x = Flatten()(x)
    return x
## two branches of cnn
def chiron_res_layer(inputs,filternum1,filtersize1,filtersize2,activation = 'relu'):
    x   = inputs
    b_1 = conv1D_layer(x,filternum1,filtersize1,activation = activation)
    b_2 = conv1D_layer(x,filternum1,filtersize1,activation = activation)
    b_2 = conv1D_layer(b_2,filternum1,filtersize2,activation = activation)
    b_2 = conv1D_layer(b_2,filternum1,filtersize1,activation = activation)
    y = keras.layers.add([b_1,b_2])
    y = Activation(activation)(y)
    return y

def chiron_rnn(inputs,hidden_num =200,rnn_layers = 3,class_num = class_num ):
    x = inputs
    for i in range(rnn_layers):
        x = chiron_bilstm_layer(x,hidden_num = hidden_num)
    #FC = Dense(class_num,activation = 'softmax',input_shape=(hidden_num*2,))
    #x = FC(x)
    return x
def chiron_bilstm_layer(inputs,hidden_num):
    firstbi = Bidirectional(LSTM(hidden_num, return_sequences=True),
                        input_shape=inputs.shape)
    x = inputs
    x = firstbi(x)
    x = BatchNormalization()(x)
    return x

# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_predict(model,inputs,beam_width = 100, top_paths = 1):
    lens = lambda l :list(map (lambda x:len(x),l))
    preds = model.predict(inputs)
    #print(preds)
    if top_paths !=1:
        decoded_preds = ctc_decode(preds,lens(inputs),greedy=False,beam_width=beam_width,top_paths=top_paths)
    else:
        decoded_preds = ctc_decode(preds,lens(inputs),beam_width=beam_width)
    return decoded_preds



if __name__ == "__main__":
    test =1
    model_weight_path = "model3_weights.h5"
    test_data_path = ""
    if test == 1:
        test_model(model_weight_path,test_data_path)
        exit()
    args = sys.argv
    with_ctc = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    evaluate = 0
    h5file_path = "../../work/data/cache/train_cache.h5"
    if len(args)>1:
        h5file_path = args[1]
        #with_ctc = int(args[2])
    h5_dict = read_h5("",h5file_path,example_num = n)
    x_tr,y_tr,y_categorical,y_labels,label_lengths = read_from_dict(h5_dict,example_num = n , class_num = 5 , seq_len = 300 ,padding = True)
    assert len(x_tr)== len(y_tr) == len(y_categorical )== len(y_labels) == len(label_lengths), "Dimension not matched"

    inputs = Input(shape=x_tr.shape[1:])
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    outputs = chiron_cnn(inputs,256,1,3)
    outputs2 = chiron_rnn(outputs,200)
    dense =  TimeDistributed(Dense(class_num))(outputs2)
    preds = TimeDistributed(Activation('softmax',name = 'softmax'))(dense)
    #print(np.ones(outputs2.shape[1])*int(outputs2.shape[2]))
    model2 = Model(inputs= inputs,outputs=preds)
    sgd = SGD()
    model2.summary()

    ##ctc decoding only used during prediction
    ## during training cross-entropy is used to calculate loss over each softmax output
    ## works fine
    if with_ctc == 0:
        model2.compile(loss = "categorical_crossentropy",optimizer = sgd)
        model2.fit(x_tr,y_categorical,batch_size = batch_size,epochs=epoch_num)
        decodeds = ctc_predict(model2,x_tr[:10])
        vals = get_value(decodeds[0][0])
        print(vals)

    ## ctc_batch_cost is used during training
    ## now obtaining too high loss values and model only predicts 1
    else:
        # Define CTC loss
        max_nuc_len = len(y_labels[0])
        labels = Input(name='the_labels', shape=[max_nuc_len], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([preds,labels,input_length,label_length])
        flattened_input_x_width = keras.backend.squeeze(input_length, axis=-1)
        top_k_decoded, _ = K.ctc_decode(preds, flattened_input_x_width)
        decoder = K.function([inputs, flattened_input_x_width], [top_k_decoded[0]])
        model3 = Model(inputs= [inputs,labels,input_length,label_length],outputs=loss_out)
        model3.summary()
        model3.compile(loss = {'ctc': lambda y_true, y_pred: y_pred},optimizer = Adam())
        input_lengths = np.array([300 for i in range(len(x_tr))])
        label_lengths = np.array(label_lengths)
        outputs = {'ctc': np.zeros(n)}
        model3.fit([x_tr,np.array(y_labels),np.array(input_lengths),np.array(label_lengths)],outputs,batch_size = batch_size,epochs=10)
        model3.save_weights("model3.h5")
