import keras
from keras.models import Sequential
from keras.layers import Conv1D,Conv2D,Dense, Dropout,Flatten,Bidirectional, Activation,BatchNormalization
from keras.optimizers import SGD
from keras.models import Sequential,Model
from keras.backend import ctc_decode, variable,get_value
import keras.backend as K
from keras.layers import Dense, Activation,Input,LSTM, Lambda
import numpy as np

n = 100
class_num = 5
batch_size = 10
seq_len = 300
pickle_path = "all_data.pk"

def read_pickle(pickle_path,example_num = 100 , class_num = 5 , seq_len = 300):
    all_data = pickle.load(open(pickle_path,"rb"))
    keys = list(all_data.keys())
    all_data[keys[0]].keys()
    x_tr = []
    y_tr = []
    labels = []
    for key in keys:
        x_tr.append(all_data[key]['x_data'])
        y_tr.append(all_data[key]['y_vec'])
        labels.append(np.array(all_data[key]["nucleotides"])-1)
    x_train = np.array(x_tr[:example_num]).reshape(example_num,seq_len,1)
    y_train = np.array(y_tr[:example_num]).reshape(n,seq_len,1)
    y_labels = np.array(labels[:example_num])
    y_train = keras.utils.to_categorical(y_train,num_classes = class_num)
    return x_train,y_train,y_labels


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
    FC = Dense(class_num,activation = 'softmax',input_shape=(hidden_num*2,))
    x = FC(x)
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
    x_train,y_train,y_labels = read_pickle(pickle_path,example_num = example_num , class_num = 5 , seq_len = 300)
    inputs = Input(shape=x_train.shape[1:])
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    outputs = chiron_cnn(inputs,256,1,3)
    outputs2 = chiron_rnn(outputs,200)
    #print(np.ones(outputs2.shape[1])*int(outputs2.shape[2]))
    model2 = Model(inputs= inputs,outputs=outputs2)
    sgd = SGD()
    model2.compile(loss = lambda y_train,outputs2 : outputs2,optimizer = sgd)
    model2.summary()
    model2.fit(x_train,y_train,batch_size = batch_size,epochs=3)
    decodeds = ctc_predict(model2,x_train[:10])
    vals = get_value(decodeds[0][0])
    print(vals)
