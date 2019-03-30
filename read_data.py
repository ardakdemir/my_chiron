
import h5py
import sys
import os
import numpy as np
path = "../../work/data/cache"
filename = 'train_cache.h5'
import keras
## read data into suitable format for the model
def split_data(inputs,test_size):
    return tuple(map(lambda x : x[:-test_size],inputs)),tuple(map(lambda x : x[-test_size:],inputs))
def read_from_dict(my_dict,example_num = 100 , class_num = 5 , seq_len = 300 ,padding = True):
    all_data = my_dict
    keys = list(my_dict.keys())
    x_tr = []
    y_tr = []
    labels = []
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
    return (x_train,y_train,y_train_class,y_labels,label_lengths)

## read h5 data file
def read_h5(path,filename,example_num = 1000):
    f = h5py.File(os.path.join(path,filename), 'r')
    keys = f.keys()
    groups = {}
    Y_ctc = f['Y_ctc']
    Y_seg = f['Y_seg']
    X_data = f['X_data']
    Y_vec = f['Y_vec']
    seq_len = f['seq_len']
    avail_data = {}
    for key in range(len(X_data[:example_num])):
        segs = np.array(Y_seg[str(key)])
        #print(key)
        avail_data[key] = {}
        avail_data[key]["x_data"] = X_data[int(key)]
        avail_data[key]["y_vec"]  = Y_vec[int(key)]
        avail_data[key]["segments"] = segs
        avail_data[key]["nucleotides"] = segmentstonucleotides(segs,Y_vec[int(key)])
    return avail_data
    #print(len(groups[list(keys)[0]][0]))
def segmentstonucleotides(segments,y_vec):
    nucleotides = [y_vec[0]]
    #print(segments)
    #print(y_vec)
    segment = segments[0]
    i = 1
    ind = segment
    while(segment!=0):
        if y_vec[ind]!=-1:
            nucleotides.append(y_vec[ind])
        segment = segments[i]
        ind += segment
        i+=1
    return nucleotides


if __name__=="__main__":
    args = sys.argv
    example_num = 1000
    dict_file = read_h5(path,filename,example_num = example_num)
    print(dict_file.keys())
