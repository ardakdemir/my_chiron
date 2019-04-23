import h5py
import sys
import os
import numpy as np
path = "../../work/data/chiron_data/eval"
filename = 'train_cache.hdf5'
import keras
import glob
#from statsmodels import robust

import random

Alphabeta = ['A', 'C', 'G', 'T']
## read data into suitable format for the model
def split_data(inputs,test_size):
    return tuple(map(lambda x : x[:-test_size],inputs)),tuple(map(lambda x : x[-test_size:],inputs))

def read_raw_into_segments(signal_folder,seq_length = 300,normalize= "mean",sample_num = 100,y_pad = 4):
    signals = glob.glob(os.path.join(signal_folder,"*.signal"))
    x_data = []
    y_data = []
    y_lengths = []
    random.shuffle(signals)

    def get_stats(error_list):
        mean = np.mean(error_list)
        median = np.median(error_list)
        std  = np.std(error_list)
        return mean,median,std

    def y_map(char):
        return Alphabeta.index(char)

    for signal in signals[:sample_num]:
        label_file_name = signal[:-6]+"label"
        label_f = open(label_file_name).readlines()
        signal_fr = open(signal).read().split()
        signal_f = [int(x) for x in signal_fr]
        mean,median,std = get_stats(signal_f)
        if normalize == "mean":
            signal_f = (signal_f - mean) / std
        elif normalize == "median":
            signal_f = (signal_f - median) / np.float(robust.mad(signal_f))
        x_dat = []
        y_dat = []
        current_len = 0
        for line in label_f:
            ls = line.split()
            s_base = int(ls[0])
            e_base = int(ls[1])
            event_len = e_base - s_base
            if current_len +event_len > seq_length:
                if current_len > seq_length/2 and len(y_dat)>2:
                    for i in range(seq_length-current_len):
                        x_dat.append(0)
                    x_data.append(x_dat)
                    y_data.append(np.array(y_dat))
                    x_dat = []
                    y_dat = []
                    current_len = 0
                else: ## skip the previous segment because it is short or contain too little bases
                    x_dat = []
                    y_dat = []
                    current_len = 0
                    for i in range(s_base,e_base):
                        x_dat.append(signal_f[i])
                    y_dat.append(y_map(ls[2]))
                    current_len = current_len + event_len
            else:
                for i in range(s_base,e_base):
                    x_dat.append(signal_f[i])
                y_dat.append(y_map(ls[2]))
                current_len = current_len + event_len
    max_all = lambda data : (np.argmax(list(map(lambda x : len(x),data))),len(data[np.argmax(list(map(lambda x : len(x),data)))]))
    max_label_length = max_all(y_data)[1]
    lengths = []
    y_padded = []
    for y in y_data:
        y_new = []
        for l in y:
            y_new.append(l)
        lengths.append(len(y))
        for i in range(max_label_length-len(y)):
            y_new.append(y_pad)
        y_padded.append(y_new)
    x_data = np.array(x_data).reshape(len(x_data),seq_len,1)
    return x_data,y_padded,lengths,max_label_length

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
        labels.append(np.array(all_data[key]["nucleotides"])-1)## X mapped to X-1
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
    groups = {}
    Y_ctc = f['Y_ctc']
    Y_seg = f['Y_seg']
    X_data = f['X_data']
    Y_vec = f['Y_vec']
    seq_len = f['seq_len']
    avail_data = {}
    #print(X_data[example_num])
    for key in range(example_num):
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
    print(len(dict_file.keys()))
