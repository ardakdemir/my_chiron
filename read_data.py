
import h5py
import sys
import os
import numpy as np
path = "../../work/data/cache"
filename = 'train_cache.h5'

## read h5 data file
def read_h5(path,filename,example_num = 100):
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
