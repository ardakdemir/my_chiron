import tables
import numpy as np
path = ""
h5file = tables.open_file(os.path.join(path,'train_cache.h5'), driver="H5FD_CORE")
root = h5file.root
#a = root._f_get_child("label_raw")._v_nchildren
Y_ctc = root._f_get_child('Y_ctc')
Y_seg = root._f_get_child('Y_seg')
X_data = root._f_get_child('X_data')
Y_vec = root._f_get_child('Y_vec')
seq_len = root._f_get_child('seq_len')
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
avail_data = {}
for key in range(len(X_data)):
    segs = np.array(Y_seg[str(key)])
    #print(key)
    avail_data[key] = {}
    avail_data[key]["x_data"] = X_data[int(key)]
    avail_data[key]["y_vec"]  = Y_vec[int(key)]
    avail_data[key]["segments"] = segs
    avail_data[key]["nucleotides"] = segmentstonucleotides(segs,Y_vec[int(key)])
avail_file = open("all_data.pk","wb")
pickle.dump(avail_data,avail_file)
