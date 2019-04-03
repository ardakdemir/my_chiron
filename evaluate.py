## given a list of normalized edit distances
## calculate and return mean and std.
import numpy as np
def get_stats(error_list):
    mean = np.mean(error_list)
    std  = np.std(error_list)
    return mean,std

## normalized edit distance
## by length of the true sequence
##normalized by length of the max distance
def norm_edit_distance(pred,truth):
    edits = [[0 for i in range(len(truth)+1)]for j in range(len(pred)+1)]
    for i in range(0,len(pred)):
        for j in range(0,len(truth)):
            if pred[i]==truth[j]:
                edits[i+1][j+1] = min(edits[i][j+1]+1,edits[i][j],edits[i+1][j]+1)
            else:
                edits[i+1][j+1] = min(edits[i][j+1],edits[i][j],edits[i+1][j])+1
    norm  = float(edits[-1][-1])/len(truth)
    return norm

## calculate the average normalized edit distance over all predictions
def evaluate_preds(preds,truths):
    distances = []
    for pred , truth in zip(preds,truths):
        #print(pred)
        #print(truth)
        norm_dist = norm_edit_distance(pred,truth)
        distances.append(norm_dist)
    stats = get_stats(distances)
    return stats
