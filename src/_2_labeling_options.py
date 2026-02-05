import numpy as np
import math
from copy import deepcopy


# Function to perform weighted majority voting on a single row
def weighted_majority_vote(row, weights):
    # Convert weights to reverse probabilities, assuming weights list is given from weight of 0 to weight of 16
    # The higher the original weight, the lower its influence should be, as per "reverse of weights as probability"
    reverse_probabilities = [1 / w if w != 0 else 0 for w in weights]

    # Dictionary to keep weighted counts
    weighted_counts = {}

    # Count each observation with its respective reversed weight
    for observation in row:
        if observation in weighted_counts:
            weighted_counts[int(observation)] += reverse_probabilities[int(observation)]
        else:
            weighted_counts[int(observation)] = reverse_probabilities[int(observation)]

    # Finding the observation with the maximum weighted count
    majority_class = max(weighted_counts, key=weighted_counts.get)
    return majority_class

def MultiPointSlicing(traces, pois, num_bits):
    print("pois:",pois, pois.shape)
    new_labels = np.zeros(traces.shape[0])
    HW_all_m = [] #(nb_traces, samplepoints)
    num_per_class = np.zeros(num_bits + 1)
    # build how many per class
    for j in range(num_bits + 1):
        num_per_class[j] = math.ceil(traces.shape[0] / (2 ** num_bits) * math.comb(num_bits, j))
    print("num_per_class", num_per_class)
    print("sum num_per_class:", np.sum(num_per_class))
    print("nb_traces:", traces.shape[0])
    for samplept in range(traces.shape[1]): #sample point
        sorted_index = np.argsort(traces[:, pois[samplept]], axis=0)  # sort in ascending order
        sorted_index = sorted_index[::-1]
        class_combi = deepcopy(num_per_class)
        HW_4_1m = np.zeros(traces.shape[0])

        class_considering = 0
        for idx in range(traces.shape[0]):
            # work on m
            while (class_considering < num_bits+1) and (num_per_class[class_considering] < 1):
                class_considering += 1
            if (class_considering < num_bits+1) and (num_per_class[class_considering] >= 1):
                HW_4_1m[sorted_index[idx]] = class_considering
                num_per_class[class_considering] = num_per_class[class_considering] - 1
            elif class_considering >= num_bits+1:
                HW_4_1m[sorted_index[idx]] = num_bits

        HW_all_m.append(HW_4_1m)
    HW_all_m = np.array(HW_all_m)
    print("HW_all_m", HW_all_m, HW_all_m.shape)
    new_labels = np.apply_along_axis(weighted_majority_vote, axis=1, arr=HW_all_m, weights=class_combi)
    print("new_labels", new_labels, new_labels.shape)
    return new_labels



def labeling_traces(X_train,poi_xors, num_bits,  save_root, labeling_type, poi_selection_mode, nb_poi, save_labels = False):
    if save_labels == True:
        if labeling_type == 'MultiPointSlicing':
            Y_train_solo_all_hw = MultiPointSlicing(X_train, poi_xors, num_bits)
        # elif labeling_type == 'MultiPointClustering':
        #     Y_train_solo_all_hw = Multi_Point_Cluster_Labeling(X_train, poi_xors)
        np.save(save_root + f"{labeling_type}_empirical_hw_{poi_selection_mode}_{nb_poi}_solo.npy", Y_train_solo_all_hw)
    else:
        print("Loading labeling options:", labeling_type)
        Y_train_solo_all_hw = np.load(
            save_root + f"{labeling_type}_empirical_hw_{poi_selection_mode}_{nb_poi}_solo.npy")
    return Y_train_solo_all_hw