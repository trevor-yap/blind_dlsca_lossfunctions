import numpy as np
import math
from copy import deepcopy
from sklearn.mixture import GaussianMixture

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


def Slice_Clustering(center_traces_full, dataset, num_branch):
    if dataset == "Kyber":
        class_combi = [1, 2, 3, 4, 8, 18, 33, 48, 55, 48, 33, 18, 8, 4, 3, 2, 1]
        assert num_branch == 2
    elif dataset == "Chipwhisperer" or dataset == "Chipwhisperer_desync":
        class_combi = [1, 2, 8, 18, 23, 18, 8, 2, 1]
        assert num_branch == 2
    elif dataset == "Ascon":
        class_combi = [3, 23, 80, 159, 199, 159, 80, 23, 3]
        assert num_branch == 3
    num_poi = center_traces_full.shape[1]//num_branch
    max_nb_class = len(class_combi)
    print("class_combi:", class_combi)

    all_majority = []
    for branch in range(num_branch):
        center_traces = center_traces_full[:, branch*num_poi: (branch+1)*num_poi]
        HW_all_m = np.zeros(center_traces.shape)
        for i in range(center_traces.shape[1]):
            sorted_index = np.argsort(center_traces[:, i], axis=0)[::-1] # sort in ascending orde

            num_per_class = deepcopy(class_combi)

            HW_4_1m = np.zeros(center_traces.shape[0])

            class_considering = 0

            for idx in range(center_traces.shape[0]):
                # work on m
                while (class_considering < max_nb_class) and (num_per_class[class_considering] < 1):
                    class_considering += 1
                if (class_considering < max_nb_class) and (num_per_class[class_considering] >= 1):
                    HW_4_1m[sorted_index[idx]] = class_considering
                    num_per_class[class_considering] = num_per_class[class_considering] - 1
                elif class_considering >= max_nb_class:
                    HW_4_1m[sorted_index[idx]] = class_considering-1
            HW_all_m[:, i] = HW_4_1m
        majority = np.apply_along_axis(weighted_majority_vote, axis=1, arr=HW_all_m, weights=class_combi)
        # print("majority:", majority, majority.shape)
        all_majority.append(majority)

    return np.array(all_majority).T

def Multi_Point_Cluster_Labeling(traces, poi_xors, dataset, num_bits = 9, num_branch =2):

    print("traces: ", traces.shape)
    print("poi_xors: ", poi_xors.shape)
    num_poi = poi_xors.shape[1]
    new_labels = np.zeros((traces.shape[0], num_branch))
    if num_branch == 2:
        X_train_joint = np.hstack((traces[:, poi_xors[0]], traces[:, poi_xors[1]]))
    elif num_branch == 3:
        X_train_joint = np.hstack((traces[:, poi_xors[0]], traces[:, poi_xors[1]], traces[:, poi_xors[2]]))
    print("X_train_joint:", X_train_joint.shape)
    #Fit the Gaussian Mixture for one branch
    n_clusters = num_bits**num_branch
    model_joint = GaussianMixture(n_components=n_clusters, random_state=0)
    model_joint.fit(X_train_joint[:1000])
    clusters_joint = model_joint.predict(X_train_joint)
    print("clusters_joint", clusters_joint, clusters_joint.shape)
    center_traces = np.zeros((n_clusters, num_branch*num_poi))
    for cluster in range(n_clusters):
        cluster_members = X_train_joint[clusters_joint == cluster, :]
        center_traces[cluster] = np.mean(cluster_members, axis=0)
    empirical_hws = Slice_Clustering(center_traces, dataset, num_branch)
    print("empirical_hws", empirical_hws.shape)

    #label all the traces in the same cluster.
    for cluster in range(n_clusters):
        cluster_label = empirical_hws[cluster]
        new_labels[clusters_joint == cluster] = cluster_label

    return new_labels



def labeling_traces(X_train,poi_xors, num_bits,  save_root, labeling_type, poi_selection_mode, dataset, save_labels = False):
    if save_labels == True:
        if labeling_type == 'MultiPointClustering':
            Y_train_solo_all_hw = Multi_Point_Cluster_Labeling(X_train, poi_xors,dataset, num_bits)

        np.save(save_root + f"{labeling_type}_empirical_hw_{poi_selection_mode}_solo.npy", Y_train_solo_all_hw)
    else:
        print("Loading labeling options:", labeling_type)
        Y_train_solo_all_hw = np.load(
            save_root + f"{labeling_type}_empirical_hw_{poi_selection_mode}_solo.npy")
    return Y_train_solo_all_hw