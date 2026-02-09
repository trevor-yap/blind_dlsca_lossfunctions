
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from src._0_theorectical_histogram import obtain_theoretical_histogram
from src._1_poi_selections import PoI_Selection_AES, poi_selection_options
from src._2_labeling_options import labeling_traces
from src._3_DL_training import train_pipeline_singletask_dl
from src._4_distinguisher import perform_joint_attack
from src.net import create_hyperparameter_space, MLP, CNN
from src.utils import load_chipwhisperer, check_accuracy, predict_attack_traces, perform_attacks, NTGE_fn, \
    obtain_var_noise, calculate_HW


def compute_noise_transition_matrix(pred, actual, classes):
    noise_transition_matrix = np.zeros((classes,classes))
    for i in range(actual.shape[0]):
        print("actual[i],", actual[i])
        print("pred[i],", pred[i])
        noise_transition_matrix[int(actual[i]), int(pred[i])] += 1.0
    for j in range(noise_transition_matrix.shape[0]):
        if np.sum(noise_transition_matrix[j,:]) != 0:
            noise_transition_matrix[j,:] = noise_transition_matrix[j,:]/np.sum(noise_transition_matrix[j,:])
    return noise_transition_matrix




if __name__ == '__main__':
    dataset = 'Chipwhisperer'
    leakage = "HW"
    labeling_type = "MultiPointClustering"
    epochs = 100
    nb_attacks = 100
    nb_traces_attacks = 1700
    poi_selection_mode = "correlation"





    result_root = "./Result/"
    model_config_root = result_root + "model_config/"
    save_root = result_root + "blind_" + dataset + "_"+ leakage + "_"+labeling_type+"/"
    image_root = result_root + "images/"
    poi_root = result_root + "poi/"
    theorectical_histogram_root = result_root + "theorectical_histogram/"
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    if not os.path.exists(model_config_root):
        os.mkdir(model_config_root)
    if not os.path.exists(image_root):
        os.mkdir(image_root)
    if not os.path.exists(poi_root):
        os.mkdir(poi_root)
    if not os.path.exists(theorectical_histogram_root):
        os.mkdir(theorectical_histogram_root)
    print("save_root:", save_root)
    print("using cuda:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if dataset == 'Chipwhisperer':
        root = "./../"
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling,plt_attack), correct_key = load_chipwhisperer(root + data_root, leakage_model=leakage)
        num_bits = 8
        num_branch = 2
        classes = (num_bits+1)**num_branch
        L_profiling = np.array([plt_profiling, Y_profiling]).T
        L_profiling_HW = np.array([calculate_HW(plt_profiling), calculate_HW(Y_profiling)]).T


    nb_poi = 50
    total_samplept = X_profiling.shape[1]
    number_of_traces = X_profiling.shape[0]
    ######################################################################Phase 1 Compute Theoretical Distribution #############################################################################################3
    ##############################################################################################################################################################################################
    theoretical_histogram = obtain_theoretical_histogram(dataset, theorectical_histogram_root)
    ######################################################################Phase 2 Obtain Empirical Distribution #############################################################################################3
    ##############################################################################################################################################################################################
    #2.1 PoI Selection
    poi_xors = poi_selection_options(dataset, nb_poi, total_samplept, number_of_traces, X_profiling, L_profiling, image_root,poi_root, poi_selection_mode, plot_cpa_image=True, save_poi=False)

    #2.2 Labeling Training traces.
    Y_train_solo_all_hw = labeling_traces(X_profiling, poi_xors, num_bits, save_root, labeling_type, poi_selection_mode, dataset, num_branch, save_labels=False)
    print("Y_train_solo_all_hw:", Y_train_solo_all_hw, Y_train_solo_all_hw.shape)  # [nb_traces, 2]
    Y_train_combined_hws = Y_train_solo_all_hw[:, 0]
    for i in range(1, Y_train_solo_all_hw.shape[1]):
        Y_train_combined_hws += Y_train_solo_all_hw[:, i] * ((num_bits + 1)**i)
    print("Y_train_combined_hws:", Y_train_combined_hws, Y_train_combined_hws.shape)  # [nb_traces,]


    print("L_train_combined_hws:", L_profiling_HW, L_profiling_HW.shape)
    L_train_combined_hws = L_profiling_HW[:, 0]
    for i in range(1, L_profiling_HW.shape[1]):
        L_train_combined_hws += L_profiling_HW[:, i] * ((num_bits + 1)**i)
    print("L_train_combined_hws:", L_train_combined_hws, L_train_combined_hws.shape)  # [nb_traces,]

    # noise_transition_matrix = compute_noise_transition_matrix(pred = Y_train_combined_hws , actual= L_train_combined_hws, classes= classes)
    # np.set_printoptions(threshold=np.inf)
    # print("noise_transition_matrix:", noise_transition_matrix)

    noise_transition_matrix_m = compute_noise_transition_matrix(pred=Y_train_solo_all_hw[:, 0], actual=L_profiling_HW[:, 0],
                                                              classes=9)

    print("noise_transition_matrix_m:", noise_transition_matrix_m)