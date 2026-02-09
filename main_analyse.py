


import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from src._2_labeling_options import labeling_traces
from src._3_DL_training import train_pipeline_singletask_dl
from src.net import create_hyperparameter_space, MLP, CNN
from src.utils import load_chipwhisperer, check_accuracy, predict_attack_traces, perform_attacks, NTGE_fn

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = 'Chipwhisperer'
    leakage = "HW"
    labeling_type = "MultiPointClustering"
    epochs = 50
    nb_attacks = 100
    nb_traces_attacks = 1700



    result_root = "./Result/"
    save_root = result_root + "blind_" + dataset + "_"+ leakage + "_"+labeling_type+"/"
    model_config_root = result_root + "model_config/"
    image_root = save_root + "images/"
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    if not os.path.exists(model_config_root):
        os.mkdir(model_config_root)
    if not os.path.exists(image_root):
        os.mkdir(image_root)

    total_num_model = 100

    for model_type in ["mlp", "cnn"]:
        for model_idx in range(total_num_model):
            for loss_type in ["CCE", "PEER_LOSS_CCE"]:  # , "PEER_LOSS_CCE"

                trained_model_root = save_root + f'result_{model_type}_{epochs}_{loss_type}/'


                result = np.load(trained_model_root + f"GE_SR_{dataset}_{leakage}_{labeling_type}_model{model_idx}_{model_type}_epochs{epochs}_{loss_type}.npy",
                       allow_pickle=True).item()
                GE = result["GE"]
                NTGE = result["NTGE"]
                print(model_type, model_idx, loss_type, "GE", GE, "NTGE", NTGE)