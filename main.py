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
    obtain_var_noise

# Press the green button in the gutter to run the script.
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
    Y_train_solo_all_hw = labeling_traces(X_profiling, poi_xors, num_bits, save_root, labeling_type, poi_selection_mode, dataset, num_branch, save_labels=True)
    print("Y_train_solo_all_hw:", Y_train_solo_all_hw, Y_train_solo_all_hw.shape)  # [nb_traces, 2]
    Y_train_combined_hws = Y_train_solo_all_hw[:, 0]
    for i in range(1, Y_train_solo_all_hw.shape[1]):
        Y_train_combined_hws += Y_train_solo_all_hw[:, i] * ((num_bits + 1)**i)
    print("Y_train_combined_hws:", Y_train_combined_hws, Y_train_combined_hws.shape)  # [nb_traces, 4]

    ######################################################################    Training DNN #############################################################################################3
    total_num_model = 100
    save_config = False
    for model_type in ["mlp", "cnn"]:
        if save_config == True:
            print("SAVING NEW CONFIGURATION")
            for model_idx in range(total_num_model):
                config = create_hyperparameter_space(model_type)
                np.save(model_config_root + "configuration" + str(model_idx) + "_" + model_type + ".npy",
                        config)
            print("Done saving")
    trainning_model = True
    for model_type in ["mlp", "cnn"]:
        for loss_type in ["CCE", "PEER_LOSS_CCE"]:  # , "PEER_LOSS_CCE"
            for model_idx in range(total_num_model):


                config = np.load(model_config_root + "configuration" + str(model_idx) + "_" + model_type + ".npy",
                                 allow_pickle=True).item()

                trained_model_root = save_root + f'result_{model_type}_{epochs}_{loss_type}/'
                if not os.path.exists(trained_model_root):
                    os.mkdir(trained_model_root)
                # Train a DNN.
                # print("X_train: ", X_train, X_train.shape)
                # print("combined_hws: ", combined_hws, combined_hws.shape)
                # print("joint_hw_train: ", joint_hw_train, joint_hw_train.shape)
                if trainning_model == True:
                    model = train_pipeline_singletask_dl(config, epochs, X_profiling, Y=Y_train_combined_hws,
                                                         classes = classes,
                                                         device=device, model_type=model_type, loss_type=loss_type,
                                                         dropout=config["dropout"])
                    torch.save(model.state_dict(),
                               trained_model_root + "model_" + str(model_idx) + "_" + loss_type + ".pth")
                else:
                    num_sample_pts = X_profiling.shape[-1]
                    if model_type == "mlp":
                        model = MLP(config, loss_type, num_sample_pts, classes, config["dropout"]).to(device)

                    elif model_type == "cnn":
                        model = CNN(config, loss_type, num_sample_pts, classes, config["dropout"]).to(device)
                    model.load_state_dict(
                        torch.load(trained_model_root + "model_" + str(model_idx) + "_" + loss_type + ".pth",
                                   map_location=torch.device(device)))



                ######################################################################Phase 3 Compare Theoretical and Empirical Distribution #############################################################################################3
                ##############################################################################################################################################################################################
                predictions_wo_softmax = predict_attack_traces(model, X_attack, device, interval_nb_trace=100)
                print(model_type, " model_idx: ", model_idx, " loss_type:", loss_type)
                predictions = F.softmax(predictions_wo_softmax, dim=1)
                predictions = predictions.cpu().detach().numpy()
                jointed_predicted_hw = np.argmax(predictions, axis=1)
                if num_branch == 2:
                    preds0 = jointed_predicted_hw % (num_bits + 1)
                    preds1 = jointed_predicted_hw // (num_bits + 1)
                elif num_branch == 3:
                    preds0 = jointed_predicted_hw % (num_bits + 1)
                    preds1 = ((jointed_predicted_hw - preds0)// (num_bits + 1)) % (num_bits + 1)
                    preds2 = ((jointed_predicted_hw - preds0 - (num_bits + 1) * preds1) // (num_bits + 1) ** 2) % (num_bits + 1)
                print("predictions_wo_softmax:", predictions_wo_softmax, predictions_wo_softmax.shape)
                print("jointed_predicted_hw:", jointed_predicted_hw, jointed_predicted_hw.shape)
                print("preds0, preds1:", preds0, preds1)
                print("preds0, preds1:", preds0.shape, preds1.shape)
                predicted_hw = np.array([preds0, preds1]).T

                var_noise_attack = obtain_var_noise(X_attack)
                GE,NTGE,SR = perform_joint_attack(nb_traces_attacks, nb_attacks, predicted_hw, theoretical_histogram, var_noise_attack,correct_key, dataset)
                np.save(trained_model_root + f"GE_SR_{dataset}_{leakage}_{labeling_type}_model{model_idx}_{model_type}_epochs{epochs}_{loss_type}.npy", {"GE": GE,  "NTGE": NTGE, "SR": SR})
