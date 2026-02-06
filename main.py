import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from src._1_poi_selections import PoI_Selection_AES
from src._2_labeling_options import labeling_traces
from src._3_DL_training import train_pipeline_singletask_dl
from src.net import create_hyperparameter_space, MLP, CNN
from src.utils import load_chipwhisperer, check_accuracy, predict_attack_traces, perform_attacks, NTGE_fn

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = 'Chipwhisperer'
    leakage = "HW"
    labeling_type = "MultiPointSlicing"
    epochs = 50
    nb_attacks = 100
    nb_traces_attacks = 1700
    poi_selection_mode = "Variance_Threshold" #Variance_Segment, Variance_Threshold





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
    print("save_root:", save_root)
    print("using cuda:", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if dataset == 'Chipwhisperer':
        root = "./../"
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling,plt_attack), correct_key = load_chipwhisperer(root + data_root, leakage_model=leakage)
        num_bits = 8
    classes = (num_bits+1)**2
    nb_poi = 50
    total_samplept = X_profiling.shape[1]
    number_of_traces = X_profiling.shape[0]
    L_profiling = np.array([Y_profiling, plt_profiling]).T
    #PoI Selection
    if dataset == "Chipwhisperer":
        poi_xors = PoI_Selection_AES(nb_poi, total_samplept, number_of_traces, X_profiling, L_profiling, image_root, plot_cpa_image=True)
    print("poi_xors:", poi_xors, poi_xors.shape)
    print(ok)
    Y_noisy =labeling_traces(X_profiling, poi_xors, num_bits, save_root, labeling_type, poi_selection_mode,save_labels=True)

    check_accuracy(Y_profiling, Y_noisy)


    ######################################################################Training DNN #############################################################################################3
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

                trained_model_root = result_root + f'result_{model_type}_{epochs}_{loss_type}/'
                if not os.path.exists(trained_model_root):
                    os.mkdir(trained_model_root)
                # Train a DNN.
                # print("X_train: ", X_train, X_train.shape)
                # print("combined_hws: ", combined_hws, combined_hws.shape)
                # print("joint_hw_train: ", joint_hw_train, joint_hw_train.shape)
                if trainning_model == True:
                    model = train_pipeline_singletask_dl(config, epochs, X_profiling, Y=Y_noisy,
                                                         classes=classes,
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

                # 4. Apply the Disinguisher

                predictions_wo_softmax = predict_attack_traces(model, X_attack, device, interval_nb_trace=100)
                print(model_type, " model_idx: ", model_idx, " loss_type:", loss_type)
                predictions = F.softmax(predictions_wo_softmax, dim=1)
                predictions = predictions.cpu().detach().numpy()
