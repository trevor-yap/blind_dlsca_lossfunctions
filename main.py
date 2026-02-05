import os
import numpy as np
from matplotlib import pyplot as plt

from src._2_labeling_options import labeling_traces
from src.utils import load_chipwhisperer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = 'Chipwhisperer'
    leakage = "HW"
    labeling_type = "MultiPointSlicing"
    poi_selection_mode = "Variance_Threshold"
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

    if dataset == 'Chipwhisperer':
        root = "./../"
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling,plt_attack), correct_key = load_chipwhisperer(root + data_root, leakage_model=leakage)
        num_bits = 8
    classes = num_bits+1

    #PoI Selection
    if poi_selection_mode == "Variance_Threshold":
        variance_trace = np.var(X_profiling, axis = 0)
        mean_trace = np.mean(X_profiling, axis = 0)

        save_fig_poi_var = False
        if save_fig_poi_var == True:
            fig, ax = plt.subplots(figsize=(12, 9))
            x_axis = [i for i in range(variance_trace.shape[0])]
            ax.plot(x_axis, variance_trace, c="red", label="Variance")
            plt.savefig(image_root + 'Variance.png')
            plt.close(fig)
        poi_highest_variance = np.argmax(variance_trace)
        print(poi_highest_variance)
        poi_xors = np.array([i for i in range(poi_highest_variance, X_profiling.shape[1],1)])

    Y_noisy =labeling_traces(X_profiling, poi_xors, num_bits, save_root, labeling_type, poi_selection_mode,save_labels=True)

    print("Y_noisy:", Y_noisy)

