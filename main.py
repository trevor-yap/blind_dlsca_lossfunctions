import os
import numpy as np
from matplotlib import pyplot as plt
from src._1_poi_selections import SOSD_SOST
from src.utils import load_chipwhisperer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = 'Chipwhisperer'
    leakage = "HW"

    result_root = "./Result/"
    save_root = result_root + "blind_" + dataset + "_"+ leakage + "/"
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
        root = "./"
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling,plt_attack), correct_key = load_chipwhisperer(root + data_root + '/', leakage_model=leakage)


    #PoI Selection
    variance_trace = np.var(X_profiling, axis = 0)
    mean_trace = np.mean(X_profiling, axis = 0)
    fig, ax = plt.subplots(figsize=(12, 9))
    x_axis = [i for i in range(variance_trace.shape[0])]
    ax.plot(x_axis, variance_trace, c="red", label="Variance")
    ax.plot(x_axis, mean_trace, c="blue", label="Mean")
    plt.savefig(image_root + 'Mean_Variance.png')