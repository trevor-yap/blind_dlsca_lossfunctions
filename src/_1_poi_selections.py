import numpy as np
from matplotlib import pyplot as plt

def cpa_method(total_samplept, number_of_traces, label, traces):
    if total_samplept == 1:
        cpa = np.zeros(total_samplept)
        cpa[0] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces])[1, 0])
    else:
        cpa = np.zeros(total_samplept)
        print("Calculate CPA!!")
        # print(traces.shape)
        for t in range(total_samplept):
            cpa[t] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces, t])[1, 0])
    return cpa


def PoI_Selection_AES(nb_poi, total_samplept, number_of_traces, traces, labels, image_root, plot_cpa_image= False):
    cpa_xors = []
    poi_xors = []
    for i in range(2):
        cpa_xors.append(cpa_method(total_samplept, number_of_traces, labels[:, i], traces))
        poi_xors.append(np.argsort(cpa_xors[i])[::-1][:nb_poi])
    if plot_cpa_image == True:
        fig, ax = plt.subplots(figsize=(12, 9))
        x_axis = [i for i in range(total_samplept)]
        color = ["blue", "orange"]
        for j in range(2):
            ax.plot(x_axis, cpa_xors[j], c=color[j], label="$HW(m_" + str(j) + ")$")

        ax.set_xlabel('Sample points', fontsize=20)
        ax.set_ylabel('(Absolute) Correlation', fontsize=20)
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=4, prop={'size': 20})
        plt.savefig(image_root + 'CPA_xor_AES.png')
    return np.array(poi_xors)