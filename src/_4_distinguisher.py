import numpy as np
from tqdm import tqdm
import math
import cupy as cp
from src.utils import NTGE_fn



def attack_blind_log(hw, histogram, std1, std2): # slight modification of Clavier's work
    # hw is in format (no_traces, 2): predicted hw
    # histogram is list of n_classes, each is a numpy array of size (max_hw+1,max_hw+1), in this case list of 3329 arrays of size 17x17
    # [std1, std2] is the standard deviation of the noises in leakage point 1 and 2 (input, output)
    mle = np.zeros((len(histogram), hw.shape[0]))
    # For all keys
    for k in tqdm(range(len(histogram))):
        temp = 1
        dist = histogram[k]
        # For all attack traces
        for i in range(hw.shape[0]):
            temp2 = 0
            for m in range(dist.shape[0]):
                for y in range(dist.shape[1]):
                    pr1 = dist[m,y]
                    pr2 = 1/(std1*np.sqrt(2*np.pi))*np.exp(-0.5*((hw[i,0] - m)/std1)**2)
                    pr3 = 1/(std2*np.sqrt(2*np.pi))*np.exp(-0.5*((hw[i,1] - y)/std2)**2)
                    tpc = pr1*pr2*pr3
                    if tpc<0.00000001: # to avoid 0 error.... just set it to very small value
                        tpc=0.00000001
                    temp2+= tpc
            temp+=np.log10(temp2)
            mle[k, i] = temp
    return mle



def attack_blind_log_gpu(hw, histogram, std): # slight modification of Clavier's work
    # hw is in format (no_traces, 2): predicted hw
    # histogram is list of n_classes, each is a numpy array of size (max_hw+1,max_hw+1), in this case list of 3329 arrays of size 17x17
    # [std1, std2] is the standard deviation of the noises in leakage point 1 and 2 (input, output)
    # start = time.time()
    hw = cp.asarray(hw)
    hw0 = cp.asarray(hw[:, 0])
    hw1 = cp.asarray(hw[:, 1])
    mle = cp.zeros((len(histogram), hw.shape[0]))

    for k in range(len(histogram)):
        dist = cp.asarray(histogram[k])
        m, y = cp.meshgrid(cp.arange(dist.shape[0]), cp.arange(dist.shape[1]), indexing='ij')
        pr1 = dist[m, y]
        pr2 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw0[:, cp.newaxis, cp.newaxis] - m) / std) ** 2)
        pr3 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw1[:, cp.newaxis, cp.newaxis] - y) / std) ** 2)
        tpc = pr1 * pr2 * pr3
        tpc = cp.where(tpc < 0.00000001, 0.00000001, tpc)
        temp2 = cp.sum(tpc, axis=(1, 2))  # Sum over the dimensions of m and y
        mle[k, :] = cp.cumsum(cp.log10(temp2))  # Adjusted to prevent log(0)
    # end = time.time()
    # print("\nTime consumed by cupy: ", end-start)
    return mle.get()  # Convert back to NumPy array if necessary


def attack_blind_log_gpu_ascon(hw, histogram, std):
    # hw is in format (no_traces, 2): predicted hw
    # histogram is list of n_classes, each is a numpy array of size (max_hw+1,max_hw+1), in this case list of 3329 arrays of size 17x17
    # [std1, std2] is the standard deviation of the noises in leakage point 1 and 2 (input, output)
    # start = time.time()
    hw = cp.asarray(hw)
    hw0 = cp.asarray(hw[:, 0])
    hw1 = cp.asarray(hw[:, 1])
    hw2 = cp.asarray(hw[:, 2])
    mle = cp.zeros((len(histogram), hw.shape[0]))

    for k in range(len(histogram)):
        dist = cp.asarray(histogram[k])
        x = cp.arange(9)
        y = cp.arange(9)
        z = cp.arange(9)
        m1, m2, y = cp.meshgrid(x, y, z, indexing='ij')
        pr1 = dist[m1, m2, y]
        pr2 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw0[:, cp.newaxis, cp.newaxis, cp.newaxis] - m1) / std) ** 2)
        pr3 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw1[:, cp.newaxis, cp.newaxis, cp.newaxis] - m2) / std) ** 2)
        pr4 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw2[:, cp.newaxis, cp.newaxis, cp.newaxis] - y) / std) ** 2)
        tpc = pr1 * pr2 * pr3 * pr4
        tpc = cp.where(tpc < 0.00000001, 0.00000001, tpc)
        temp2 = cp.sum(tpc, axis=(1, 2, 3))  # Sum over the dimensions of m and y
        mle[k, :] = cp.cumsum(cp.log10(temp2))  # Adjusted to prevent log(0)
    # end = time.time()
    # print("\nTime consumed by cupy: ", end-start)
    return mle.get()  # Convert back to NumPy array if necessary

def perform_joint_attack(nb_traces_attack,nb_attacks, predicted_hw, theoretical_histogram, var_noise, correct_key, dataset):
    sr_order = 0  # todo: give whatever the sr order
    ge = np.zeros(nb_traces_attack)
    ge_rounds = np.zeros((nb_attacks, nb_traces_attack))
    success_rate_sum = np.zeros(nb_traces_attack)
    print("Attack Phase:")
    for i in tqdm(range(nb_attacks)):
        # print("using test set to attack")
        idx_trace = np.arange(predicted_hw.shape[0])
        np.random.shuffle(idx_trace)
        if dataset == "Ascon":
            mle = attack_blind_log_gpu_ascon(predicted_hw[idx_trace[:nb_traces_attack], :],
                                       theoretical_histogram,
                                       std=math.sqrt(var_noise))
        else:
            mle = attack_blind_log_gpu(predicted_hw[idx_trace[:nb_traces_attack], :],
                                       theoretical_histogram,
                                       std=math.sqrt(var_noise))
        final_rank = np.zeros(nb_traces_attack)
        for j in range(nb_traces_attack):
            tmp_idx = np.argsort(mle[:, j])[::-1]
            # print("tmp_idx:", tmp_idx)
            final_rank[j] = np.float32(np.where(tmp_idx == correct_key)[0][0])
            if final_rank[j] <= sr_order:
                success_rate_sum[j] += 1
        ge_rounds[i] = final_rank
        ge += final_rank

    ge = ge / nb_attacks
    success_rate = success_rate_sum / nb_attacks
    print("GE: ", ge)

    NTGE = NTGE_fn(ge)
    print("NTGE:", NTGE)
    return ge, NTGE, success_rate