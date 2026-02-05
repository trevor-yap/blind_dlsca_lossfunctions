import numpy as np
import torch
from torch.utils.data import Dataset


class Blind_Single_Dataset(Dataset):
    def __init__(self, X, Y, transform = None):
        self.X = np.expand_dims(X, 1)
        self.Y = Y
        self.transform = transform
        # print("blind dataset self.Y", self.Y.shape)
        # print("blind dataset self.Y_actual", self.Y_actual.shape)
        # Convert the label to classes**2:
        # self.Y = 9 * self.Y[:, 0] + self.Y[:, 1]
        # self.Y_actual = 9 * self.Y_actual[:, 0] + self.Y_actual[:, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X[idx]
        sensitive = self.Y[idx]
        # actual_label = self.Y_actual[idx]
        # plaintext = self.Plaintext[idx]
        sample = {'trace': trace, 'sensitive': sensitive} #, , 'actual_label': actual_label'plaintext': plaintext}
        # print(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor_trace_blind(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # trace, label, plaintext= sample['trace'], sample['sensitive'], sample['plaintext']
        trace, label= sample['trace'], sample['sensitive']#, sample['plaintext']
        # actual_label= sample['actual_label']#, sample['plaintext']
        return torch.from_numpy(trace).float(), torch.from_numpy(np.array(label)).float()  # , torch.from_numpy(np.array(actual_label)).float()