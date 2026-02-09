import math
import torch
import torch.nn.functional as F


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return mae.mean()

class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()
