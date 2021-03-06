import torch
import torch.nn as nn 

class BiSiNetLoss(nn.Module):
    def __init__(training=False):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.training = training
        pass
    def forward(self, pred, target, features_1=0, features_2=0):
        if self.training:
            loss_1 = self.loss(pred, target)
            loss_2 = self.loss(features_1, target)
            loss_3 = self.loss(features_2, target)
            return (loss_1 + loss_2 + loss_3) / 3
        return self.loss(pred, target)
