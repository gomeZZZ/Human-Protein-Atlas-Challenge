import torch
from torch import nn
class F1_Loss(nn.Module):
    def __init__(self,eps=1e-8):
        super(F1_Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        '''F1 loss.
        Args:
          pred: (tensor) sized [N,D].
          target: (tensor) sized [N,D].
        Return:
          (tensor) F1_Loss.
        '''
        
        #inspired by https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
        tp = (pred * target).sum(dim=1)                 #True positives
        tn = ( (1-pred) * (1-target)).sum(dim=1)        #True negatives
        fp = (pred * (1-target)).sum(dim=1)             #False positives
        fn = ((1-pred) * target).sum(dim=1)             #False negatives

        p = tp.div(tp + fp + self.eps)                  #Precision
        r = tp.div(tp + fn + self.eps)                  #Recall

        f1 = (2 * p * r).div(p + r + self.eps)


        return 1 - torch.mean(f1)
