import numpy as np
import torch
import torch.nn.functional as F


# cls_model need to have weight attribute

class ConvneXt_with_Arcface(torch.nn.Module):

    def __init__(self, cls_model, in_feature, out_feature, margin=0.45, scale=64, eps=1e-6):

        super(ConvneXt_with_Arcface, self).__init__()
        self.cls_model = cls_model
        
        #shape and initialized weights
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.AF_linear = torch.nn.Linear(
            in_feature, out_feature, bias=False)
        with torch.no_grad():
            self.AF_linear.weight.copy_(self.cls_model.cls_layer.weight)
        #normalizer
        self.normalizer = F.normalize

        # hyperparameter
        self.margin = margin
        self.scale = scale
        self.eps = eps

    def forward(self, x, labels=None):
        # get the L2normalized feature vector/matrix and weights
        x = self.cls_model(x, return_feats=True)
        x = self.normalizer(x)
        with torch.no_grad():
            self.AF_linear.weight = torch.nn.Parameter(
                self.normalizer(self.AF_linear.weight))
            
        # conpute the arcCos to get the angle theta between W and x
        # Wx = ||W||||x||cos(\theta) and ||W|| = 1, ||x|| = 1 => cos(\theta) = Wx
        # clip the value to (-1,1) to prevent NaN when calculating arcCos
        cosine = self.AF_linear(x)
        torch.clamp(cosine, min=-1.0+self.eps, max=1.0-self.eps)
        theta = torch.acos(cosine)

        # add the margin, take log and multiply scale
        M = F.one_hot(labels, num_classes=self.out_feature) * self.margin
        theta += M
        logit = self.scale * torch.cos(theta)
        x = logit

        return x