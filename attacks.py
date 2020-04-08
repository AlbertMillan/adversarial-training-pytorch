import numpy as np
import torch
# from torchvision import transforms
import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cf


class Attacks:
    
    def __init__(self, model, eps):
        self.model = model.cuda()
        self.eps = eps
        
        self.freeze()
        
            
    def freeze(self):
        # Freeze weights, no need to adjust them
        for param in self.model.parameters():
            param.requires_grad = False
           

    def fast_pgd(self, x_batch, y_batch, max_iter):
        """
        Generates adversarial examples using  projected gradient descent (PGD)
        
        Input:
            - x_batch : batch images to compute adversaries 
            - y_batch : labels of the batch
            - max_iter : # of iterations to generate adversarial example (FGSM=1)
        
        Output:
            - x : batch containing adversarial examples
        """
        
        x = x_batch.clone().detach().requires_grad_(True).cuda()
        
        # Reshape to  [1,C,1,1] to enable broadcasintg
        alpha = (self.eps * cf.eps_size / max_iter)[np.newaxis,:,np.newaxis,np.newaxis] 
        alpha = torch.FloatTensor(alpha).cuda()
        
        for _ in range(max_iter):
            
            logits = self.model(x)
            loss = self.model.module.loss(logits, y_batch)
            
            loss.backward()
            
            # Get gradient
            x_grad = x.grad.data
            
            # Broad cast alpha to shape [N,C,H,W]
            x.data = x.data + alpha * torch.sign(x_grad)
            
            # Clamp data between valid ranges
            x.data[:,0,:,:].clamp_(min=cf.min_val[0], max=cf.max_val[0])
            x.data[:,1,:,:].clamp_(min=cf.min_val[1], max=cf.max_val[1])
            x.data[:,2,:,:].clamp_(min=cf.min_val[2], max=cf.max_val[2])
            
            x.grad.zero_()
            
        return x