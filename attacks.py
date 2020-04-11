import numpy as np
import torch
# from torchvision import transforms
import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cf


class Attacks:
    
    def __init__(self, model, eps, N_train, N_test):
        self.adv_examples = {'train': None, 'test': None}
        self.adv_labels = {'train': None, 'test': None}
        self.adv_stored = {'train': False, 'test': False}
        self.count = {'train': 0, 'test': 0}
        
        self.model = model.cuda()
        self.eps = eps
        
        self.freeze()
        self.reset_imgs(N_train, N_test)
        
            
    def freeze(self):
        """Freeze weights."""
        for param in self.model.parameters():
            param.requires_grad = False
            
    
    def reset_imgs(self, N_train, N_test):
        """ Resets new variable to store adversarial examples."""
        self.adv_examples['train'] = torch.zeros(N_train,3,32,32)
        self.adv_examples['test'] = torch.zeros(N_test,3,32,32)
        self.adv_labels['train'] = torch.zeros((N_train), dtype=torch.long)
        self.adv_labels['test'] = torch.zeros((N_test), dtype=torch.long)
        self.adv_stored['train'] = False
        self.adv_stored['test'] = False
        self.count['train'] = 0
        self.count['test'] = 0
        
    
    def set_stored(self, mode, is_stored):
        """ Defines whether adversarial examples have been stored."""
        # TODO: Health-check to test whether the last element in 'adv_examples' contains and adv
        self.adv_stored[mode] = is_stored
        
    
    def get_adversarial(self, batch_size, mode):
        """ Retrieve adversarial examples if they have already been generated."""
        # Restart at beggining if iteration if complete
        if self.count[mode] >= self.adv_examples[mode].size(0):
            self.count[mode] = 0
        
        x_batch = self.adv_examples[mode][(self.count[mode]):(self.count[mode] + batch_size)]
        y_batch = self.adv_labels[mode][(self.count[mode]):(self.count[mode] + batch_size)]
        self.count[mode] = self.count[mode] + batch_size
        
        return x_batch.cuda(), y_batch.cuda()
        
           

    def fast_pgd(self, x_batch, y_batch, max_iter, mode):
        """
        Generates adversarial examples using  projected gradient descent (PGD).
        If adversaries have been generated, retrieve them.
        
        Input:
            - x_batch : batch images to compute adversaries 
            - y_batch : labels of the batch
            - max_iter : # of iterations to generate adversarial example (FGSM=1)
            - mode : batch from 'train' or 'test' set
        
        Output:
            - x : batch containing adversarial examples
        """
        # Retrieve adversaries
        if self.adv_stored[mode]:
            return self.get_adversarial(x_batch.size(0), mode)
        
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
        
        # Store adversarial images to array
        self.adv_examples[mode][(self.count[mode]):(self.count[mode] + x.size(0))] = x.clone().detach()
        self.adv_labels[mode][(self.count[mode]):(self.count[mode] + x.size(0))] = y_batch.clone().detach()
        self.count[mode] = self.count[mode] + x.size(0)
        
        return x, y_batch