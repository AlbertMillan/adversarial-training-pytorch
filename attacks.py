import numpy as np
import torch
from torch.autograd import Variable
# from torchvision import transforms
import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cf



class Attacks:
    
    def __init__(self, model, raw_data_loader, N, eps, parent_folder=None, folder=None):
        self.print_freq = 10
        self.parent_folder = parent_folder
        self.folder = folder
        
        self.count = 0
        
        # Save the advesarial example in tensor format
        self.x_adv_arr = np.zeros((N,3,32,32))
        self.y_label_arr = np.zeros(N)
        
        self.eps = eps
        self.model = model
        self.data_loader = raw_data_loader
        
        self.run_model()
        # Maybe I can create an external class to handle this...
        self.save_arr(loc=self.parent_folder)
        
    
    def run_model(self):
        
        # Perform this if there is no folder created in desired path and 
        # the number of images does not match that in the folder.
        arr = []
        
        acc_org = cf.AverageMeter()
        acc_adv = cf.AverageMeter()
        
        # Freeze weights, no need to compute them
        for param in self.model.parameters():
            param.requires_grad = False
        
        
        for i, (x, y) in enumerate(self.data_loader):
            
            x = x.cuda().requires_grad_()
            y = y.cuda()
            
            # Generate Adversarial
#             x_adv = self.pgd(x, y, max_iter=20)
            x_adv = self.fast_pgd(x, y, max_iter=100)
            
            # Compute output of adversarial and input
            logits = self.model.forward(x)
            logits_adv = self.model.forward(x_adv)
            
            
            prec_org = cf.accuracy(logits, y)
            prec_adv = cf.accuracy(logits_adv, y)
            acc_org.update(prec_org.item(), x.size(0))
            acc_adv.update(prec_adv.item(), x_adv.size(0))
            
            
            # Test: Display Image
#             self.test(x, torch.sign(noise), self.eps)

            if i % self.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Acc_org: {acc_org.val:.3f} ({acc_org.avg:.3f})\t'
                      'Acc_adv: {acc_adv.val:.3f} ({acc_adv.avg:.3f})'.format(
                          i, len(self.data_loader), 
                          acc_org=acc_org,
                          acc_adv=acc_adv))

        # Print Accuracy Results
        print(' * Acc Org: {acc_org.avg:.3f}'.format(acc_org=acc_org))
        print(' * Acc Adv: {acc_adv.avg:.3f}'.format(acc_adv=acc_adv))
        
        
        
    def fast_pgd(self, x_batch, y, max_iter):
        
        x = x_batch.clone().detach().requires_grad_(True).cuda()
        
        # Reshape to  [1,C,1,1] to enable broadcasintg
        alpha = (self.eps * cf.eps_size / max_iter)[np.newaxis,:,np.newaxis,np.newaxis] 
        alpha = torch.FloatTensor(alpha).cuda()
        
        for _ in range(max_iter):
            
            logits = self.model(x)
            loss = self.model.module.loss(logits, y)
            
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
        
        # Store adversarial example & label in all array
        self.x_adv_arr[self.count:(self.count + len(y))] = x.cpu().data.numpy()
        self.y_label_arr[self.count:(self.count + len(y))] = y.cpu().data.numpy()
        
        self.count = self.count + len(y)
            
        return x
    
    
# ============================= SUPPORT FUNCTIONS =================================

    def compute_scaled_update(self, x, x_grad, alpha):
        x_scaled = self.scale_batch(x)
        
        output = x_scaled + alpha * torch.sign(x_grad)
        
        output = torch.clamp(output, min=0.0, max=1.0)
        
        out = self.gaussian_normalize(output)
        
        return out
        
    
    
    def scale_batch(self, x):
        x_copy = x.clone().detach()
        
        x_copy[:,0,:,:] = x_copy[:,0,:,:] * cf.std[0] + cf.mean[0]
        x_copy[:,1,:,:] = x_copy[:,1,:,:] * cf.std[1] + cf.mean[1]
        x_copy[:,2,:,:] = x_copy[:,2,:,:] * cf.std[2] + cf.mean[2]
        
        x_copy = torch.clamp(x_copy, min=0.0, max=1.0)
        
        return x_copy
        
        
    def gaussian_normalize(self, x):
        
        x_copy = x.clone().detach()
        
        x_copy[:,0,:,:] = (x_copy[:,0,:,:] - cf.mean[0]) / cf.std[0]
        x_copy[:,1,:,:] = (x_copy[:,1,:,:] - cf.mean[1]) / cf.std[1]
        x_copy[:,2,:,:] = (x_copy[:,2,:,:] - cf.mean[2]) / cf.std[2]
        
        return x_copy
    
    def save_arr(self, root='datasets/', loc='cifar-10-fgsm'):
        path = root + loc
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        np.save(os.path.join(path, 'x_adv_'+self.folder+'.npy'), self.x_adv_arr)
        np.save(os.path.join(path,'y_'+self.folder+'.npy'), self.y_label_arr)
    
    
# =========================== SLOW METHODS: TESTING ============================
            

    def fgsm(self, x, y, max_iter):
        
        eps = self.eps * cf.eps_size
        
        alpha = eps 
        
        for _ in range(max_iter):
            
            logits = self.model(x)
            loss = self.model.module.loss(logits, y)
            
            loss.backward()
            
            x_grad = x.grad.data
            
            # Three ways to compute eps
#             x.data = self.compute_scaled_update(x, x_grad, alpha)
            

            x.data[:,0,:,:] = x.data[:,0,:,:] + alpha[0] * torch.sign(x_grad[:,0,:,:])
            x.data[:,1,:,:] = x.data[:,1,:,:] + alpha[1] * torch.sign(x_grad[:,1,:,:])
            x.data[:,2,:,:] = x.data[:,2,:,:] + alpha[2] * torch.sign(x_grad[:,2,:,:])
            
            x.grad.zero_()
            
        return x
    
    
    def pgd(self, x_batch, y, max_iter=20):
        
        res = torch.zeros((x_batch.size()))
        
        for i, x_init in enumerate(x_batch):
            
            x_img = x_init.unsqueeze(0).clone().detach().requires_grad_(True).cuda()
            y_t = y[i].unsqueeze(0)
            
            res[i] = self.fgsm(x_img, y_t, max_iter)
            
        return res