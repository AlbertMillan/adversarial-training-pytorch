import numpy as np
import torch
import torchvision
from torch import optim
# from torchvision.datasets import CIFAR10
# from torchvision.models import wide_resnet50_2

import argparse
import time
import math
import shutil
import sys, os


from attacks import Attacks
from model import WideResNet
from config import *

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

class Classifier:
    
    def __init__(self, ds_name, ds_path, lr, iterations, batch_size, 
                 print_freq, k, eps, model_dir=None, model_name=None,
                 mode=None, loc=None):
        
        # Load Data
        if ds_name == 'CIFAR10':
            self.train_data = torchvision.datasets.CIFAR10(ds_path, train=True, transform=normalize(), download=True)
            self.test_data = torchvision.datasets.CIFAR10(ds_path, train=False, transform=normalize(), download=True)
            
        
        # collate_fn
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size)
        
        # Var storing
        self.model_dir = model_dir
        self.model_name = model_name
        
        # Load Model
        self.learning_rate = lr
        self.iterations = iterations
        self.print_freq = print_freq
        self.k = k
        self.model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
        
        self.cuda = torch.cuda.is_available()
        
        if self.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()
            is_loaded = self.load_checkpoint()
            
        if is_loaded:
            if mode == 'test':
                adv_generator = Attacks(self.model, self.test_loader, len(self.test_data), 
                                        eps=eps, parent_folder=loc, folder=mode)
            else:
                adv_generator = Attacks(self.model, self.train_loader, len(self.train_data), 
                                        eps=eps, parent_folder=loc, folder=mode)
            
            sys.exit()
    
    
    def train(self, momentum, nesterov, weight_decay, k=1):
        
        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        
        best_pred = 0.0
        
        end = time.time()
        
        for itr in range(self.iterations):
            
            self.model.train()
            
            optimizer = optim.SGD(self.model.parameters(), lr=compute_lr(self.learning_rate, itr), 
                                  momentum=momentum, nesterov=nesterov,
                                  weight_decay=weight_decay)
            
            losses = AverageMeter()
            batch_time = AverageMeter()
            top1 = AverageMeter()
            
            for i, (x, y) in enumerate(self.train_loader):

                x = x.cuda()
                y = y.cuda()

                # Set grads to zero for new iter
                optimizer.zero_grad()
                
                # Compute output
                logits = self.model(x)
                loss = self.model.module.loss(logits, y)
                
                # Update Mean loss for current iteration
                losses.update(loss.item(), x.size(0))
                prec1 = accuracy(logits.data, y, k=k)
                top1.update(prec1.item(), x.size(0))
                
                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()
#                 scheduler.step()
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                              itr, i, len(self.train_loader), batch_time=batch_time,
                              loss=losses, top1=top1))
            
            # evaluate on validation set
            test_loss, test_prec1 = self.test(self.test_loader)
            
            train_loss_hist.append(losses.avg)
            train_acc_hist.append(top1.avg)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_prec1)
            
            # Store best model
            is_best = best_pred < test_prec1
            self.save_checkpoint(is_best, (itr+1), self.model.state_dict())
            if is_best:
                best_pred = test_prec1
                
        return (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist)
              
    
    def test(self, batch_loader):
        self.model.eval()
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        top1 = AverageMeter()
        
        end = time.time()
        
        for i, (x,y) in enumerate(batch_loader):
            
            x = x.cuda()
            y = y.cuda()
            
            with torch.no_grad():
                logits = self.model(x)
                loss = self.model.module.loss(logits, y)

            # Update Mean loss for current iteration
            losses.update(loss.item(), x.size(0))
            prec1 = accuracy(logits.data, y, k=self.k)
            top1.update(prec1.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(batch_loader), batch_time=batch_time,
                          loss=losses, top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return (losses.avg, top1.avg)
                      
        

    def save_checkpoint(self, is_best, epoch, state, base_name="chkpt_plain"):
        """Saves checkpoint to disk"""
        directory = "chkpt/"
        filename = base_name + ".pth.tar"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, directory + base_name + '__model_best.pth.tar')
            
    
    def load_checkpoint(self):
        """Load checkpoint from disk"""
        filepath = self.model_dir + self.model_name
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.model.load_state_dict(state_dict)
            print("Loaded checkpoint...")
            return True

        return False
                      


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--ds_name', default='CIFAR10', metavar='Dataset', type=str, help='Dataset name')
    parser.add_argument('--ds_path', default='datasets/', metavar='Path', type=str, help='Dataset path')
    parser.add_argument('--lr', default=0.1, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=200, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--batch_size', default=128, metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--momentum', '--m', default=0.9, type=float, help='Momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (default: 10)')
    parser.add_argument('--topk', '-k', default=1, type=int, help='Compute accuracy over top k-predictions (default: 1)')
    parser.add_argument('--eps', '-e', default=(8./255.), type=float, help='Epsilon (default: 0.1)')
    parser.add_argument('--model_dir', '--md', default='chkpt_orgn/', type=str, help='Path to Model (default: chkpt_orgn/)')
    parser.add_argument('--model_name', '--mn', default='chkpt_plain.pth.tar', type=str, help='File Name (default: chkpt_plainmodel_best.pth.tar)')
    parser.add_argument('--mode', '-m', default='train', type=str, help='Adversaries from train/test folder. (default: train)')
    parser.add_argument('--location', '--loc', default='cifar-10-fgsm', type=str, help='Adversaries from train/test folder. (default: train)')
    
    
    args = parser.parse_args()
    
    classifier = Classifier(args.ds_name, args.ds_path, args.lr, args.itr, args.batch_size, 
                            args.print_freq, args.topk, args.eps, args.model_dir, args.model_name,
                            args.mode, args.location)
    
    print("==================== TRAINING ====================")
    
    train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = classifier.train(args.momentum, args.nesterov, args.weight_decay)
    np.save("results/train_loss__plain.npy", train_loss_hist)
    np.save("results/train_acc__plain.npy", train_acc_hist)
    np.save("results/test_loss__plain.npy", test_loss_hist)
    np.save("results/test_acc__plain.npy", test_acc_hist)
    
