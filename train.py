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


# from attack_tester import AttacksTester
from attacks import Attacks

from model import WideResNet
from config import *

os.environ["CUDA_VISIBLE_DEVICES"]="4,6"


MODE_PLAIN = 0
MODE_PGD = 1
MODE_CW = 2

TEST_PLAIN = 0
TEST_ADV = 1
TEST_BOTH = 2


class Classifier:
    """
    
    """
    
    
#     MODE_PLAIN, MODE_PGD, MODE_CW= 0, 1, 2
#     TEST_PLAIN, TEST_ADV, TEST_BOTH = 0, 1, 2
    
    def __init__(self, ds_name, ds_path, lr, iterations, batch_size, 
                 print_freq, k, eps, load_dir=None, load_name=None,
                 save_dir=None, attack=MODE_PLAIN, test_mode=TEST_PLAIN):
        
        # Load Data
        if ds_name == 'CIFAR10':
            self.train_data = torchvision.datasets.CIFAR10(ds_path, train=True, transform=normalize(), download=True)
            self.test_data = torchvision.datasets.CIFAR10(ds_path, train=False, transform=normalize(), download=True)
            
        
        # collate_fn
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size)
        
        # Other Variables
        self.save_dir = save_dir
        self.test_raw = (test_mode == TEST_PLAIN or test_mode == TEST_BOTH)
        self.test_adv = (test_mode == TEST_ADV or test_mode == TEST_BOTH)
        
        
        # Set Model Hyperparameters
        self.learning_rate = lr
        self.iterations = iterations
        self.print_freq = print_freq
        self.model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
        
        self.cuda = torch.cuda.is_available()
        
        if self.cuda:
            self.model = self.model.cuda()
#             self.model = torch.nn.DataParallel(self.model).cuda()

        # Define attack method
        self.is_adv_training = (attack != MODE_PLAIN)
        if self.is_adv_training:
            
            # Load pre-trained model
            adversarial_model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
            adversarial_model = torch.nn.DataParallel(adversarial_model).cuda()
            adversarial_model = self.load_checkpoint(adversarial_model, load_dir, load_name)
            
            # Define adversarial generator model
            self.adversarial_generator = Attacks(adversarial_model, eps)
            
            self.attack_fn = None
            if attack == MODE_PGD:
                self.attack_fn = self.adversarial_generator.fast_pgd
            elif attack == MODE_CW:
                self.attack_fn = self.adversarial_generator.carl_wagner

#         if is_loaded:
#             if mode == 'test':
#                 adv_generator = Attacks(self.model, self.test_loader, len(self.test_data), 
#                                         eps=eps, parent_folder=loc, folder=mode)
#             else:
#                 adv_generator = Attacks(self.model, self.train_loader, len(self.train_data), 
#                                         eps=eps, parent_folder=loc, folder=mode)
            
#             sys.exit()
    
    
    
    def train_step(self, x_batch, y_batch, optimizer, losses, top1, k=1):
        # Compute output for example
        logits = self.model(x_batch)
        loss = self.model.loss(logits, y_batch)

        # Update Mean loss for current iteration
        losses.update(loss.item(), x_batch.size(0))
        prec1 = accuracy(logits.data, y_batch, k=k)
        top1.update(prec1.item(), x_batch.size(0))
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        # Set grads to zero for new iter
        optimizer.zero_grad()
        
    
    def test_step(self, x_batch, y_batch, losses, top1, k=1):
        with torch.no_grad():
            logits = self.model(x_batch)
            loss = self.model.loss(logits, y_batch)

        # Update Mean loss for current iteration
        losses.update(loss.item(), x_batch.size(0))
        prec1 = accuracy(logits.data, y_batch, k=self.k)
        top1.update(prec1.item(), x_batch.size(0))
        
    
    
    def train(self, momentum, nesterov, weight_decay, max_iter=1):
        
        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        
        best_pred = 0.0
        
        end = time.time()
        
        for itr in range(self.iterations):
            
            self.model.train()
            
            optimizer = optim.SGD(self.model.parameters(), lr=compute_lr(self.learning_rate, itr), 
                                  momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
            
            losses = AverageMeter()
            batch_time = AverageMeter()
            top1 = AverageMeter()
            
            x_adv = None
            
            for i, (x, y) in enumerate(self.train_loader):

                x = x.cuda()
                y = y.cuda()

                # Train raw examples
                self.train_step(x, y, optimizer, losses, top1)
                
                # Train adversarial examples if applicable
                if self.is_adv_training:
                    x_adv = self.attack_fn(x, y, max_iter)
                    self.train_step(x_adv, y, optimizer, losses, top1)
                
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
            test_loss, test_prec1 = self.test(self.test_loader, max_iter)
            
            train_loss_hist.append(losses.avg)
            train_acc_hist.append(top1.avg)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_prec1)
            
            # Store best model
            is_best = best_pred < test_prec1
            self.save_checkpoint(is_best, (itr+1), self.model.state_dict(), self.save_dir)
            if is_best:
                best_pred = test_prec1
                
        return (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist)
              
    
    
    def test(self, batch_loader, max_iter=1):
        self.model.eval()
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        top1 = AverageMeter()
        
        end = time.time()
        
        for i, (x,y) in enumerate(batch_loader):
            
            x = x.cuda()
            y = y.cuda()
            
            # Test on adversarial
            if self.test_raw:
                self.test_step(x, y, losses, top1)
            
            # Test on adversarial examples
            if self.test_adv:
                x_adv = self.attack_fn(x, y, max_iter)
                self.test_step(x_adv, y, losses, top1)

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
                      
        

    def save_checkpoint(self, is_best, epoch, state, save_dir, base_name="chkpt_plain"):
        """Saves checkpoint to disk"""
        directory = save_dir
        filename = base_name + ".pth.tar"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, directory + base_name + '__model_best.pth.tar')
            
    
    def load_checkpoint(self, model, load_dir, load_name):
        """Load checkpoint from disk"""
        filepath = load_dir + load_name
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict)
            print("Loaded checkpoint...")
            return model
        
        print("Failed to load model. Exiting...")
        sys.exit(1)
                      


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training. See code for default values.')
    
    # STORAGE LOCATION VARIABLES
    parser.add_argument('--ds_name', default='CIFAR10', metavar='Dataset', type=str, help='Dataset name')
    parser.add_argument('--ds_path', default='datasets/', metavar='Path', type=str, help='Dataset path')
    parser.add_argument('--load_dir', '--ld', default='model_chkpt/chkpt_plain/', type=str, help='Path to Model')
    parser.add_argument('--load_name', '--ln', default='chkpt_plain.pth.tar', type=str, help='File Name')
    parser.add_argument('--save_dir', '--sd', default='model_chkpt/new/', type=str, help='Path to Model')
#     parser.add_argument('--save_name', '--mn', default='chkpt_plain.pth.tar', type=str, help='File Name')
    
    
    # MODEL HYPERPARAMETERS
    parser.add_argument('--lr', default=0.1, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=200, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--batch_size', default=128, metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--momentum', '--m', default=0.9, type=float, help='Momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (default: 10)')
    parser.add_argument('--topk', '-k', default=1, type=int, help='Compute accuracy over top k-predictions (default: 1)')
    
    
    
    # ADVERSARIAL GENERATOR PROPERTIES
    parser.add_argument('--eps', '-e', default=(8./255.), type=float, help='Epsilon (default: 8/255)')
    parser.add_argument('--attack', '--att', default=0, type=int, help='Attack Type (default: 0)')
    parser.add_argument('--max_iter', default=1, type=int, help='Iterations required to generate adversarial examples (default: 1)')
    parser.add_argument('--test_mode', default=0, type=int, help='Test on raw images (0), adversarial images (1) or both (2) (default: 0)')
    
    
    
#     parser.add_argument('--mode', '-m', default='train', type=str, help='Adversaries from train/test folder. (default: train)')
    
    
    args = parser.parse_args()
    
    classifier = Classifier(args.ds_name, args.ds_path, args.lr, args.itr, args.batch_size, 
                            args.print_freq, args.topk, args.eps, args.load_dir, args.load_name,
                            args.save_dir, args.attack, args.test_mode)
    
    print("==================== TRAINING ====================")
    
    train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = classifier.train(args.momentum,
                                                                                      args.nesterov, 
                                                                                      args.weight_decay,
                                                                                      max_iter=args.max_iter )
    
    model_type = ['plain','PGD','CW']
    
    np.save("results/train_loss__"+str(model_type[args.attack])+"__"+str(args.max_iter)+".npy", train_loss_hist)
    np.save("results/train_acc__"+str(model_type[args.attack])+"__"+str(args.max_iter)+".npy", train_acc_hist)
    np.save("results/test_loss__"+str(model_type[args.attack])+"__"+str(args.max_iter)+".npy", test_loss_hist)
    np.save("results/test_acc__"+str(model_type[args.attack])+"__"+str(args.max_iter)+".npy", test_acc_hist)
    
