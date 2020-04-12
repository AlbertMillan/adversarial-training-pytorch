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

# os.environ["CUDA_VISIBLE_DEVICES"]="1,3"


MODE_PLAIN = 0
MODE_PGD = 1
MODE_CW = 2

RAW = 0
ADV = 1
BOTH = 2

TRAIN_AND_TEST = 0
TEST = 1



class Classifier:
    """
    
    """
    
    
#     MODE_PLAIN, MODE_PGD, MODE_CW= 0, 1, 2 
#     TEST_PLAIN, TEST_ADV, TEST_BOTH = 0, 1, 2
    
    def __init__(self, ds_name, ds_path, lr, iterations, batch_size, 
                 print_freq, k, eps, adv_momentum, load_dir=None, load_name=None,
                 load_adv_dir=None, load_adv_name=None, save_dir=None, 
                 attack=MODE_PLAIN, train_mode=RAW, test_mode=RAW, mode=TRAIN_AND_TEST):
        
        # Load Data
        if ds_name == 'CIFAR10':
            self.train_data = torchvision.datasets.CIFAR10(ds_path, train=True, transform=train_augmentation(), download=True)
            self.test_data = torchvision.datasets.CIFAR10(ds_path, train=False, transform=test_augmentation(), download=True)
            
        
        # collate_fn
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size)
        
        # Other Variables
        self.save_dir = save_dir
        self.train_raw = (train_mode == RAW or train_mode == BOTH)
        self.train_adv = (train_mode == ADV or train_mode == BOTH)
        self.test_raw = (test_mode == RAW or test_mode == BOTH)
        self.test_adv = (test_mode == ADV or test_mode == BOTH)
        
        
        # Set Model Hyperparameters
        self.learning_rate = lr
        self.iterations = iterations
        self.print_freq = print_freq
        self.model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
        
        self.cuda = torch.cuda.is_available()
        
        if self.cuda:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model).cuda()
            
            # Load pre-trained model if we just want to evaluate model on test set
            if mode == TEST:
                self.model = self.load_checkpoint(self.model, load_dir, load_name)

        # Define attack method
        if self.train_adv:
            
            # Load pre-trained model
            adversarial_model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
            adversarial_model = torch.nn.DataParallel(adversarial_model).cuda()
            adversarial_model = self.load_checkpoint(adversarial_model, load_adv_dir, load_adv_name)
            
            # Define adversarial generator model
            self.adversarial_generator = Attacks(adversarial_model, eps, len(self.train_data), len(self.test_data), adv_momentum)
#             self.test_adversarial_generator = Attacks(adversarial_model, eps, len(self.train_data))
            
            self.attack_fn = None
            if attack == MODE_PGD:
                self.attack_fn = self.adversarial_generator.fast_pgd
#                 self.test_attack_fn = self.test_adversarial_generator.fast_pgd
            elif attack == MODE_CW:
                self.attack_fn = self.adversarial_generator.carl_wagner
#                 self.test_attack_fn = self.test_adversarial_generator.carl_wagner
                

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
        loss = self.model.module.loss(logits, y_batch)

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
            loss = self.model.module.loss(logits, y_batch)

        # Update Mean loss for current iteration
        losses.update(loss.item(), x_batch.size(0))
        prec1 = accuracy(logits.data, y_batch, k=k)
        top1.update(prec1.item(), x_batch.size(0))
        
    
    
    def train(self, momentum, nesterov, weight_decay, train_max_iter=1, test_max_iter=1):
        
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
                if self.train_raw:
                    self.train_step(x, y, optimizer, losses, top1)
                
                # Train adversarial examples if applicable
                if self.train_adv:
                    x_adv, y_adv = self.attack_fn(x, y, train_max_iter, mode='train')
                    self.train_step(x_adv, y_adv, optimizer, losses, top1)
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                              itr, i, len(self.train_loader), batch_time=batch_time,
                              loss=losses, top1=top1))
            
            # Evaluate on validation set
            test_loss, test_prec1 = self.test(self.test_loader, test_max_iter)
            
            train_loss_hist.append(losses.avg)
            train_acc_hist.append(top1.avg)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_prec1)
            
            # Store best model
            is_best = best_pred < test_prec1
            self.save_checkpoint(is_best, (itr+1), self.model.state_dict(), self.save_dir)
            if is_best:
                best_pred = test_prec1
            
            # Adversarial examples generated on the first iteration. No need to compute them again.
            if self.train_adv:
                self.adversarial_generator.set_stored('train', True)
                
        return (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist)
              
    
    
    def test(self, batch_loader, test_max_iter=1):
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
                x_adv, y_adv = self.attack_fn(x, y, test_max_iter, mode='test')
                self.test_step(x_adv, y_adv, losses, top1)

            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(batch_loader), batch_time=batch_time,
                          loss=losses, top1=top1))
        
        # Test adversarial examples generated on the first iteration. No need to compute them again.
        if self.test_adv:
            self.adversarial_generator.set_stored('test', True)

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
    parser.add_argument('--load_dir', '--ld', default='model_chkpt_new/chkpt_plain/', type=str, help='Path to Model')
    parser.add_argument('--load_name', '--ln', default='chkpt_plain__model_best.pth.tar', type=str, help='File Name')
    parser.add_argument('--load_adv_dir', '--lad', default='model_chkpt_new/chkpt_plain/', type=str, help='Path to Model')
    parser.add_argument('--load_adv_name', '--lan', default='chkpt_plain__model_best.pth.tar', type=str, help='File Name')
    parser.add_argument('--save_dir', '--sd', default='model_chkpt_new/new/', type=str, help='Path to Model')
#     parser.add_argument('--save_name', '--mn', default='chkpt_plain.pth.tar', type=str, help='File Name')
    
    
    # MODEL HYPERPARAMETERS
    parser.add_argument('--lr', default=0.1, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=76, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--batch_size', default=64, metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--momentum', '--m', default=0.9, type=float, help='Momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, help='weight decay (default: 2e-4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (default: 10)')
    parser.add_argument('--topk', '-k', default=1, type=int, help='Compute accuracy over top k-predictions (default: 1)')
    
    
    
    # ADVERSARIAL GENERATOR PROPERTIES
    parser.add_argument('--eps', '-e', default=(8./255.), type=float, help='Epsilon (default: 8/255)')
    parser.add_argument('--adv_momentum', default=None, type=float, help='Momentum for adversarial training (default: 8/255)')
    parser.add_argument('--attack', '--att', default=0, type=int, help='Attack Type (default: 0)')
    parser.add_argument('--train_max_iter', default=1, type=int, help='Iterations required to generate adversarial examples  during training (default: 1)')
    parser.add_argument('--test_max_iter', default=1, type=int, help='Iterations required to generate adversarial examples during testing (default: 1)')
    
    parser.add_argument('--train_mode', default=0, type=int, help='Train on raw images (0), adversarial images (1) or both (2) (default: 0)')
    parser.add_argument('--test_mode', default=0, type=int, help='Test on raw images (0), adversarial images (1) or both (2) (default: 0)')
    
    # OTHER PROPERTIES
    parser.add_argument('--gpu', default="0,1", type=str, help='GPU devices to use (0-7) (default: 0,1)')
    parser.add_argument('--mode', default=0, type=int, help='Wether to perform test without trainig (default: 0)')
    

    
    
#     parser.add_argument('--mode', '-m', default='train', type=str, help='Adversaries from train/test folder. (default: train)')
    
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    classifier = Classifier(args.ds_name, args.ds_path, args.lr, args.itr, args.batch_size, 
                            args.print_freq, args.topk, args.eps, args.adv_momentum,
                            args.load_dir, args.load_name, args.load_adv_dir, args.load_name, 
                            args.save_dir, args.attack, args.train_mode, args.test_mode, args.mode)
    
    print("==================== TRAINING ====================")
    
    if args.mode == TRAIN_AND_TEST:
        train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = classifier.train(args.momentum,
                                                                                          args.nesterov, 
                                                                                          args.weight_decay,
                                                                                          train_max_iter=args.train_max_iter,
                                                                                          test_max_iter=args.test_max_iter)

        model_type = ['plain','PGD','CW']

        np.save("results_2/train_loss__"+str(model_type[args.attack])+"__"+str(args.test_max_iter)+".npy", train_loss_hist)
        np.save("results_2/train_acc__"+str(model_type[args.attack])+"__"+str(args.test_max_iter)+".npy", train_acc_hist)
        np.save("results_2/test_loss__"+str(model_type[args.attack])+"__"+str(args.test_max_iter)+".npy", test_loss_hist)
        np.save("results_2/test_acc__"+str(model_type[args.attack])+"__"+str(args.test_max_iter)+".npy", test_acc_hist)
    
    print("==================== TESTING ====================")
    
    if args.mode == TEST:
        classifier.test(classifier.test_loader, args.test_max_iter)
    
