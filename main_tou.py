from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from touNet import TouNet
from data import get_training_set, get_eval_set
import pdb
import socket
import time
import math
import skimage
from skimage import measure
import numpy as np

def print_log(text):
    try:
        f = open('log.txt', 'a+', encoding='utf-8')
    except IOError:
        f = open('log.txt', 'w', encoding='utf-8')
    f.write(text+'\n')
    f.close()

def print_test_log(text):
    try:
        f = open('test_log.txt', 'a+', encoding='utf-8')
    except IOError:
        f = open('test_log.txt', 'w', encoding='utf-8')
    f.write(text+'\n')
    f.close()

def Huber(y_true, y_pred, delta):
    abs_error = torch.abs(y_pred - y_true)
    quadratic = torch.min(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=4, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-5, help='LearningRate.Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='../data/train')
parser.add_argument('--test_dir', type=str, default='../data/val')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='TouNet')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--pretrained_sr',default='4x_ubuntu-W580-G20TouNetF7_epoch_4.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def test(epoch, testing_data_loader):
    epoch_loss = 0
    model.train()
    avg_psnr = 0
    for iteration, batch in enumerate(testing_data_loader, 1):
        t0 = time.time()
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]
        prediction = model(input, neigbor, flow)
        if opt.residual:
            prediction = prediction + bicubic
        loss = criterion(prediction, target)
        epoch_loss += loss.data[0]
        psnr_score = 0
        for (pred, tar) in zip(prediction, target):
            pred = pred.cpu()
            pred = pred.squeeze().detach().numpy().astype(np.float32)
            pred = pred * 255
            
            tar = tar.cpu()
            tar = tar.squeeze().numpy().astype(np.float32)
            tar = tar * 255
            psnr_score += measure.compare_psnr(pred, tar, 255)
        psnr_score /= len(prediction)
        avg_psnr += psnr_score*len(prediction)
        t1 = time.time()
        print_test_log("===> Epoch[{}]({}/{}): Loss: {:.6f} || Timer: {:.6f} sec. || PSNR: {:.6f}".format(epoch, iteration, len(testing_data_loader), loss.data[0], (t1 - t0), psnr_score))
        print("===> Epoch[{}]({}/{}): Loss: {:.6f} || Timer: {:.6f} sec. || PSNR: {:.6f}".format(epoch, iteration, len(testing_data_loader), loss.data[0], (t1 - t0), psnr_score))
    avg_psnr /= len(testing_data_loader)

    print_test_log("===> Epoch[{}]({}/{}): Loss: {:.6f} || Timer: {:.6f} sec. || PSNR: {:.6f}".format(epoch, iteration, len(testing_data_loader), loss.data[0], (t1 - t0), avg_psnr))
    print("===> Epoch[{}]({}/{}): Loss: {:.6f} || Timer: {:.6f} sec. || PSNR: {:.6f}".format(epoch, iteration, len(testing_data_loader), loss.data[0], (t1 - t0), avg_psnr))


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        t0 = time.time()
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        optimizer.zero_grad()
        prediction = model(input, neigbor, flow)
        
        if opt.residual:
            prediction = prediction + bicubic
            
        #loss = Huber(prediction, target, 0.05)
        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        psnr_score = 0
        for (pred, tar) in zip(prediction, target):
            pred = pred.cpu()
            pred = pred.squeeze().detach().numpy().astype(np.float32)
            pred = pred * 255
            
            tar = tar.cpu()
            tar = tar.squeeze().numpy().astype(np.float32)
            tar = tar * 255
            psnr_score += measure.compare_psnr(pred, tar, 255)
            #psnr_score += PSNR(pred, tar, shave_border=opt.upscale_factor)
        psnr_score /= len(prediction)
       
        print_log("===> Epoch[{}]({}/{}): Loss: {:.6f} || Timer: {:.6f} sec. || PSNR: {:.6f}".format(epoch, iteration, len(training_data_loader), loss.data[0], (t1 - t0), psnr_score))
        print("===> Epoch[{}]({}/{}): Loss: {:.6f} || Timer: {:.6f} sec. || PSNR: {:.6f}".format(epoch, iteration, len(training_data_loader), loss.data[0], (t1 - t0), psnr_score))

    print_log("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(training_data_loader)))



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    imdff = pred - gt
    print_log(str(np.mean(imdff ** 2)))
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def checkpoint(epoch):
    model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
#train_set = get_training_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
test_set = get_training_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
#test_set = get_training_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation)
#training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'TouNet':
    model = TouNet(num_channels=3, base_filter=192,  feat = 48, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor) 
    #model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor) 

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
'''
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    test(epoch)
    #train(epoch)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/4) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
'''
test_set = get_training_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
#test_set = get_training_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation)
#training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
test(1)
