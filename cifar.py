'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse

from models import *
from torch.autograd import Variable

from tensorboard_logger import configure, log_value
from utilities import *
start_time = time.time()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 & SVHN Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train (default: 100)')
parser.add_argument('--model', type=str, default='densenet_cifar10', help='cnn model')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--distribute', type=int, default=1, help='use multiple GPUs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# load dataset
if args.dataset == 'cifar10':
    from data_loader.cifar10_data_loader import *
elif args.dataset == 'svhn':
    from data_loader.svhn_data_loader import *
train_loader, val_loader = get_train_valid_loader(batch_size=args.batch_size)
test_loader = get_test_loader(batch_size=args.batch_size)

# logger
configure('log/{}/'.format(args.model))
csv_logger = open('log/{}/log.csv'.format(args.model), 'w')
print_logger = logger('log/{}.log'.format(args.model), False, False)

print_logger.info(vars(args))

# Model
model_map  = {'densenet_cifar10': get_cifar10_densenetBC_L100_k12,
              'densenet_cifar10_drop': get_cifar10_densenetBC_L100_k12_drop,
              'resnet_cifar10': get_cifar10_resnet_110,
              'resnet_cifar10_drop': get_cifar10_resnet_110_drop,
              'vgg_cifar10': get_cifar10_VGG19,
              'vgg_cifar10_drop': get_cifar10_VGG19_drop,
              'sumnet': SumNet121}
if args.resume:
    # Load checkpoint.
    print_logger.info('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}.ckpt'.format(args.model))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print_logger.info('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = SumNet121()
    net = model_map[args.model]()

print_logger.info('==> Number of params: {}'.format(
    sum([param.data.nelement() for param in net.parameters()])))

if use_cuda:
    net.cuda()
    if args.distribute:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, [int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)

# train/val/test one epoch
def run_epoch(epoch, data_loader, is_train=True):
    if is_train:
        print_logger.info('\nEpoch: %d' % epoch)
        net.train()
    else:
        net.eval()

    total_loss, correct_cnt, total_cnt = 0., 0, 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if is_train:
            optimizer.zero_grad()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        batch_size, batch_loss, batch_correct_cnt = targets.size(0), loss.data[0], predicted.eq(targets.data).cpu().sum()
        total_loss += batch_loss
        total_cnt += batch_size
        correct_cnt += batch_correct_cnt

        print_logger.info('%d %d Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (batch_idx, len(data_loader), loss.data[0], 100.*batch_correct_cnt/batch_size, correct_cnt, total_cnt))

    return (total_loss*args.batch_size)/len(data_loader), float(correct_cnt)/total_cnt


for epoch in range(start_epoch, start_epoch+args.epoch):
    # Run train/val/test set
    train_loss, train_acc = run_epoch(epoch, train_loader, is_train=True)
    val_loss, val_acc = run_epoch(epoch, val_loader, is_train=False)
    test_loss, test_acc = run_epoch(epoch, test_loader, is_train=False)
    scheduler.step(epoch=epoch)

    # Save loss/acc in Tensorboard
    log_value('train_loss', train_loss, epoch)
    log_value('train_acc', train_acc, epoch)
    log_value('val_loss', val_loss, epoch)
    log_value('val_acc', val_acc, epoch)
    log_value('test_loss', test_loss, epoch)
    log_value('test_acc', test_acc, epoch)

    # Save loss/acc in csv
    csv_logger.write('{},{},{},{},{},{},{}\n'.format(
        epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
    csv_logger.flush()

    # Save checkpoint.
    if val_acc > best_acc:
        print_logger.info('Saving Best Val {0}-{1} Model..'.format(val_acc, test_acc))
        state = {
            'net': net.module if (use_cuda and args.distribute) else net,
            'acc': val_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.ckpt'.format(args.model))
        best_acc = val_acc

csv_logger.close()
print_logger.info('Total Training Time: %s' %(timeSince(since=start_time)))
