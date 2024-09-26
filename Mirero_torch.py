import os
import random
from tqdm import tqdm

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
from multiprocessing import freeze_support
from torch.autogard import Variable


def train() :
    freeze_support()

    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])

    train_data = datasets.ImageFolder(root='', transform = transform)

    train_idx, valid_idx = train_test_split(np.arange(len(train_data)), test_size =0.2
                           random_state = 42, stratify = train_data_targets)

    batch_size = 16
    num_workers int(cpu_count() / 2)

    train_loader = DataLoader(train_data, batch_size = batch_size,
                     sampler = SubsetRandomSampler(train_idx), num_workers = num_workers)

    valid_loader = DataLoader(train_data, batch_size = batch_size,
                     sampler = SubsetRandomSampler(valid_idx), num_workers = num_workers)

    train_total = len(train_idx)
    valid_total = len(valid_idx)

    train_batches = len(train_loader)
    valid_batches = len(valid_loader)



    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.is_available()

    net = EfficientNet.from_pretrained('efficientnet-b3',num_classes = num_classes)

    net.fc = nn.Linear(1000,num_classes)
    net = net.to(device)

    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(params = net.parameters(), lr = 0.001, weight_decay = 0.0001)

    epoch = 100
    best_accuracy = 0

    def mixup_data(x,y, alpha = 0.3, use_cuda = True) :
        if alpha > 0 :
            lam = np.random.beta(alpha, alpha)
        else :
            lam = 1

        batch_size = x.size()[0]
        if use_cuda :
            index = torch.randperm(batch_size).cuda()
        else :
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b , lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam) :
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train(epoch) :
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0
        tqdm_dataset = tqdm(train_loader)
        for batch_idx, (inputs, targets) in enumerate(tqdm_dataset) :
            if device :
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets_a, targets_b, lam = mixup_data(inputs,targets, 0.2)

            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            loss = mixup_criterion(criterion, outputs,targets_a, targets_b, lam)
            tqdm_dataset.set_postfix({
                'Epoch' : epoch + 1
                'Loss' : '{:06f}'.format(loss.item()),})

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().item()
                        + (1 - lam) * predicted.eq(targets_b_data).cpu().sum().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return (train_loss/batch_idx, 100.*correct/total)


    def test(epoch) :
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        tqdm_dataset = tqdm(valid_loader)
        for batch_idx, (inputs,targets) in enumerate(tqdm_dataset) :
            if device :
                inputs, targets = input.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile = True), Variable(targets)
            outpus = net(inputs)

            loss = criterion(outpus, targets)

            tqdm_dataset.set_postfix({
                'Epoch' : epoch +1,
                'Loss' : '{:06f}'.format(loss.item()),})

      
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().item())
            
        acc  = 100.*correct/total
        return (test_loss/batch_idx, 100* correct/total)


    for epoch in range(0,epoch) :
        train_loss, train_acc = train(epoch)
        valid_loss, valid_acc = test(epoch)
        tqdm_dataset = tqdm(train_loader)

        print('train loss', train_loss, 'train acc', train_acc, 'valid loss', valid_loss, 'valid acc',valid acc)
        if valid_acc > best_accuracy :
            print(f'best acc : {best_accuracy} -> {valid_acc} ")
            best_accuracy = valid_acc
            path = ''
            torch.save(net.state_dict(), path)


if __name__ == '__main__' :
    train()
