import os
import random
from typing import Tuple, Sequence, Callable
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import albumentations as A

from torchinfo import summary

from torchvision import transforms
from torch.autograd import Variable

from model import *


class MnistModel_efficientb3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')

        self.backbone._fc = nn.Linear(1536, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.backbone._swish
        self.classifier = nn.Linear(512, 26)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.classifier(x)

        return x

class MnistModel_efficientb4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')

        self.backbone._fc = nn.Linear(1792, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.backbone._swish
        self.classifier = nn.Linear(512, 26)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.classifier(x)

        return x



class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                torch.stack([
                  p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups for p in group["params"]
                      if p.grad is not None
                    ]),
                    p=2
               )
        return norm

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
seed_everything(42)

class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

transforms_train = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(180, expand=False),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

trainset = MnistDataset('../train', '../dirty_mnist_2nd_answer.csv', transforms_train)

dataset_size = len(trainset)
validation_split = 0.1
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler, num_workers=8)
validation_loader = torch.utils.data.DataLoader(trainset, batch_size=16, sampler=valid_sampler, num_workers=4)
test_loader = DataLoader(testset, batch_size=16, num_workers=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MnistModel_efficientb4().to(device)
print(summary(model, input_size=(1, 3, 256, 256), verbose=0))


optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3)
scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=60,
                                          cycle_mult=1.0,
                                          max_lr=0.005,
                                          min_lr=0.0001,
                                          warmup_steps=12,
                                          gamma=1.0)
criterion = nn.MultiLabelSoftMarginLoss()



class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

num_epochs = 60

best_loss = 1e10
best_acc = 0
no_improvement = 0
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        images = images.to(device) 
        targets = targets.to(device)
        
        images, targets_a, targets_b, lam = mixup_data(images, targets)
        images, targets_a, targets_b = map(Variable, (images, targets_a, targets_b))

        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        #mixup_criterion(criterion, model(images), targets_a, targets_b, lam).backward()
        #optimizer.second_step(zero_grad=True)

        outputs = outputs > 0.5
        acc = (outputs == targets).float().mean()
        print(f'EPOCH: {epoch}/{num_epochs} | {i} / {len(train_loader)} | LOSS: {loss.item():.5f}, ACCURACY: {acc.item():.5f}')


  losses = []
    accuracies = []
    for i, (images, targets) in enumerate(validation_loader):
        model.eval()
        with torch.no_grad():
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
        print(f'EPOCH: {epoch}/{num_epochs} | {i} / {len(validation_loader)} | LOSS: {loss.item():.5f}, ACCURACY: {acc.item():.5f}')
        losses.append(loss.item())
        accuracies.append(acc.item())
    
    flag = 0
    if sum(losses) / len(losses) < best_loss:
        print("Best loss achieved at {}".format(sum(losses) / len(losses)))
        best_loss = sum(losses) / len(losses)
        torch.save(model.state_dict(), 'mixup_loss_model_effnetb4_nosam.pt')
        flag += 1
        no_improvement = 0

    if sum(accuracies) / len(accuracies) > best_acc:
        print("Best acc achieved at {}(Ignore outputs!)".format(sum(accuracies) / len(accuracies)))
        best_acc = sum(accuracies) / len(accuracies)
        torch.save(model.state_dict(), 'mixup_acc_model_effnetb4_nosam.pt')
        flag += 1
        no_improvement = 0
    
    if flag == 0:
        no_improvement += 1
    
    if no_improvement > 30:
        print("No improvement for 3 epochs, stopping")
        break

    scheduler.step()

print("Best acc achieved:", best_acc)
print("Best loss achieved:", best_loss)