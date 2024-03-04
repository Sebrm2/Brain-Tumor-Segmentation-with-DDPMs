import copy
import time
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from Dataloader import BRATSDataset
from config import model_config
from utils.Evaluator import Evaluator
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights
import utils.scoring as scoring

# Load the configuration
args = model_config()

# Check cuda availability
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set random seed for GPU or CPU
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Set arguments for Dataloaders
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Load the data for training
directory = 'BraTS2023_StructuredData/BraTS2023_AxialSlices'
train_loader = DataLoader(
    BRATSDataset(data_path=directory, dataset_type='train',
                 transform=transforms.Compose([
                     transforms.ToTensor()
                 ]), limit_samples=3000),
    args.batch_size,
    shuffle=True, **kwargs)

# Create a DataLoader instance for test
test_loader = DataLoader(
    BRATSDataset(data_path=directory,
                 dataset_type='test',
                 transform=transforms.Compose([
                     transforms.ToTensor()
                 ]), limit_samples=100),
    args.batch_size,
    shuffle=False,
    **kwargs)

# Load the model
load_model = False
model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, pretrained=True, progress=True)
cp_model = copy.deepcopy(model)

# Modify the classifier layers
num_classes = 2  # Change this based on your dataset
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

if args.cuda:
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model)
    model.cuda()

# Optimizer & loss
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()
metrics = Evaluator()

# Learning rate scheduler
scheduler = StepLR(optimizer,step_size=args.step_size, gamma=args.gamma)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).long().squeeze_(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output['out'], target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(epoch):
    DSC =[]
    model.eval()
    metrics.reset()
    test_loss = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).long().squeeze_(1)
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output['out'], target).item()
        pred = output['out'].cpu()
        pred = F.softmax(pred, dim=1).numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        metrics.add_batch(target, pred)

        lista = []
        for i in range(target.shape[0]):
            lista.append(scoring.dice_coef(target[i], pred[i]))
        DSC.extend(lista)
    
    Dice = np.mean(DSC)
    with open(f"dice{args.model}.txt", "a") as f:
        f.write(f"Test\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Dice: {Dice}\n")
    print('Test:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(test_loader.dataset))) # type: ignore
    print("Dice:{}".format(Dice))

    return test_loss

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Main
if __name__ == '__main__':
    best_loss = None
    if load_model:
        best_loss = test(0)
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(epoch)
            test_loss = test(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '.format(
                epoch, time.time() - epoch_start_time))
            print('-' * 89)

            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                #with open(args.save, 'wb') as fp:
                #    state = model.state_dict()
                #    torch.save(state, fp)

            # Update the learning rate with the scheduler
            scheduler.step()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')