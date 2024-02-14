##################################################################################################
#                                     IMPORT LIBRARIES                                           #
##################################################################################################
print("\033[1;35;40m Importing libraries...\033[0m")

import time
import pickle
import argparse
import os.path as osp
import copy
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import datasets, transforms, models
from Dataloader import BRATSDataset
from Evaluator import Evaluator
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as sch
import utils.scoring as scoring


##################################################################################################
#                                     ADD ARGUMENTS                                              #
##################################################################################################
print("\033[1;35;40m Adding arguments...\033[0m")
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='models/modelFCN101monai_CORONAL.pt', #TODO: Chage according to the model
                    help='file on which to save model weights')
parser.add_argument('--model', type=str, default='FCN101monai_CORONAL', #TODO: Chage according to the model (file name changes)
                    help='name of the model to use (default: UNET)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


##################################################################################################
#                                     LOAD THE FOLDERS                                           #
##################################################################################################

print("\033[1;35;40m Loading the folders...\033[0m")
directory = 'BraTS2023_StructuredData/BraTS2023_AxialSlices' # TODO: change to Axial, Coronal or Sagittal
# Create a DataLoader instance for train 
train_loader = DataLoader(
    BRATSDataset(data_path = directory, dataset_type='train',
                 transform=transforms.Compose([
                       transforms.ToTensor()
                   ])), 
    args.batch_size,
    shuffle=True, **kwargs)
# Create a DataLoader instance for test
test_loader = DataLoader(
    BRATSDataset(data_path = directory,
                  dataset_type='test',
                  transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    args.batch_size,
    shuffle=False,
    **kwargs)

##################################################################################################
#                                     GENERATE MODEL                                             #
##################################################################################################

print("\033[1;35;40m Generating the model...\033[0m")
"""                           MODEL DEEPLABV3 WITH BACKBONE RESNET50                           """

model = models.segmentation.deeplabv3_resnet101(pretrained=True)
breakpoint()
cp_model = copy.deepcopy(model)

#model.conv = nn.Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1)) #TODO: this is not for deeplab3 maybe?
# change the number of input channels of the first conv layer
#model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
model.classifier[4] = nn.Conv2d(256, 4, kernel_size=(1,1), stride=(1, 1)) # TODO: change to 256 if it is deeplabv3_resnet50, or 512 if it is fcn_resnet50
model.aux_classifier[4] = nn.Conv2d(256, 4, kernel_size=(1,1), stride=(1, 1))

# define state_dict
sd_model = model.state_dict()
#sd_cp_model = cp_model.state_dict()

# average the weights over the channel dimension
#sd_model['backbone.conv1.weight'] = torch.mean(sd_cp_model['backbone.conv1.weight'], dim = 1, keepdim = True)

# update state_dict
model.load_state_dict(sd_model)
#cp_model = copy.deepcopy(model) #we don't use it 

if args.cuda:
    model.cuda()


load_model = False
"""
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True
"""

##################################################################################################
#                                     OPTIMIZER & LOSS                                           #
##################################################################################################

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = sch.StepLR(optimizer,step_size=10,gamma=args.gamma)

#class_weights =torch.tensor([0.3, 1],dtype=torch.float).cuda()
#criterion = monai.losses.DiceCELoss(softmax=True,to_onehot_y=True,include_background=False,reduction="mean",ce_weight=class_weights) #type: ignore
criterion = nn.CrossEntropyLoss(reduction='mean')
metrics = Evaluator()

##################################################################################################
#                                     TRAIN, VALID, TEST                                         #
##################################################################################################
def train(epoch)-> None:
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = data.float()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target).long().squeeze_(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output['out'], target) #TODO: change to output['out'] if it is deeplabv3_resnet50 or fcn_resnet50. Change to output if it is unet and target.unsqueeze(1) if it criterion monai
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), # type: ignore
                100. * batch_idx / len(train_loader), loss.item()))
        loss_list.append(loss.item())
    print("Mean Training Loss: ", np.mean(loss_list))
            
def test(epoch)-> float:
    DSC =[]
    pred_list = []
    model.eval()
    metrics.reset()
    test_loss = 0.
    x=0

    for data, target in test_loader:

        datis, targit = data, target
        
        data=data.float()
        if args.cuda:
            data, target = data.cuda(), target.cuda() 
        data, target = Variable(data), Variable(target).long().squeeze_(1)
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output['out'], target).item() #TODO: change to output['out'] if it is deeplabv3_resnet50 and target.unsqueeze(1) if it is criterion monai
        pred = output['out'].cpu() # TODO: change to output['out'] if it is deeplabv3_resnet50 or fcn_resnet50
        pred = F.softmax(pred, dim=1).numpy()
        target = target.cpu().numpy()
        pred_list.append(pred)
        
        pred = np.argmax(pred, axis=1)
        
        datito, predi = data, pred

        if epoch in (args.epochs,1,args.epochs//2):
            #gr.plot_image(datis,targit,datito,predi,epoch, args.model,x)
            x += 1
        lista = []
        for i in range(target.shape[0]):
            lista.append(scoring.dice_coef(target[i], pred[i]))
        DSC.extend(lista)
    
    if epoch == args.epochs:
        with open (f"models/{args.model}.pkl", "wb") as f:
            pickle.dump(pred_list, f)

    Dice = np.mean(DSC)
    #gr.save_value(round(Dice,4),f"data/dice{args.model}.txt")
    with open(f"data/dice{args.model}.txt", "a") as f:
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
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

##################################################################################################
#                                     MAIN                                                       #
##################################################################################################

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
                with open(args.save, 'wb') as fp:
                    state = model.state_dict()
                    torch.save(state, fp)
            else:
                scheduler.step(test_loss) 
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')