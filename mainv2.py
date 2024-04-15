import time
import copy
import torch
import monai
import wandb
import pickle
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import os.path as osp
import torch.optim as optim
import utils.Graphics as gr
import matplotlib.pyplot as plt
import utils.scoring as scoring
import torch.nn.functional as F
import torch.optim.lr_scheduler as sch
import torch.utils.model_zoo as model_zoo
from config import model_config
from torch.autograd import Variable
from Dataloader import BRATSDataset
from utils.Evaluator import Evaluator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# Load the configuration
args = model_config()

# Check cuda availability
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set random seed for GPU or CPU
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

# Set arguments for Dataloaders
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Set main directory of data
directory = "/home/srodriguez47/ddpm/BraTS2023_StructuredDataV3.0-t1c/BraTS2023_AxialSlices" # TODO: change this to the correct path

# Initialize WandB
wandb.login()
wandb.init(project="brainDDPM", name="seb_3.0_t1c_32batch_DL", config=args)
args = wandb.config
# Load the data
print("\033[1;35;40m Loading the folders...\033[0m")

train_loader = DataLoader(BRATSDataset(data_path = directory, dataset_type='train',
                transform=transforms.Compose([transforms.ToTensor()])), args.batch_size,
                shuffle=True, **kwargs)

test_loader = DataLoader(BRATSDataset(data_path = directory, dataset_type='test',
                transform=transforms.Compose([transforms.ToTensor()])), args.batch_size,
                shuffle=False, **kwargs)

# Load the model
print("\033[1;35;40m Loading the model...\033[0m")

model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
#model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
model.classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
model.aux_classifier[4] = nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
load_model = False
#breakpoint()
# Set number of GPUs to use
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model)
        model.cuda()


"""
# Load the model if it exists
if osp.exists(args.save):
    with open(args.save, 'rb') as fp:
        state = torch.load(fp)
        model.load_state_dict(state)
        load_model = True
"""

# Set the optimizer, scheduler and loss function

optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay) 
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = sch.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
class_weights = torch.tensor([0.5,1,1,1],dtype=torch.float).cuda() #idk if this is the correct way to do it
criterion = monai.losses.DiceLoss(softmax=True,to_onehot_y=True,include_background=False,reduction="mean")
#criterion = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)
# Initialize the evaluator
metrics = Evaluator()

# Train the model
def train(epoch)-> None:
    
    wandb.watch(model, criterion, log="all", log_freq=1)
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = data.float()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data), Variable(target).long().squeeze_(1) #for crossentropyloss
        data, target = Variable(data), Variable(target).long() #for diceloss
        
        optimizer.zero_grad()
        output = model(data)
   
        #loss = criterion(output['out'], target) # for crossentropy
        loss = criterion(output['out'], target) # for diceloss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), # type: ignore
                100. * batch_idx / len(train_loader), loss.item()))
        loss_list.append(loss.item())
    print("Mean Training Loss: ", np.mean(loss_list))

    wandb.log({"train_loss": np.mean(loss_list), "epoch": epoch})
            
# Test the model
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
        #data, target = Variable(data), Variable(target).long().squeeze_(1) #for crossentropyloss
        data, target = Variable(data), Variable(target).long() #for diceloss
        
        with torch.no_grad():
            output = model(data)

        test_loss += criterion(output['out'], target).item() # for diceloss or crossentropy
        pred = output['out'].cpu() 
        pred = F.softmax(pred, dim=1).numpy()
        target = target.cpu().numpy()
        pred_list.append(pred)
        pred = np.argmax(pred, axis=1)

        if epoch in (args.epochs, 1, args.epochs // 2):
 
            for i in range(target.shape[0]):
                pred_tensor = torch.tensor(pred[i], dtype=torch.float64)
                wandb.log({"input_image": wandb.Image(datis[i]),
                           "ground_truth": wandb.Image(targit[i]),
                           "predicted_segmentation": wandb.Image(pred_tensor.unsqueeze(0))})

        lista = []
        for i in range(target.shape[0]):
            lista.append(scoring.multiclass_dice_score(target[i], pred[i], 4))
        DSC.extend(lista)
    
    if epoch == args.epochs:
        with open (f"{args.model}.pkl", "wb") as f:
            pickle.dump(pred_list, f)

    Dice = np.nanmean(DSC)
    wandb.log({"dice_score": Dice, "epoch": epoch})
    wandb.log({"test_loss": test_loss, "epoch": epoch})

    with open(f"dice{args.model}.txt", "a") as f:
        f.write(f"Test\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Dice: {Dice}\n")
    print('Test:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, len(test_loader.dataset))) # type: ignore
    print("Dice:{}".format(Dice))

    return test_loss


# Run the model
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
            
            scheduler.step() 

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')