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
#from Dataloader import BRATSDataset
from Dataloader_3D import BRATSDataset
from utils.Evaluator import Evaluator
from monai.networks.nets import UNETR, UNet
import pandas as pd
from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose, ResizeD, Orientationd, NormalizeIntensityd, ClipIntensityPercentilesd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

load_model = False
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "dataset.csv"

# Initialize WandB
wandb.login()
wandb.init(project="brainDDPM", name="lina_3D_4channels_UNETR_DL_Percentilenorm", config=args)
args = wandb.config
# Load the data
print("\033[1;35;40m Loading the folders...\033[0m")

transform = Compose([
    LoadImaged(keys=["t1c", "t1n", "t2f", "t2w", "seg"], image_only=False),  # Include 'seg' for the label
    EnsureChannelFirstd(keys=["t1c", "t1n", "t2f", "t2w", "seg"]),
    ResizeD(keys=["t1c", "t1n", "t2f", "t2w", "seg"], spatial_size=(128, 128, 128)),
    #NormalizeIntensityd(keys=["t1c", "t1n", "t2f", "t2w"]),
    ClipIntensityPercentilesd(keys=["t1c", "t1n", "t2f", "t2w"], lower=5, upper=95),
    #Orientationd(keys=["t1c", "t1n", "t2f", "t2w", "seg"], axcodes="PLI"),
])

train_loader = DataLoader(BRATSDataset("dataset.csv", modality="train", transform=transform), args.batch_size,
                shuffle=True, **kwargs)
test_loader = DataLoader(BRATSDataset("dataset.csv", modality="test", transform=transform), args.batch_size,
                shuffle=False, **kwargs)

# Load the model
print("\033[1;35;40m Loading the model...\033[0m")

model = UNETR(
    in_channels=4,        
    out_channels=4,
    img_size=(128, 128, 128))


'''
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=0)
'''
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() >= 1:
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
#class_weights = torch.tensor([1,1,1],dtype=torch.float).cuda() #idk if this is the correct way to do it
criterion = monai.losses.DiceLoss(softmax=True,to_onehot_y=True,include_background=False,reduction="mean")
#criterion = monai.losses.GeneralizedDiceLoss(softmax=True,to_onehot_y=True, include_background=False, reduction="mean")
#criterion = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)
# Initialize the evaluator
metrics = Evaluator()

# Train the model
def train(epoch)-> None:
    
    wandb.watch(model, criterion, log="all", log_freq=1)
    model.train()
    loss_list = []
    for batch_idx, diccio in enumerate(train_loader):
        
        data = diccio['image'].float()
        data, target = data.to(device), diccio["label"].to(device)
        data, target = Variable(data), Variable(target).long()
        optimizer.zero_grad()
        output = model(data)
        
        #print("output shape", output.shape)
        loss = criterion(output, target) # for unet
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), # type: ignore
                100. * batch_idx / len(train_loader), loss.item()))
        loss_list.append(loss.item())
    print("Mean Training Loss: ", np.mean(loss_list))

    wandb.log({"train_loss": np.mean(loss_list), "epoch": epoch})


class_t = {
    0: "background",
    1: "NCR",
    2: "ET",
    3: "ED"
}

# Test the model
def test(epoch, best_dice)-> float:
    dice_class = {1: [], 2: [], 3: []} 
    DSC =[]
    pred_list = []
    model.eval()
    metrics.reset()
    test_loss = 0.
    x=0

    for diccio in test_loader:
        
        data=diccio["image"].float()
        data, target = data.to(device), diccio["label"].to(device)
        #data, target = Variable(data), Variable(target).long().squeeze_(1) #for crossentropyloss
        data, target = Variable(data), Variable(target).long() #for diceloss
    
        with torch.no_grad():
            output = model(data)
            #print("output shape", output['out'].shape)

        #test_loss += criterion(output['out'], target).item() # for diceloss or crossentropy
        test_loss += criterion(output, target).item() # for unet

        pred = output.cpu()
        #print("pred shape1", pred.shape) 
        pred = F.softmax(pred, dim=1).numpy()
        #print("pred softmax shape", pred.shape)
        target = target.cpu().numpy()
        pred_list.append(pred)
        pred = np.argmax(pred, axis=1)
        
        if epoch in (args.epochs, 1, args.epochs // 2):

            slices = np.linspace(40, 85, 10)

            for i in range(target.shape[0]):
                for slice in slices:
                    
                    #print("a",pred[i].shape)
                    pred_tensor = torch.tensor(pred[i], dtype=torch.float64)
                    #print("b",pred_tensor.shape)
                    slice = int(slice)
                    wandb.log(
                    {"Prediction" : wandb.Image(data[i][0][:,:,slice], masks={
                        "predictions" : {
                            "mask_data" : np.array(pred_tensor[:,:,slice]),
                            "class_labels" : class_t
                    }
                        
                    })
                    })

                    wandb.log(
                    {"Ground Truth" : wandb.Image(data[i][0][:,:,slice], masks={
                        
                        "ground_truth" : {
                            "mask_data" : np.array((target[i].squeeze(0))[:,:,slice]),
                        "class_labels" : class_t
                    }
                    })})

        lista = []
        lista_1, lista_2, lista_3 = [], [], []
        for i in range(target.shape[0]):
            diccio_l = scoring.multiclass_dice_score(target[i], pred[i], 4)
            lista.append(diccio_l[0])
            if 1 in diccio_l.keys():
                lista_1.append(diccio_l[1])
            if 2 in diccio_l.keys():
                lista_2.append(diccio_l[2])
            if 3 in diccio_l.keys():
                lista_3.append(diccio_l[3])
        DSC.extend(lista)
        dice_class[1].extend(lista_1)
        dice_class[2].extend(lista_2)
        dice_class[3].extend(lista_3)
    
   

    Dice = np.nanmean(DSC)
    channel_1 = np.nanmean(dice_class[1])
    channel_2 = np.nanmean(dice_class[2])
    channel_3 = np.nanmean(dice_class[3])
    wandb.log({"dice_score": Dice, "epoch": epoch})
    wandb.log({"test_loss": test_loss, "epoch": epoch})
    wandb.log({f"dice_score_{class_t[1]}": channel_1, "epoch": epoch})
    wandb.log({f"dice_score_{class_t[2]}": channel_2, "epoch": epoch})
    wandb.log({f"dice_score_{class_t[3]}": channel_3, "epoch": epoch})

    if Dice > best_dice:
        best_dice = Dice
        torch.save(model.state_dict(), f"{args.model}_best.pth")

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
    best_dice = 0
    if load_model:
        best_loss = test(0)
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(epoch)
            test_loss = test(epoch, best_dice)
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