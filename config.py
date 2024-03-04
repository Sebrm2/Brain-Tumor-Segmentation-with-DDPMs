"""Config used for training and testing the model."""

import argparse

def model_config():

    parser = argparse.ArgumentParser(description='PyTorch Brain Tumor Segmentation')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: idk)')
    parser.add_argument('--epochs', type=int, default=6,
                        help='number of epochs to train (default: idk)')
    parser.add_argument('--lr', type=float, default= 0.001,
                        help='learning rate (default: idk)')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor (default: 0.5)')
    parser.add_argument('--step-size', type=int, default=3,
                        help='step size for scheduler (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=22,
                        help='random seed (default: 22)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before '
                            'logging training status')
    parser.add_argument('--save', type=str, default='modelDL101monai_32BS.pt', #TODO: Change according to the model (.pt name changes)
                        help='file on which to save model weights')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
    parser.add_argument('--model', type=str, default='DLR101_32BATCH_monai', #TODO: Change according to the model (.txt name changes)
                        help='name of the model to use (default: idk)')
    args = parser.parse_args()

    return args