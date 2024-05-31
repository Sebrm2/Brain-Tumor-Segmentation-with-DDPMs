from monai.data import DataLoader
from monai.config import print_config
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
import pandas as pd
from torch.utils.data import Dataset
print_config()
import nibabel as nib
import numpy as np
import torch

class BRATSDataset(Dataset):
    def __init__(self, csv_file, modality, image_cols=["t1c", "t1n", "t2f", "t2w"], label_col="seg", transform=None):
        super(BRATSDataset, self).__init__()
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['modality'] == modality]  # Filter by modality
        self.image_cols = image_cols
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = {}

        # Get paths for each image channel
        for col in self.image_cols:
            item[col] = self.get_paths(self.df, col)[index]

        # Load label path if available
        if self.label_col is not None:
            item[self.label_col] = self.get_paths(self.df, self.label_col)[index]
        
        # Apply transformations
        if self.transform:
            item = self.transform(item)

        # Stack images and return
        image = torch.cat([item[col] for col in self.image_cols], dim=0)

        if self.label_col is not None:
            return {"image": image, "label": item[self.label_col]}
        else:
            return {"image": image}
    
    def get_paths(self, df, channel):
        return df[channel].tolist()