import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from torchvision import transforms
from PIL import Image
from monai.transforms import ScaleIntensityRangePercentiles

# Dataset class
class BRATSDataset(Dataset):
    def __init__(self, data_path, transform=None, dataset_type='train', limit_samples=None):
        super(BRATSDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.image_files = [[] for _ in range(len(data_path))]
        self.label_files = []
        self.limit_samples = limit_samples

        if dataset_type == 'train':
            image_dir = 'Train/Data'
            label_dir = 'Train/Mask'

        elif dataset_type == 'test':
            image_dir = 'Test/Data'
            label_dir = 'Test/Mask'
        else:
            raise ValueError(f'Invalid dataset type: {dataset_type}')

        for i, modality_path in enumerate(data_path):
            print(modality_path)
            for root, dirs, files in os.walk(os.path.join(modality_path, image_dir)):
                for file in sorted(files):
                    if file.endswith('.npy'):
                        self.image_files[i].append(os.path.join(root, file))

        for root, dirs, files in os.walk(os.path.join(data_path[0], label_dir)):
            for file in sorted(files):
                if file.endswith('.npy'):
                    self.label_files.append(os.path.join(root, file))
        
        assert(len(self.image_files[0]) == len(self.label_files))

        print(f"The total number of images in {dataset_type}: {len(self.label_files)}")

    def __len__(self):
        'Denotes the total number of samples'

        if self.limit_samples is not None:
            return self.limit_samples
        return len(self.label_files)

    def __getitem__(self, idx):

        'Generates one sample of data'
        image_paths_ch1 = os.path.join(self.image_files[0][idx])
        image_paths_ch2 = os.path.join(self.image_files[1][idx])
        image_paths_ch3 = os.path.join(self.image_files[2][idx])
        image_paths_ch4 = os.path.join(self.image_files[3][idx])
        label_path = os.path.join(self.label_files[idx])

        image_ch1 = np.load(image_paths_ch1)
        image_ch2 = np.load(image_paths_ch2)
        image_ch3 = np.load(image_paths_ch3)
        image_ch4 = np.load(image_paths_ch4)
        
        label = np.load(label_path)
    
        assert not np.any(np.isnan(label))
        
        image_ch1 = resize(image_ch1, (256, 256), mode='reflect')
        image_ch2 = resize(image_ch2, (256, 256), mode='reflect')
        image_ch3 = resize(image_ch3, (256, 256), mode='reflect')
        image_ch4 = resize(image_ch4, (256, 256), mode='reflect')
        label = resize(label, (256, 256), mode='reflect')
        
        m_ch1, s_ch1 = np.mean(image_ch1, axis=(0, 1)), np.std(image_ch1, axis=(0, 1))
        m_ch2, s_ch2 = np.mean(image_ch2, axis=(0, 1)), np.std(image_ch2, axis=(0, 1))
        m_ch3, s_ch3 = np.mean(image_ch3, axis=(0, 1)), np.std(image_ch3, axis=(0, 1))
        m_ch4, s_ch4 = np.mean(image_ch4, axis=(0, 1)), np.std(image_ch4, axis=(0, 1))

        if self.transform:
        
            image_ch1 = self.transform(image_ch1)
            image_ch2 = self.transform(image_ch2)
            image_ch3 = self.transform(image_ch3)
            image_ch4 = self.transform(image_ch4)
            label = self.transform(label)
            '''
            if (s_ch1==0).any():
                image_ch1 = transforms.Normalize(mean=m_ch1, std=s_ch1 + 1e-6)(image_ch1)
                image_ch2 = transforms.Normalize(mean=m_ch2, std=s_ch2 + 1e-6)(image_ch2)
                image_ch3 = transforms.Normalize(mean=m_ch3, std=s_ch3 + 1e-6)(image_ch3)
                image_ch4 = transforms.Normalize(mean=m_ch4, std=s_ch4 + 1e-6)(image_ch4)
            else:
                image_ch1 = transforms.Normalize(mean=m_ch1, std=s_ch1)(image_ch1)
                image_ch2 = transforms.Normalize(mean=m_ch2, std=s_ch2)(image_ch2)
                image_ch3 = transforms.Normalize(mean=m_ch3, std=s_ch3)(image_ch3)
                image_ch4 = transforms.Normalize(mean=m_ch4, std=s_ch4)(image_ch4)
            '''
            percentilescaler = ScaleIntensityRangePercentiles(5,95, b_min=0.0, b_max=1.0)
            image_ch1 = percentilescaler(image_ch1)
            image_ch2 = percentilescaler(image_ch2)
            image_ch3 = percentilescaler(image_ch3)
            image_ch4 = percentilescaler(image_ch4)
    
        image = torch.cat((image_ch1, image_ch2, image_ch3, image_ch4), dim=0)

        return image, label

    
    def get_path(self, index: int):
        return self.image_files[index], self.label_files[index]
