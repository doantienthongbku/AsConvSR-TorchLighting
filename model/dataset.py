from pathlib import Path
import os
import glob
import cv2
import functools
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF


class SRDataset(Dataset):
    def __init__(self, images_dir: str = "./datasets", crop_size: int = 96, upscale_factor: int = 2, 
                 mode: str = "train", image_format: str = "png", preupsample: bool = False):
        super(SRDataset, self).__init__()
        self.image_path_list = glob.glob(images_dir + "/*." + image_format)
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.mode = mode
        self.preupsample = preupsample
        
        if self.mode == "train":
            self.transforms = transforms.Compose([
                transforms.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='reflect'),
                transforms.RandomApply([
                    functools.partial(TF.rotate, angle=0),
                    functools.partial(TF.rotate, angle=90),
                    functools.partial(TF.rotate, angle=180),
                    functools.partial(TF.rotate, angle=270),
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        elif self.mode == "valid":
            self.transforms = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
            ])
        else:
            raise ValueError("The mode must be either 'train' or 'valid'.")

    def __len__(self) -> int:
        return len(self.image_path_list)
    
    def __getitem__(self, index: int):
        image = Image.open(self.image_path_list[index]).convert('RGB')
        image_hr = self.transforms(image)
        down_size = [l // self.upscale_factor for l in image_hr.size[::-1]]
        image_lr = TF.resize(image_hr, down_size, interpolation=Image.BICUBIC)
        
        if self.preupsample:
            image_lr = TF.resize(image_lr, image_hr.size[::-1], interpolation=Image.BICUBIC)
            
        return {'lr': TF.to_tensor(image_lr), 'hr': TF.to_tensor(image_hr)}
        