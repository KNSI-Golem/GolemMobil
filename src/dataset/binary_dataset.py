import torch
import os
import glob
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, random_hflip=False):
        super(BinaryDataset, self).__init__()
        self.directory = directory
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        ann = self.annotations[index]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
        
        return image, int(ann['is_free'])
    
    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        return items[0]
        
        
    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            is_free = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'is_free': is_free
            }]
        
    def save_entry(self, image, is_free):
        if not os.path.exists(self.directory):
            subprocess.call(['mkdir', '-p', self.directory])
            
        filename = '%d_%s.jpg' % (is_free, str(uuid.uuid1()))
        
        image_path = os.path.join(self.directory, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
        
    def get_count(self):
        return len(self.annotations)