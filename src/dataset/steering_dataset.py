import torch
import os
import glob
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np


class SteeringDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, random_hflip=False):
        super(SteeringDataset, self).__init__()
        self.directory = directory
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
        self.count = 0
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        steering = ann['steering']
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            steering = - steering
            
        return image, torch.Tensor([steering])
    
    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        steering = items[0]
        return steering
        
    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            steering = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'steering': steering
            }]
        
    def save_entry(self, image, steering):
        if not os.path.exists(self.direcory):
            subprocess.call(['mkdir', '-p', self.directory])
        filename = '%0.3f_%s.jpg' % (round(steering, 3), str(uuid.uuid1()))
        
        image_path = os.path.join(category_dir, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
        
    def get_count(self):
        return self.count

    
if __name__ == "__main__":
    d = SteeringDataset('.', ["nic"])
    d.save_entry('nic', None , -1)