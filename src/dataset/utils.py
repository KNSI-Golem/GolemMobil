import torch
import torchvision.transforms as transforms
import PIL.Image

# import torch.nn.functional as F
# import cv2
# import numpy as np

from torch.utils.data import DataLoader, Subset


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def create_dataloaders(
    dataset,
    test_split: float = 0.2,
    batch_size: int = 1,
    shuffle: bool = False,
):
    # If shuffle set to True, dataloader shuffles train dataset on each epoch

    dataset_size = len(dataset)
    train_size = int(dataset_size * (1 - test_split))

    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
