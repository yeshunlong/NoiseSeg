import os
import h5py
import itertools
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class DataSet(Dataset):
    def __init__(self, transform, split='train'):
        self.image_list = []
        self.transform = transform
        file_list_path = os.path.join('dataset', split + '.list')
        with open(file_list_path, 'r') as f:
            for line in f:
                self.image_list.append(os.path.join('dataset', os.path.join(line.strip(), 'mri_norm2.h5')))
                
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        file_path = self.image_list[index]
        content = h5py.File(file_path, 'r')
        image = content['image'][:]
        label = content['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        w, h, d = image.shape
        if w <= self.output_size[0] or h <= self.output_size[1] or d <= self.output_size[2]:
            pw = max((self.output_size[0] - w) // 2 + 3, 0)
            ph = max((self.output_size[1] - h) // 2 + 3, 0)
            pd = max((self.output_size[2] - d) // 2 + 3, 0)
            image = np.pad(image, ((pw, pw), (ph, ph), (pd, pd)), mode='constant', constant_values=0)
            label = np.pad(label, ((pw, pw), (ph, ph), (pd, pd)), mode='constant', constant_values=0)
        w, h, d = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}

class RandomRotFlip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis).copy()
        label = np.flip(label, axis).copy()
        return {'image': image, 'label': label}

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        image = torch.from_numpy(image)
        label = torch.from_numpy(sample['label']).long()
        return {'image': image, 'label': label}
    
class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for(primary_batch, secondary_batch) in zip(grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
