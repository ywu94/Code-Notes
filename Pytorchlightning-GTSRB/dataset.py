import os
import time
import pickle
import numbers
import pandas as pd
import numpy as np

import cv2
from skimage import io, transform

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torchvision.transforms.functional as VF

from utils import initiate_logger

class Rescale(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			new_h, new_w = self.output_size, self.output_size
		else:
			new_h, new_w = self.output_size

		img = transform.resize(image, (new_h, new_w))

		return {'image': img, 'label': label}

class ToTensor(object):
	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))               
		return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}

class Normalize(object):
	def __init__(self, mean, std, inplace=False):
		self.mean = mean
		self.std = std
		self.inplace = inplace

	def __call__(self, sample):
		image, label = sample['image'], sample['label']
		img = VF.normalize(image, self.mean, self.std, self.inplace)
		return {'image': img, 'label': label}

class GTSRB_Dataset(torch.utils.data.Dataset):
	def __init__(self, idx_list, transform=None):
		self.idx_list = idx_list
		self.transform = transform

	def __len__(self):
		return len(self.idx_list)

	def __getitem__(self, idx):
		start = time.time()
		flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
		image = cv2.cvtColor(cv2.imread(self.idx_list[idx][0], flags), cv2.COLOR_BGR2RGB)
		image = np.concatenate([np.expand_dims(cv2.equalizeHist(image[:,:,i]), axis=2) for i in range(3)], axis=2)
		image = image.astype(np.float32)/255
		label = np.array(self.idx_list[idx][1])
		sample = {'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample

def get_GTSRB_dataloader(target, batch_size=256, num_workers=8, pin_memory=True):
	cwd = os.getcwd()
	if target=='train':
		path = os.path.join(cwd, 'train_idx.pkl')
		shuffle = True
	elif target=='test':
		path = os.path.join(cwd, 'test_idx.pkl')
		shuffle = False
	elif target=='validation':
		path = os.path.join(cwd, 'validation_idx.pkl')
		shuffle = False
	else:
		raise Exception('unknown target file argument {}'.format(target))
	with open(path, 'rb') as f:
		idx = pickle.load(f)
	dataset = GTSRB_Dataset(idx, transform=transforms.Compose([
		Rescale(96), 
		ToTensor(),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	]))
	dataloader = torch.utils.data.DataLoader(dataset, 
		batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
	return dataloader

if __name__=='__main__':
	logger = initiate_logger('dataset.log')
	dataloader = get_GTSRB_dataloader('validation')
	for idx, batch in enumerate(dataloader):
		logger.info('Batch {} ready'.format(idx))