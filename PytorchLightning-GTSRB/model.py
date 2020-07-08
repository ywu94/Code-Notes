import os
import re
import pickle

import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl

from utils import initiate_logger
from dataset import get_GTSRB_dataloader

class ResNet_Classifier(nn.Module):
	def __init__(self, out_size, *args, **kwargs):
		super(ResNet_Classifier, self).__init__()
		self.out_size = out_size
		self.model = torchvision.models.resnet34(pretrained=True)
		self.model.fc = nn.Linear(self.model.fc.in_features, out_size)

	def _set_feature_extract(self):
		for name, param in self.named_parameters():
			if re.search('model.fc', name):
				param.requires_grad = True
			else:
				param.requires_grad = False

	def _set_fine_tune(self):
		for _, param in self.named_parameters():
			param.requires_grad = True

	def _check_parameter_requires_grad(self):
		params_to_update = []
		for name, param in self.named_parameters():
			if param.requires_grad == True:
				params_to_update.append(param)
		return params_to_update

	def forward(self, inp):
		return self.model(inp)

class PL_ResNet_Classifier(pl.LightningModule):
	def __init__(self, out_size, learning_rate, *args, **kwargs):
		super(PL_ResNet_Classifier, self).__init__()
		self.save_hyperparameters()

		self.model = ResNet_Classifier(self.hparams.out_size)
		self._reset_metric_state()

	def _reset_metric_state(self):
		self.stat = {
			'train': {'loss': []},
			'validation': {'loss': [], 'accuracy': []},
			'test': {'loss': [], 'accuracy': []}
		}

	def forward(self, inp):
		return self.model(inp)

	def training_step(self, batch, batch_idx):
		x, y = batch['image'], batch['label']
		yp = F.softmax(self.model(x), dim=1)
		loss = F.cross_entropy(yp,y)
		return {'loss': loss}

	def training_epoch_end(self, outputs):
		loss_mean = torch.stack([x['batch_loss'] for x in outputs]).mean()
		self.stat['train']['loss'].append(float(loss_mean.cpu().detach().numpy()))
		return {'loss': loss_mean}

	def validation_step(self, batch, batch_idx):
		x, y = batch['image'], batch['label']
		yp = F.softmax(self.model(x), dim=1)
		loss = F.cross_entropy(yp,y)
		return {'val_loss': loss, 'pred': yp, 'true': y}

	def validation_epoch_end(self, outputs):
		val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
		true = torch.cat([x['true'] for x in outputs], dim=0).cpu().detach().numpy()
		pred = torch.cat([x['pred'] for x in outputs], dim=0).cpu().detach().numpy()
		pred = np.argmax(pred, -1).reshape((-1,))
		true = true.reshape((-1,))
		acc_score = accuracy_score(true, pred)
		self.stat['validation']['loss'].append(float(val_loss_mean.cpu().detach().numpy()))
		self.stat['validation']['accuracy'].append(acc_score)
		return {'val_loss': val_loss_mean, 'val_accuracy': acc_score}

	def test_step(self, batch, batch_idx):
		x, y = batch['image'], batch['label']
		yp = F.softmax(self.model(x), dim=1)
		loss = F.cross_entropy(yp,y)
		return {'test_loss': loss, 'pred': yp, 'true': y}

	def test_epoch_end(self, outputs):
		test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
		true = torch.cat([x['true'] for x in outputs], dim=0).cpu().detach().numpy()
		pred = torch.cat([x['pred'] for x in outputs], dim=0).cpu().detach().numpy()
		pred = np.argmax(pred, -1).reshape((-1,))
		true = true.reshape((-1,))
		acc_score = accuracy_score(true, pred)
		self.stat['test']['loss'].append(float(test_loss_mean.cpu().detach().numpy()))
		self.stat['test']['accuracy'].append(acc_score)
		return {'test_loss': test_loss_mean, 'test_accuracy': acc_score}

	def train_dataloader(self):
		return get_GTSRB_dataloader('train')

	def val_dataloader(self):
		return get_GTSRB_dataloader('validation')

	def test_dataloader(self):
		return get_GTSRB_dataloader('test')

	def configure_optimizers(self):
		params_to_update = self.model._check_parameter_requires_grad()
		return torch.optim.Adam(params_to_update, lr=self.hparams.learning_rate)

	# def configure_optimizers(self):
	# 	params_to_update = self.model._check_parameter_requires_grad()
	# 	opt = torch.optim.Adam(params_to_update, lr=self.hparams.learning_rate)
	# 	sch = {
	# 		'scheduler': optim.lr_scheduler.CyclicLR(
	# 			opt, 
	# 			base_lr=self.hparams.learning_rate, 
	# 			max_lr=2*self.hparams.learning_rate, 
	# 			step_size_up=100
	# 		),
	# 		'interval': 'step',
	# 		'frequency': 1,
	# 	}
	# 	return [opt], [sch]