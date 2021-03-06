import os
import re
import pickle
import argparse
from collections import namedtuple

import numpy as np

import torch 
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import BertModel, BertConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import initiate_logger
from dataset import get_TSE_dataloader_kfold
from callback import PyLoggerCallback
from config import get_config

class BERT_TSE(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.transformer = config.model_cls(config.model_config).from_pretrained(config.pretrain_wt)

		if 'distilbert' in config.model_name.split('-')[0]:
			self.n_feature = self.transformer.transformer.layer[-1].ffn.lin2.out_features
		elif 'albert' in config.model_name.split('-')[0]:
			self.n_feature = self.transformer.encoder.albert_layer_groups[-1].albert_layers[-1].ffn_output.out_features
		elif 'bert' in config.model_name.split('-')[0]:
			self.n_feature = self.transformer.pooler.dense.out_features
		elif 'roberta' in config.model_name.split('-')[0]:
			self.n_feature = self.transformer.pooler.dense.out_features

		self.logits = nn.Sequential(
			nn.Linear(self.config.n_layer*self.n_feature, 128),
			nn.Tanh(),
			nn.Linear(128, 2)
		)

		self.dropout = nn.Dropout(p=0.5)

	def _set_feature_extract(self):
		for name, param in self.named_parameters():
			if re.search('logits', name):
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

	def forward(self, token_ids, token_type_ids, mask):
		if 'distil' in self.config.model_name.split('-')[0]:
			hidden_states = self.transformer(
				token_ids, 
				attention_mask=mask, 
				output_hidden_states=True
			)[-1]
		else:
			hidden_states = self.transformer(
				token_ids, 
				attention_mask=mask, 
				token_type_ids=token_type_ids, 
				output_hidden_states=True
			)[-1]
		features = torch.cat(hidden_states[:self.config.n_layer], dim=-1)

		if self.config.multi_sample_dropout and self.training:
			logits = torch.mean(torch.stack([self.logits(self.dropout(features)) for _ in range(5)], dim=0), dim=0)
		else:
			logits = self.logits(features)
		start_logits, end_logits = logits[:,:,0], logits[:,:,1]

		return start_logits, end_logits

	def _pred_token_to_char(self, text, offsets, token_pred):
		char_pred = np.zeros(len(text))
		for i, offset in enumerate(offsets):
			if offset[0] or offset[1]: char_pred[offset[0]:offset[1]]=token_pred[i]
		return char_pred

	def _pred_selected(self, text, offsets, start_token_pred, end_token_pred):
		start_pred = self._pred_token_to_char(text, offsets, start_token_pred)
		end_pred = self._pred_token_to_char(text, offsets, end_token_pred)
		start_idx = np.argmax(start_pred)
		end_idx = len(end_pred) - 1 - np.argmax(end_pred[::-1])
		return text[start_idx:end_idx+1]

	def _jaccard(self, str1, str2): 
		a = set(str1.lower().split()) 
		b = set(str2.lower().split())
		c = a.intersection(b)
		return float(len(c)) / (len(a) + len(b) - len(c))

	def _evaluate_pred_selected(self, pred, actual):
		return self._jaccard(actual, pred)

class PL_BERT_TSE(pl.LightningModule):
	def __init__(self, fold, dl_config, ml_config, batch_size=64, learning_rate=1e-4, *args, **kwargs):
		super().__init__()
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.save_hyperparameters()
		self.hparams.num_warmup_steps = int(22000/self.batch_size*0.3)
		self.hparams.num_training_steps = int(5*22000/self.batch_size)

		self.model = BERT_TSE(ml_config)
		self._reset_metric_state()

	def _reset_metric_state(self):
		self.stat = {
			'train': {'loss': [], 'start_loss': [], 'end_loss': []},
			'validation': {'loss': [], 'start_loss': [], 'end_loss': [], 'jaccard_score': []}
		}

	def forward(self, token_ids, token_type_ids, mask):
		start_logits, end_logits = self.model(token_ids, token_type_ids, mask)
		return start_logits, end_logits

	def training_step(self, batch, batch_idx):
		token_ids = batch['ids']
		token_type_ids = batch['type_ids']
		mask = batch['mask']
		start_label = batch['start_label']
		end_label = batch['end_label']

		start_logits, end_logits = self.model(token_ids, token_type_ids, mask)
		start_probs = torch.softmax(start_logits, dim=1)
		end_probs = torch.softmax(end_logits, dim=1)

		start_loss = F.kl_div(torch.log(start_probs), start_label, reduction='batchmean')
		end_loss = F.kl_div(torch.log(end_probs), end_label, reduction='batchmean')

		return {'loss': start_loss+end_loss, 'start_loss': start_loss, 'end_loss': end_loss}

	def training_epoch_end(self, outputs):
		loss_mean = torch.stack([x['callback_metrics']['loss'] for x in outputs]).mean()
		start_loss_mean = torch.stack([x['callback_metrics']['start_loss'] for x in outputs]).mean()
		end_loss_mean = torch.stack([x['callback_metrics']['end_loss'] for x in outputs]).mean()
		self.stat['train']['loss'].append(float(loss_mean.cpu().detach().numpy()))
		self.stat['train']['start_loss'].append(float(start_loss_mean.cpu().detach().numpy()))
		self.stat['train']['end_loss'].append(float(end_loss_mean.cpu().detach().numpy()))
		return {'loss': loss_mean}

	def validation_step(self, batch, batch_idx):
		token_ids = batch['ids']
		token_type_ids = batch['type_ids']
		mask = batch['mask']
		start_label = batch['start_label']
		end_label = batch['end_label']

		texts = batch['text']
		offsets = batch['offsets']
		selected_texts = batch['selected_text']

		start_logits, end_logits = self.model(token_ids, token_type_ids, mask)
		start_probs = torch.softmax(start_logits, dim=1)
		end_probs = torch.softmax(end_logits, dim=1)

		start_loss = F.kl_div(torch.log(start_probs), start_label, reduction='batchmean')
		end_loss = F.kl_div(torch.log(end_probs), end_label, reduction='batchmean')

		start_probs = start_probs.cpu().detach().numpy()
		end_probs = end_probs.cpu().detach().numpy()

		selected_pred = [self.model._pred_selected(text, offset, start_token_pred, end_token_pred) 
			for text, offset, start_token_pred, end_token_pred in zip(texts, offsets, start_probs, end_probs)]

		jaccard_score = np.mean([self.model._evaluate_pred_selected(p,t) for p, t in zip(selected_pred, selected_texts)])

		return {'val_loss': start_loss+end_loss, 'val_start_loss': start_loss, 'val_end_loss': end_loss, 'val_jaccard_score': jaccard_score}

	def validation_epoch_end(self, outputs):
		loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
		start_loss_mean = torch.stack([x['val_start_loss'] for x in outputs]).mean()
		end_loss_mean = torch.stack([x['val_end_loss'] for x in outputs]).mean()
		jaccard_score_mean = np.mean([x['val_jaccard_score'] for x in outputs])
		self.stat['validation']['loss'].append(float(loss_mean.cpu().detach().numpy()))
		self.stat['validation']['start_loss'].append(float(start_loss_mean.cpu().detach().numpy()))
		self.stat['validation']['end_loss'].append(float(end_loss_mean.cpu().detach().numpy()))
		self.stat['validation']['jaccard_score'].append(jaccard_score_mean)
		return {'val_loss': loss_mean, 'val_jaccard_score': jaccard_score_mean}

	def train_dataloader(self):
		dataloader = get_TSE_dataloader_kfold('train', self.hparams.fold, self.hparams.batch_size, self.hparams.dl_config)
		return dataloader

	def val_dataloader(self):
		dataloader = get_TSE_dataloader_kfold('validation', self.hparams.fold, self.hparams.batch_size, self.hparams.dl_config)
		return dataloader

	def configure_optimizers(self):
		params_to_update = self.model._check_parameter_requires_grad()
		opt = transformers.AdamW(params_to_update, lr=self.hparams.learning_rate, weight_decay=1e-2)
		sch = {
			'scheduler': transformers.get_linear_schedule_with_warmup(
				opt, 
				num_warmup_steps=self.hparams.num_warmup_steps, 
				num_training_steps=self.hparams.num_training_steps
			),
			'interval': 'step',
			'frequency': 1,
		}
		return [opt], [sch]
