import os
import gc
import pickle
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import initiate_logger
from model import PL_BERT_TSE
from callback import PyLoggerCallback
from config import get_config

def find_config(args):
	logger = initiate_logger(args.logpath)
	logger.info(args)

	dl_config, ml_config, batch_size = get_config(args.model) 
	model = PL_BERT_TSE(
		0, 
		dl_config, 
		ml_config, 
		batch_size=batch_size if batch_size is not None else args.batchsize,
		learning_rate=args.lr)
	model.model._set_fine_tune()

	trainer = pl.Trainer(
		gpus=args.gpus,
		progress_bar_refresh_rate=10,
		weights_summary=None,
		gradient_clip_val=100,
		num_sanity_val_steps=0
	)

	# Some problems with the batch size finder	

	# suggested_batch_size = trainer.scale_batch_size(model)
	# logger.info('Max Batch Size: {}'.format(suggested_batch_size))
	# model.hparams.batch_size = suggested_batch_size

	lr_finder = trainer.lr_find(
		model, 
		min_lr=1e-08, 
		max_lr=1,
		num_training=100,
		mode='exponential'
	)

	res = lr_finder.results
	with open('lr_find.pkl', 'wb') as f:
		pickle.dump(res, f)

	suggested_lr = lr_finder.suggestion()
	logger.info('Suggested Learning Rate: {:.8f}'.format(suggested_lr))

def finetune_train_5fold(args):
	logger = initiate_logger(args.logpath)
	logger.info(args)
	if not gc.isenabled(): gc.enable()

	for fold in range(5):
		logger.info('Training fold {}/5'.format(fold+1))

		dl_config, ml_config, batch_size = get_config(args.model) 
		model = PL_BERT_TSE(
			fold, 
			dl_config, 
			ml_config, 
			batch_size=batch_size if batch_size is not None else args.batchsize,
			learning_rate=args.lr)
		model.model._set_fine_tune()

		es = EarlyStopping(monitor='val_loss', patience=0, mode='min')
		ck = ModelCheckpoint(
			filepath=os.path.join(args.ckpath, '{}-fold-{}-'.format(args.model, fold+1)+'{epoch}-{val_loss:.4f}'),
			save_top_k=1,
			mode='min',
			save_weights_only=False,
			period=0
		)
		trainer_ft = pl.Trainer(
			max_epochs=args.epoch,
			gpus=args.gpus,
			progress_bar_refresh_rate=10,
			weights_summary=None,
			gradient_clip_val=100,
			num_sanity_val_steps=0,
			callbacks=[PyLoggerCallback(logger)],
			early_stop_callback=es,
			checkpoint_callback=ck
		)

		trainer_ft.fit(model)

		torch.cuda.empty_cache()
		_ = gc.collect()


if __name__=='__main__':
	cwd = os.getcwd()

	parser = ArgumentParser()
	parser.add_argument('model', type=str)

	group = parser.add_mutually_exclusive_group()
	group.add_argument('--findconfig', action='store_true', default=False)
	group.add_argument('--finetune', action='store_true', default=False)

	parser.add_argument('--gpus', type=int, default=1)
	parser.add_argument('--lr', type=float, default=7e-5)
	parser.add_argument('--epoch', type=int, default=2)
	parser.add_argument('--batchsize', type=int, default=64)

	parser.add_argument('--logpath', type=str, default='')
	parser.add_argument('--ckpath', type=str, default='')
	parser.add_argument('--ckfile', type=str, default='')

	args = parser.parse_args()

	if not args.logpath: args.logpath = os.path.join(cwd, '{}.log'.format(args.model))
	if not args.ckpath: args.ckpath = os.path.join(cwd, '{}-checkpoint-artifact'.format(args.model))

	if args.findconfig:
		find_config(args)
	elif args.finetune:
		finetune_train_5fold(args)