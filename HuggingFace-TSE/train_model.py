import os
import pickle
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import initiate_logger
from model import PL_BERT_TSE
from callback import PyLoggerCallback

def find_lr(hparams):
	logger = initiate_logger(hparams.logpath)
	logger.info('Task: Find learning rate for fine tune task')
	logger.info(hparams)

	net = PL_BERT_TSE()
	trainer = pl.Trainer(
		gpus=hparams.gpus,
		progress_bar_refresh_rate=10,
		weights_summary=None,
		gradient_clip_val=100,
		num_sanity_val_steps=0
	)
	net.model._set_fine_tune()
	lr_finder = trainer.lr_find(net, 
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

def finetune_train(hparams):
	logger = initiate_logger(hparams.logpath)
	logger.info('Task: Fine tune the entire model')
	logger.info(hparams)

	net = PL_BERT_TSE(learning_rate=hparams.lr)
	es = EarlyStopping(monitor='val_jaccard_score', patience=3, mode='max')
	ck = ModelCheckpoint(
		filepath=os.path.join(hparams.ckpath, 'bert-finetune-{epoch}-{val_loss:.4f}'),
		save_top_k=1,
		mode='min',
		save_weights_only=False,
		period=0
	)
	trainer_ft = pl.Trainer(
		max_epochs=hparams.epoch,
		gpus=hparams.gpus,
		progress_bar_refresh_rate=10,
		weights_summary=None,
		gradient_clip_val=100,
		num_sanity_val_steps=0,
		callbacks=[PyLoggerCallback(logger)],
		early_stop_callback=es,
		checkpoint_callback=ck
	)
	net.model._set_fine_tune()
	trainer_ft.fit(net)


if __name__=='__main__':
	cwd = os.getcwd()

	parser = ArgumentParser()
	group = parser.add_mutually_exclusive_group()
	group.add_argument('--findlr', action='store_true', default=False)
	group.add_argument('--finetune', action='store_true', default=False)
	parser.add_argument('--gpus', type=int, default=1)
	parser.add_argument('--lr', type=float, default=7e-5)
	parser.add_argument('--epoch', type=int, default=40)
	parser.add_argument('--logpath', type=str, default=os.path.join(cwd, 'bert_base_exp.log'))
	parser.add_argument('--ckpath', type=str, default=os.path.join(cwd, 'bert_base_checkpoint_artifact'))
	parser.add_argument('--ckfile', type=str)

	args = parser.parse_args()

	if args.findlr:
		find_lr(args)
	elif args.finetune:
		finetune_train(args)