import os
import random
import pickle
import argparse
import pandas as pd

def prepare_train(train_ratio=0.8, seed=1898):
	random.seed(seed)
	cwd = os.getcwd()
	df = pd.read_csv(os.path.join(cwd, 'raw_artifact', 'train.csv'))

	positive_idx = df[df['sentiment']=='positive'].index.tolist()
	random.shuffle(positive_idx)
	neutral_idx = df[df['sentiment']=='neutral'].index.tolist()
	random.shuffle(neutral_idx)
	negative_idx = df[df['sentiment']=='negative'].index.tolist()
	random.shuffle(negative_idx)

	data_dir = os.path.join(cwd, 'data_artifact')
	if not os.path.isdir(data_dir): os.mkdir(data_dir)

	train_idx = (
		positive_idx[:int(len(positive_idx)*train_ratio)]
		+ neutral_idx[:int(len(neutral_idx)*train_ratio)]
		+ negative_idx[:int(len(negative_idx)*train_ratio)]
	)
	random.shuffle(train_idx)
	df.iloc[train_idx,:].to_csv(os.path.join(data_dir, 'train.csv'), index=False)

	validation_idx = (
		positive_idx[int(len(positive_idx)*train_ratio):]
		+ neutral_idx[int(len(neutral_idx)*train_ratio):]
		+ negative_idx[int(len(negative_idx)*train_ratio):]
	)
	random.shuffle(validation_idx)
	df.iloc[validation_idx,:].to_csv(os.path.join(data_dir, 'validation.csv'), index=False)

def prepare_test():
	cwd = os.getcwd()
	df = pd.read_csv(os.path.join(cwd, 'raw_artifact', 'test.csv'))

	data_dir = os.path.join(cwd, 'data_artifact')
	if not os.path.isdir(data_dir): os.mkdir(data_dir)

	df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--ratio', type=float, default=0.8, help='ratio of training data')
	parser.add_argument('-s', '--seed', type=int, default=1898, help='random seed')
	args = parser.parse_args()

	prepare_train(train_ratio=args.ratio, seed=args.seed)
	prepare_test()