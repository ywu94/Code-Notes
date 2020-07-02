import os
import random
import pickle
import argparse
import pandas as pd

def prepare_train(train_ratio=0.8, seed=1898):
	random.seed(seed)
	tr, vl, cwd = [], [], os.getcwd()
	for directory in os.listdir(os.path.join(cwd, 'train_artifact')):
		label = int(directory)
		file_list = [(os.path.join(cwd, 'train_artifact', directory,f), label) 
			for f in os.listdir(os.path.join(cwd, 'train_artifact', directory)) if f[-3:]=='ppm']
		random.shuffle(file_list)
		split_index = int(len(file_list)*train_ratio)
		tr.extend(file_list[:split_index])
		vl.extend(file_list[split_index:])
	random.shuffle(tr)
	with open(os.path.join(cwd, 'train_idx.pkl'), 'wb') as f:
		pickle.dump(tr, f)
	random.shuffle(vl)
	with open(os.path.join(cwd, 'validation_idx.pkl'), 'wb') as f:
		pickle.dump(vl, f)

def prepare_test():
	cwd = os.getcwd()
	anno = pd.read_csv(os.path.join(cwd, 'test_artifact', 'GT-final_test.csv'), delimiter=';')
	te = [(os.path.join(cwd, 'test_artifact', row[0]), int(row[-1])) for row in anno.values]
	with open(os.path.join(cwd, 'test_idx.pkl'), 'wb') as f:
		pickle.dump(te, f)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--ratio', type=float, default=0.8, help='ratio of training data')
	parser.add_argument('-s', '--seed', type=int, default=1898, help='random seed')
	args = parser.parse_args()

	prepare_train(train_ratio=args.ratio, seed=args.seed)
	prepare_test()

