import os
import random
import pickle
import argparse
import pandas as pd

def preprocess(orig_text, orig_selected, sentiment):
    orig_text = str(orig_text)
    orig_selected = str(orig_selected)
    start_idx = orig_text.find(orig_selected)
    if orig_text[max(start_idx-2,0):start_idx]=='  ':
        start_idx -= 2
    if start_idx > 0 and orig_text[start_idx-1]==' ':
        start_idx -= 1
    end_idx = start_idx + len(orig_selected)
        
    start_idx = max(0, start_idx)
    
    if '  ' in orig_text[:start_idx] and sentiment!='netural':
        selected = ' '.join(orig_text.split())[start_idx:end_idx].strip()
        if len(selected)>1 and selected[-2]==' ':
            selected = selected[:-2]
    else:
        selected = orig_selected
        
    selected = selected.lstrip(".,;:")
    
    return selected

def prepare_train():
	cwd = os.getcwd()

	df = pd.read_csv(os.path.join(cwd, 'raw_artifact', 'train.csv'))
	df['selected_text'] = [preprocess(*i) for i in df[['text', 'selected_text', 'sentiment']].values]

	data_dir = os.path.join(cwd, 'data_artifact')
	if not os.path.isdir(data_dir): os.mkdir(data_dir)

	df.to_csv(os.path.join(data_dir, 'train_fold.csv'), index=False)

def prepare_train_split(train_ratio=0.8, seed=1898):
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
	parser.add_argument('-f', '--fold', action='store_true', default=False)
	parser.add_argument('-r', '--ratio', type=float, default=0.8, help='ratio of training data')
	parser.add_argument('-s', '--seed', type=int, default=1898, help='random seed')
	args = parser.parse_args()

	if args.fold:
		prepare_train()
		prepare_test()
	else:
		prepare_train_split(train_ratio=args.ratio, seed=args.seed)
		prepare_test()