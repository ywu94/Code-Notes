import os
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

import torch
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

from utils import initiate_logger

def jaccard_iterable(a, b):
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c))/(len(a)+len(b)-len(c))

def get_bert_tokenizer(vocab_path):
    tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=True)
    spc_token = {
        'cls': tokenizer.token_to_id('[CLS]'),
        'sep': tokenizer.token_to_id('[SEP]'),
        'pad': tokenizer.token_to_id('[PAD]')
    }
    for sentiment in ['positive', 'negative', 'neutral']:
        ids = tokenizer.encode(sentiment).ids
        spc_token[sentiment] = ids[0] if ids[0] != spc_token['cls'] else ids[1]
    return tokenizer, spc_token

def process_bert_data(text, sentiment, tokenizer, spc_token, selected_text=None, max_len=128, alpha=0.3):
    text = ' ' + ' '.join(str(text).split())
    if selected_text is not None: 
        selected_text = ' ' + ' '.join(str(selected_text).split())

    idx_start, idx_end, char_target = None, None, None
    if selected_text is not None:
        for idx in [i for i, e in enumerate(text) if e==selected_text[1]]:
            if ' ' + text[idx:idx+len(selected_text)-1] == selected_text:
                idx_start = idx
                idx_end = idx + len(selected_text) - 2
                break 
        assert idx_start is not None, 'Error in text: {}, selected_text: {}'.format(text, selected_text)
        char_target = [0 for _ in range(len(text))]
        for idx in range(idx_start, idx_end+1): char_target[idx] = 1

    tokenized = tokenizer.encode(text)
    assert tokenized.ids[0]==spc_token['cls'] and tokenized.ids[-1]==spc_token['sep']
    ids = tokenized.ids[1:-1]
    tokens = tokenized.tokens[1:-1]
    offsets = tokenized.offsets[1:-1]

    ids = [spc_token['cls'], spc_token[sentiment], spc_token['sep']] + ids[:max_len-4] + [spc_token['sep']]
    tokens = ['[CLS]', sentiment, '[SEP]'] + tokens[:max_len-4] + [['SEP']]
    type_ids = [0, 0, 0] + [1 for _ in range(len(ids)-3)]
    offsets = [(0,0) for _ in range(3)] + offsets[:max_len-4] + [(0,0)]
    mask = [1 for _ in range(len(ids))]

    token_target_start, token_target_end = None, None
    if char_target is not None:
        token_target = []
        for idx, (idx_start, idx_end) in enumerate(offsets):
            if sum(char_target[idx_start:idx_end+1])>0:
                token_target.append(idx)
        token_target_start = token_target[0]
        token_target_end = token_target[-1]

    tar_start, tar_end = np.zeros(len(ids)), np.zeros(len(ids))
    if token_target_start is not None:
        sentence = np.arange(len(ids))
        answer = sentence[token_target_start:token_target_end+1]

        for i in range(3,token_target_end+1):
            jac_score = jaccard_iterable(answer, sentence[i:token_target_end+1])
            tar_start[i] = jac_score + jac_score**2
        tar_start = (1-alpha)*tar_start/tar_start.sum()
        tar_start[token_target_start] += alpha
 
        for i in range(token_target_start,len(ids)-1):
            jac_score = jaccard_iterable(answer, sentence[token_target_start:i+1])
            tar_end[i] = jac_score + jac_score**2
        tar_end = (1-alpha)*tar_end/tar_end.sum()
        tar_end[token_target_end] += alpha

    padding_len = max_len - len(ids)
    ids = ids + [spc_token['pad'] for _ in range(padding_len)]
    tokens = tokens + ['[PAD]' for _ in range(padding_len)]
    type_ids = type_ids + [0 for _ in range(padding_len)]
    offsets = offsets + [(0,0) for _ in range(padding_len)]
    mask = mask + [0 for _ in range(padding_len)]
    tar_start = list(tar_start) + [0.0 for _ in range(padding_len)]
    tar_end = list(tar_end) + [0.0 for _ in range(padding_len)]

    return {
        'ids': ids, 
        'tokens': tokens,
        'type_ids': type_ids,
        'offsets': offsets,
        'mask': mask,
        'tar_start': tar_start,
        'tar_end': tar_end,
        'text': text,
        'selected_text': selected_text,
        'sentiment': sentiment
    }

def get_ByteLevelBPE_tokenizer(vocab_path, merge_path):
    tokenizer = ByteLevelBPETokenizer(vocab_path, merge_path, lowercase=True, add_prefix_space=True)
    spc_token = {'cls': 0, 'sep': 2, 'pad': 1}
    for sentiment in ['positive', 'negative', 'neutral']:
        ids = tokenizer.encode(sentiment).ids
        spc_token[sentiment] = ids[0] if ids[0] != spc_token['cls'] else ids[1]
    return tokenizer, spc_token

def process_roberta_data(text, sentiment, tokenizer, spc_token, selected_text=None, max_len=128, alpha=0.3):
    text = ' ' + ' '.join(str(text).split())
    if selected_text is not None: 
        selected_text = ' ' + ' '.join(str(selected_text).split())

    idx_start, idx_end, char_target = None, None, None
    if selected_text is not None:
        for idx in [i for i, e in enumerate(text) if e==selected_text[1]]:
            if ' ' + text[idx:idx+len(selected_text)-1] == selected_text:
                idx_start = idx
                idx_end = idx + len(selected_text) - 2
                break 
        assert idx_start is not None, 'Error in text: {}, selected_text: {}'.format(text, selected_text)
        char_target = [0 for _ in range(len(text))]
        for idx in range(idx_start, idx_end+1): char_target[idx] = 1

    tokenized = tokenizer.encode(text)
    ids = tokenized.ids
    tokens = tokenized.tokens
    offsets = tokenized.offsets

    ids = [spc_token['cls'], spc_token[sentiment], spc_token['sep']] + ids[:max_len-4] + [spc_token['sep']]
    tokens = ['[CLS]', sentiment, '[SEP]'] + tokens[:max_len-4] + [['SEP']]
    type_ids = [0, 0, 0] + [0 for _ in range(len(ids)-3)]
    offsets = [(0,0) for _ in range(3)] + offsets[:max_len-4] + [(0,0)]
    mask = [1 for _ in range(len(ids))]

    token_target_start, token_target_end = None, None
    if char_target is not None:
        token_target = []
        for idx, (idx_start, idx_end) in enumerate(offsets):
            if sum(char_target[idx_start:idx_end+1])>0:
                token_target.append(idx)
        token_target_start = token_target[0]
        token_target_end = token_target[-1]

    tar_start, tar_end = np.zeros(len(ids)), np.zeros(len(ids))
    if token_target_start is not None:
        sentence = np.arange(len(ids))
        answer = sentence[token_target_start:token_target_end+1]

        for i in range(3,token_target_end+1):
            jac_score = jaccard_iterable(answer, sentence[i:token_target_end+1])
            tar_start[i] = jac_score + jac_score**2
        tar_start = (1-alpha)*tar_start/tar_start.sum()
        tar_start[token_target_start] += alpha
 
        for i in range(token_target_start,len(ids)-1):
            jac_score = jaccard_iterable(answer, sentence[token_target_start:i+1])
            tar_end[i] = jac_score + jac_score**2
        tar_end = (1-alpha)*tar_end/tar_end.sum()
        tar_end[token_target_end] += alpha

    padding_len = max_len - len(ids)
    ids = ids + [spc_token['pad'] for _ in range(padding_len)]
    tokens = tokens + ['[PAD]' for _ in range(padding_len)]
    type_ids = type_ids + [0 for _ in range(padding_len)]
    offsets = offsets + [(0,0) for _ in range(padding_len)]
    mask = mask + [0 for _ in range(padding_len)]
    tar_start = list(tar_start) + [0.0 for _ in range(padding_len)]
    tar_end = list(tar_end) + [0.0 for _ in range(padding_len)]

    return {
        'ids': ids, 
        'tokens': tokens,
        'type_ids': type_ids,
        'offsets': offsets,
        'mask': mask,
        'tar_start': tar_start,
        'tar_end': tar_end,
        'text': text,
        'selected_text': selected_text,
        'sentiment': sentiment
    }

class TSE_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, spc_token, process_fn, train=True, max_len=64, alpha=0.3):
        self.df = df
        self.tokenizer = tokenizer
        self.spc_token = spc_token
        self.process_fn = process_fn
        self.train = train
        self.max_len = max_len
        self.alpha = alpha

        self.texts = df['text'].values
        self.sentiments = df['sentiment'].values
        if train: self.selected_texts = df['selected_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.train:
            res = self.process_fn(self.texts[idx], self.sentiments[idx], 
                self.tokenizer, self.spc_token, max_len=self.max_len,
                alpha = self.alpha, selected_text=self.selected_texts[idx]
            )
            return {
                'ids': torch.tensor(res['ids'], dtype=torch.long),
                'type_ids': torch.tensor(res['type_ids'], dtype=torch.long),
                'offsets': torch.tensor(res['offsets'], dtype=torch.long),
                'mask': torch.tensor(res['mask'], dtype=torch.long),
                'start_label': torch.tensor(res['tar_start'], dtype=torch.float),
                'end_label': torch.tensor(res['tar_end'], dtype=torch.float),
                'tokens': res['tokens'],
                'text': res['text'],
                'sentiment': res['sentiment'],
                'selected_text': res['selected_text']
            }
        else:
            res = process_bert_data(self.texts[idx], self.sentiments[idx], 
                self.tokenizer, self.spc_token, max_len=self.max_len
            )
            return {
                'ids': torch.tensor(res['ids'], dtype=torch.long),
                'type_ids': torch.tensor(res['type_ids'], dtype=torch.long),
                'offsets': torch.tensor(res['offsets'], dtype=torch.long),
                'mask': torch.tensor(res['mask'], dtype=torch.long),
                'tokens': res['tokens'],
                'text': res['text'],
                'sentiment': res['sentiment']
            }

def get_TSE_dataloader_kfold(target, fold, batch_size, config):
    assert fold>=0 and fold<config.n_splits
    cwd = os.getcwd()
    if isinstance(config.vocab, str):
        vocab_path = os.path.join(cwd, 'vocab_artifact', config.vocab)
        tokenizer, spc_token = config.tokenizer_fn(vocab_path)
    else:
        vocab_path = os.path.join(cwd, 'vocab_artifact', config.vocab['vocab_file'])
        merge_path = os.path.join(cwd, 'vocab_artifact', config.vocab['merge_file'])
        tokenizer, spc_token = config.tokenizer_fn(vocab_path, merge_path)

    if target=='test':
        df = pd.read_csv(os.path.join(cwd, 'data_artifact', 'test.csv'))
        train, shuffle = False, False
    else:
        data = pd.read_csv(os.path.join(cwd, 'data_artifact', 'train_fold.csv'))
        kf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        idx_list = [(train_idx, val_idx) for train_idx, val_idx in kf.split(np.arange(data.shape[0]), data['sentiment'].values)]
        if target=='train':
            df = data.iloc[idx_list[fold][0], :]
            train, shuffle = True, True
        elif target=='validation':
            df = data.iloc[idx_list[fold][1], :]
            train, shuffle = True, False

    dataset = TSE_Dataset(df, tokenizer, spc_token, config.process_fn, train=train, max_len=config.max_len, alpha=config.alpha)
    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=config.pin_memory)

    return dataloader