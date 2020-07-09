from collections import namedtuple

from dataset import get_bert_tokenizer, process_bert_data, get_ByteLevelBPE_tokenizer, process_roberta_data

from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig

DATALOADER_CONFIG = namedtuple(
	'DATALOADER_CONFIG', 
	['vocab', 'tokenizer_fn', 'process_fn', 'seed', 'n_splits','max_len','alpha','num_workers','pin_memory'],
	defaults=[1898, 5, 64, 0.3, 8, True]
)

MODEL_CONFIG = namedtuple(
	'MODEL_CONFIG',
	['model_name', 'model_cls', 'pretrain_wt', 'model_config', 'n_layer', 'multi_sample_dropout'],
	defaults=[8, True]
)

def get_config(model):
	if model=='bert-base-uncased':
		dl_config = DATALOADER_CONFIG(
			'bert-base-uncased-vocab.txt', 
			get_bert_tokenizer, 
			process_bert_data
		)
		ml_config = MODEL_CONFIG(
			model,
			BertModel,
			model,
			BertConfig.from_pretrained(model)
		)
		batch_size = 64
	elif model=='bert-large-uncased-whole-word-masking-finetuned-squad':
		dl_config = DATALOADER_CONFIG(
			'bert-large-uncased-vocab.txt', 
			get_bert_tokenizer, 
			process_bert_data
		)
		ml_config = MODEL_CONFIG(
			model,
			BertModel,
			model,
			BertConfig.from_pretrained(model)
		)
		batch_size = 16
	elif model=='roberta-base':
		dl_config = DATALOADER_CONFIG(
			{
				'vocab_file': 'roberta-base-vocab.json',
				'merge_file': 'roberta-base-merges.txt'
			},
			get_ByteLevelBPE_tokenizer,
			process_roberta_data
		)
		ml_config = MODEL_CONFIG(
			model, 
			RobertaModel,
			model, 
			RobertaConfig.from_pretrained(model)
		)
		batch_size = 64
	return dl_config, ml_config, batch_size
