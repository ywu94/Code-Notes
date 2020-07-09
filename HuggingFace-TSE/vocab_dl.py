import os
import wget

def download(url, tar):
	if not os.path.isfile(tar):
		wget.download(url, out=tar)

if __name__=='__main__':
	cwd = os.getcwd()
	vocab_dir = os.path.join(cwd, 'vocab_artifact')
	if not os.path.isdir(vocab_dir): os.mkdir(vocab_dir)

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))

	file_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json'
	download(file_url, os.path.join(vocab_dir, file_url.split('/')[-1]))