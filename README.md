# Code-Notes

Codes for some projects

## Thrift & Protobuf Experiment

```
thrift -gen py sample.thrift
mv -vi ./gen-py/sample .
protoc --python_out=. ./sample.
python write_pb2.py address_book.pb
python server.py
python client.py
```

#### References

[1] Thrift Tutorial: https://thrift-tutorial.readthedocs.io/en/latest/usage-example.html

[2] Protocol Buffer Tutorial: https://developers.google.com/protocol-buffers/docs/pythontutorial


## Hugging Face Experiments with Twitter Sentiment Extraction Dataset

Bert-Base: `0.714` , Bert-Large-WWM: `0.712`, RoBERTa-Base: `0.713`, RoBERTs-Base-SQuAD2: `0.711`, RoBERTa-Large-MNLI: `0.715`

#### Execution

```
python vocab_dl.py
python prepare.py -f
python train_model.py bert-base-uncased --finetune --lr 1.2e-4
python train_model.py bert-large-uncased-whole-word-masking-finetuned-squad --finetune --lr 1e-4
python train_model.py roberta-base --finetune --lr 1.4e-4
```

#### References

[1] Kaggle Twitter Sentiment Extraction Competition: https://www.kaggle.com/c/tweet-sentiment-extraction

[2] Rank 1 Solution: [Post](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477#891253), [Notebook 1](https://www.kaggle.com/aruchomu/no-sampler-ensemble-normal-sub-0-7363), [Notebook 2](https://www.kaggle.com/theoviel/character-level-model-magic/)

[3] Other Solution: [Dataset Preprocessing Magic](https://www.kaggle.com/tkm2261/pre-postprosessing-guc)

[4] BERT Fine Tune: [Pretrained Model Inventory](https://huggingface.co/transformers/pretrained_models.html), [Overview](https://zhuanlan.zhihu.com/p/62642374?utm_source=wechat_session&utm_medium=social&utm_oi=629832652505616384), [Constructing Auxiliary Sentence](https://arxiv.org/pdf/1903.09588.pdf)

[5] Hugging Face Tokenizer:  [Overview](https://towardsdatascience.com/comparing-transformer-tokenizers-686307856955), [Quick Start](https://heartbeat.fritz.ai/hands-on-with-hugging-faces-new-tokenizers-library-baff35d7b465)

[6] Multi-Sample Dropout for Accelerated Training and Better Generalization: [Paper](https://arxiv.org/pdf/1905.09788.pdf)

## PyTorch Lightning Experiments with German Traffic Sign Recognition Benchmark (GTSRB) Dataset

Use PyTorch Ligntning framework to fine tune a pretrained `ResNet34` model, achieved `99.17%` accuracy on test dataset.

#### Execution

```
. perequisite.sh
python prepare.py -r 0.8 -s 1898
python train_model.py --freeze --freezelr 1e-3 --gpus 1
python train_model.py --findlr --ckfile {checkpoint file name} --gpus 1
python train_model.py --finetune --ckfile {checkpoint file name} --finetunelr 1e-4 --gpus 1
python train_model.py --test --ckfile {checkpoint file name}
```

#### Reference

[1] PyTorch Lightning Documentation: https://pytorch-lightning.readthedocs.io/en/latest/

[2] Finetune Torchvision Model: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

[3] Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/pdf/1506.01186.pdf
