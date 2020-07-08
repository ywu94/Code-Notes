# Code-Notes

Codes for some projects

## Hugging Face Experiments with Twitter Sentiment Extraction Dataset

Bert: `0.709` 

#### Execution

```
python vocab_dl.py
python prepare.py -r 0.8 -s 1898
python train_model.py bert --findlr 
python train_model.py bert --finetune --lr 5e-5
```

#### References

[1] Kaggle Twitter Sentiment Extraction Competition: https://www.kaggle.com/c/tweet-sentiment-extraction

[2] Rank 1 Solution: [Post](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477#891253), [Notebook 1](https://www.kaggle.com/aruchomu/no-sampler-ensemble-normal-sub-0-7363), [Notebook 2](https://www.kaggle.com/theoviel/character-level-model-magic/)

[3] Other Solution: [Rank 13 Post](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159505)

[4] BERT Fine Tune: [Overview](https://zhuanlan.zhihu.com/p/62642374?utm_source=wechat_session&utm_medium=social&utm_oi=629832652505616384), [Constructing Auxiliary Sentence](https://arxiv.org/pdf/1903.09588.pdf)

[5] Hugging Face Tokenizer:  [Overview](https://towardsdatascience.com/comparing-transformer-tokenizers-686307856955), [Quick Start](https://heartbeat.fritz.ai/hands-on-with-hugging-faces-new-tokenizers-library-baff35d7b465)

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
