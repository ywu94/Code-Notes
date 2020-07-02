# Code-Notes

Codes for some projects

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
