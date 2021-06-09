# TODA-BERT (Task-oriented Dialogue Act BERT)
Code and pre-trained model is uploaded as a part of the master's thesis *Supervised Pre-training for Dialogue Act Classifiaction in Task-oriented Dialogue*

TODA-BERT is a BERT-based model for dialogue act classification in task-oriented dialogue. The model is pre-trained in a supervised fashion with a pre-training corpus consisting of the following datasets: 
- [DSTC 2](https://www.aclweb.org/anthology/W14-4337.pdf)
- [DSTC 3](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/write_up.pdf)
- [MultiWOZ 2.3](https://arxiv.org/pdf/2010.05594.pdf)
- [Frames](https://www.aclweb.org/anthology/W17-5526v2.pdf) 
- [SGD](https://arxiv.org/pdf/1909.05855.pdf)
- [M2M](https://arxiv.org/pdf/1801.04871.pdf)
- [E2E](https://arxiv.org/pdf/1807.11125.pdf)

Datasets were aligned to the universal DA schema from [Paul et al.](https://arxiv.org/pdf/1907.03020.pdf) before pre-training. TODA-BERTs output size is 13, the same as the number of user acts in the universal DA schema.

The pre-trained model can be downloaded from the following link:
- [TODA-BERT](https://drive.google.com/file/d/1GB7yPYhvOAli_10Dt7OE0mgubFfwWfr2/view?usp=sharing)

## Fine-tuning
The provided fine-tuning code assumes data is in .csv format with the fields *utterance*, *DAs*, and *actor*.

Fine-tuning with the TODA-BERT-add architecture:
```shell
python toda-bert_finetune.py
  --output_dir=/path/to/output_dir
  --train_data_src=/path/to/train_data
  --test_data_src=/path/to/test_data
  --load_dir=/path/to/pretrained
  --add_layer=True
```

Fine-tuning with the TODA-BERT-filter architecture (recommended if dialogue acts follow the universal DA schema):
```shell
python toda-bert_finetune.py
  --output_dir=/path/to/output_dir
  --train_data_src=/path/to/train_data
  --test_data_src=/path/to/test_data
  --load_dir=/path/to/pretrained
```
