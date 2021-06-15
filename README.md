# TODA-BERT (Task-oriented Dialogue Act BERT)
Code and pre-trained model is uploaded as a part of the master's thesis *Supervised Pre-training for Dialogue Act Classifiaction in Task-oriented Dialogue*

TODA-BERT is a BERT-based model for dialogue act classification in task-oriented dialogue. The model is pre-trained in a supervised fashion with a pre-training corpus consisting of the following datasets: 

| **Dataset** | **#Dialogues** | **#Utterances**  | **#Domains** | **Type** |
| :----------- |:--------------:|:----------------:|:------------:|:--------:|
| [DSTC 2](https://www.aclweb.org/anthology/W14-4337.pdf) & [3](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/write_up.pdf) | 5,510 | 88,650 | 1 | Spoken |
| [Frames](https://www.aclweb.org/anthology/W17-5526v2.pdf) | 1,369 | 19,986 | 1 | Written |
| [MultiWOZ 2.3](https://arxiv.org/pdf/2010.05594.pdf) | 10,438 | 143,048 | 7 | Written |
| [E2E](https://arxiv.org/pdf/1807.11125.pdf) | 10,087 | 74,686 | 3 | Written |
| [M2M](https://arxiv.org/pdf/1801.04871.pdf) | 3,008 | 27,120 | 2 | Generated |
| [SGD](https://arxiv.org/pdf/1909.05855.pdf) | 22,825 | 463,284 | 20 | Generated |
| **Total** | 53,237 | 816,774 | 24 unique |  |

Datasets were aligned to the universal DA schema from [Paul et al.](https://arxiv.org/pdf/1907.03020.pdf) before pre-training. TODA-BERTs output size is 13, the same as the number of user acts in the universal DA schema.

The pre-trained model can be downloaded from the following link:
- [TODA-BERT](https://drive.google.com/file/d/1GB7yPYhvOAli_10Dt7OE0mgubFfwWfr2/view?usp=sharing)

## Fine-tuning
The provided fine-tuning code assumes data is in .csv format with the fields *utterance*, *DAs*, and *actor*.

Fine-tuning with the TODA-BERT-add architecture (Fig. 1):
```shell
python toda-bert_finetune.py
  --output_dir=/path/to/output_dir
  --train_data_src=/path/to/train_data.csv
  --test_data_src=/path/to/test_data.csv
  --load_src=/path/to/TODA-BERT.pth
  --add_layer=True
```

Fine-tuning with the TODA-BERT-filter architecture (Fig. 2) (recommended if dialogue acts follow the universal DA schema):
```shell
python toda-bert_finetune.py
  --output_dir=/path/to/output_dir
  --train_data_src=/path/to/train_data.csv
  --test_data_src=/path/to/test_data.csv
  --load_src=/path/to/TODA-BERT.pth
```
<p align="center">
  <img src="/figures/TODA-BERT-add.png" alt="TODA-BERT-add architecture" width="500"/>
</p>

<p align="center"><b>Figure 1:</b> The TODA-BERT-add architecture, which fine-tunes TODA-BERT with and added feed forward layer.</p>

<p align="center">
  <img src="/figures/TODA-BERT-filter.png" alt="TODA-BERT-filter architecture" width="500"/>
</p>

<p align="center"><b>Figure 2:</b> The TODA-BERT-filter architecture, which fine-tunes TODA-BERT and post-filters outputs to only include those relevant for the target dataset.</p>
