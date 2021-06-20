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
The provided fine-tuning code assumes data is in .csv format with the fields *utterance*, *DAs*, and *actor* (see [sample_dialogue.csv](sample_dialogue.csv)).

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

## Results 
The following table shows micro and macro-F1 scores achieved when fine-tuning on each of the pre-training datasets. The dataset being fine-tuned on was left out from pre-training.

<table>
  <tr>
    <td></td>
    <td colspan="2" align="center">DSTC</td>
    <td colspan="2" align="center">M2M</td>
    <td colspan="2" align="center">Frames</td>
    <td colspan="2" align="center">SGD</td>
    <td colspan="2" align="center">E2E</td>
    <td colspan="2" align="center">MWOZ</td>
  </tr>
  <tr>
    <td></td>
    <td>micro</td>
    <td>macro</td>
    <td>micro</td>
    <td>macro</td>
    <td>micro</td>
    <td>macro</td>
    <td>micro</td>
    <td>macro</td>
    <td>micro</td>
    <td>macro</td>
    <td>micro</td>
    <td>macro</td>
  </tr>
  <tr>
    <td>BERT</td>
    <td>98.10</td>
    <td>74.35</td>
    <td>94.29</td>
    <td>82.46</td>
    <td>79.35</td>
    <td>52.58</td>
    <td>92.81</td>
    <td>90.20</td>
    <td>91.03</td>
    <td>47.27</td>
    <td>97.85</td>
    <td>38.40</td>
  </tr>
   <tr>
    <td>TODA-BERT-add</td>
    <td>98.15</td>
    <td>81.13</td>
    <td>93.81</td>
    <td>85.80</td>
    <td>78.09</td>
    <td>51.25</td>
    <td>92.87</td>
    <td>90.31</td>
    <td>90.89</td>
    <td>48.44</td>
    <td>97.78</td>
    <td>42.17</td>
  </tr>
</table>

Results can be reproduced by fine-tuning the respective leave-one-out pre-trained models available from [here](https://drive.google.com/file/d/1-5XPj_z7tFhuoCFfAiz8okO_EyMFI8Sx/view?usp=sharing). Domain information was stripped from the labels in MultiWOZ 2.3 resulting in 12 user acts. The test set of DSTC 2 was used for the combined dataset of DSTC 2 and 3.
