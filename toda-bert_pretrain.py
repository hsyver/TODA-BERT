import logging
import sys
import os
import pathlib
import time
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import MobileBertTokenizer, MobileBertModel
from sklearn.preprocessing import MultiLabelBinarizer
from torch import cuda

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def save_model(model, save_dir, epoch, iteration, run):
    filename = 'model_Epoch'+str(epoch)+'It'+str(iteration)
    file = os.path.join(save_dir, "{}.pth".format(filename))
    torch.save(model.state_dict(), file)
    logger.info("Model saved")

    last_checkpoint = os.path.join(save_dir, 'last_checkpoint'+str(run)+'.txt')
    with open(last_checkpoint, "w") as f:
        f.write(file)

def load_model(model, last_checkpoint):
    last_checkpoint = os.path.join(OUTPUT_DIR, last_checkpoint)
    last_saved = None
    if os.path.exists(last_checkpoint):
        try:
            with open(last_checkpoint, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            last_saved = ""

    if not last_saved:
        logger.info("No checkpoint found")
        return model

    logger.info(f'Loading checkpoint from {last_saved}')
    model.load_state_dict(torch.load(last_saved, map_location=torch.device("cpu")))
    return model

def load_data(sample, data_dir):
    dstc2and3 = pd.read_csv(data_dir+'/dstc2and3/dstc2and3_train.csv')
    m2m = pd.read_csv(data_dir+'/m2m/m2m_train.csv')
    frames = pd.read_csv(data_dir+'/frames/frames_train.csv')
    sgd = pd.read_csv(data_dir+'/sgd/sgd_train.csv')
    e2e = pd.read_csv(data_dir+'/e2e/e2e_train.csv')
    mwoz = pd.read_csv(data_dir+'/mwoz/mwoz_train.csv')
    datasets = [dstc2and3, m2m, frames, sgd, e2e, mwoz]
    classes = [['ack'], ['affirm'], ['bye'], ['deny'], ['inform'], ['repeat'], ['reqalts'], ['request'], ['restart'], ['thank_you'], ['user-confirm'], ['user-hi'], ['user-negate']]

    label_encoder = MultiLabelBinarizer().fit(classes)

    utterances = []
    labels = []
    for dataset in datasets:
        dataset = dataset.loc[dataset['actor'] == 'User']

        downsampled_dataset = dataset
        if len(dataset) > sample:
            downsampled_dataset = downsampled_dataset.sample(frac=1).reset_index().head(sample)

        utterances.append(downsampled_dataset['utterance'])
        labels.append(label_encoder.transform(list(map(eval, downsampled_dataset['DAs']))))

    return utterances, labels

class CustomDataset(Dataset):

    def __init__(self, utterances, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.utterances = utterances
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        utterance = str(self.utterances[index])
        utterance = " ".join(utterance.split())
        inputs = self.tokenizer.encode_plus(
            utterance,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.labels[index], dtype=torch.float),
        }

def make_dataloaders(train_datasets):
    X = pd.concat([utterances[i] for i in train_datasets])
    y = np.concatenate([labels[i] for i in train_datasets])

    val_frac = 0.1
    val_indices = set(random.sample(list(range(len(X))), int(val_frac * len(X))))

    X_train = [n for i,n in enumerate(X) if i not in val_indices]
    y_train = [n for i,n in enumerate(y) if i not in val_indices]

    X_val = [n for i,n in enumerate(X) if i in val_indices]
    y_val = [n for i,n in enumerate(y) if i in val_indices]

    logger.info("FULL Dataset: {}".format(len(X)))
    logger.info("TRAIN Dataset: {}".format(len(X_train)))
    logger.info("VAL Dataset: {}".format(len(X_val)))

    training_set = CustomDataset(X_train, y_train, tokenizer, MAX_LEN)
    validation_set = CustomDataset(X_val, y_val, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    }

    val_params = {'batch_size': TRAIN_BATCH_SIZE,
                   'shuffle': True,
                   }

    train_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(validation_set, **val_params)

    return train_loader, val_loader

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 13)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)[1]
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def validation(model, val_loader, device):
    model.eval()
    targets=[]
    outputs=[]
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            target = data['targets'].to(device, dtype = torch.float)
            output = model(ids, mask, token_type_ids)
            targets.extend(target.cpu().detach().numpy().tolist())
            outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())

    outputs = np.array(outputs) >= 0.5
    microf1 = metrics.f1_score(targets, outputs, average='micro')
    macrof1 = metrics.f1_score(targets, outputs, average='macro')
    return microf1, macrof1

def train(epochs, run):
    model.train()
    start_time = time.time()
    global_step = 0

    losses = OrderedDict()
    eval_micro = OrderedDict()
    eval_macro = OrderedDict()

    for epoch in range(epochs):
        for iteration, data in enumerate(train_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses[global_step] = loss.item()
            if iteration % 10 == 0:
                lr = optimizer.param_groups[0]['lr']

                logger.info(f'Epoch: {str(epoch).zfill(3)}, Iteration: {str(iteration).zfill(3)}, Loss:  {loss.item()}, lr: {lr}')
                model.train()

            if iteration % EVAL_STEP == 0 and iteration != 0:
                micro, macro = validation(model, val_loader, device)
                logger.info(f'Validation micro: {micro}, macro: {macro}')
                eval_micro[global_step] = micro
                eval_macro[global_step] = macro
                model.train()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            global_step+=1

        micro, macro = validation(model, val_loader, device)
        logger.info(f'Validation micro: {micro}, macro: {macro}')
        eval_micro[global_step] = micro
        eval_macro[global_step] = macro
        model.train()

        if epoch % MODEL_SAVE_STEP == 0:
            save_model(model, OUTPUT_DIR, epoch, 'end', run)

    save_model(model, OUTPUT_DIR, 'final', 'final_run'+str(run), run)
    total_training_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(total_time_str)

    plt.plot(list(losses.keys()), list(losses.values()), label='training loss')
    plt.plot(list(eval_micro.keys()), list(eval_micro.values()), label='val micro')
    plt.plot(list(eval_macro.keys()), list(eval_macro.values()), label='val macro')
    plt.xlabel('iterations')
    plt.legend()
    plt.ylim(0,1)
    plt.savefig(OUTPUT_DIR+'/plot_run'+str(run)+'.png', bbox_inches='tight')
    plt.ylim(0.60, 1)
    plt.savefig(OUTPUT_DIR+'/ylim_plot_run'+str(run)+'.png', bbox_inches='tight')

if __name__ == "__main__":
    OUTPUT_DIR = sys.argv[1]
    DATA_DIR = sys.argv[2]
    MAX_LEN = 80
    TRAIN_BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 3e-05
    WEIGHT_DECAY = 0.01
    EVAL_STEP = 2000
    MODEL_SAVE_STEP = 2 #in epochs
    TRAIN_DATASETS = [0, 1, 2, 3, 4, 5]
    DATASETS = {0: 'dstc2and3', 1: 'm2m', 2: 'frames', 3: 'sgd', 4: 'e2e', 5: 'mwoz'}
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    device = 'cuda' if cuda.is_available() else 'cpu'

    logger = setup_logger('pretraining', OUTPUT_DIR)

    RUNS = 1
    for run in range(RUNS):
        utterances, labels = load_data(SAMPLE, DATA_DIR)

        model = BERTClass()
        model.to(device)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        train_loader, val_loader = make_dataloaders(TRAIN_DATASETS)

        num_steps = len(train_loader) * EPOCHS
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        logger.info(f'Training datasets: {[DATASETS[i] for i in TRAIN_DATASETS]}')
        logger.info(f'\n MAX_LEN: {MAX_LEN} \n TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE} \n EPOCHS: {EPOCHS} \n LEARNING_RATE: {LEARNING_RATE} \n WEIGHT_DECAY: {WEIGHT_DECAY} \n MODEL: {model}')

        logger.info('RUN '+str(run))
        train(EPOCHS, run)