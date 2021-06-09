import logging
import argparse
import sys
import os
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

    fh = logging.FileHandler(os.path.join(save_dir, name+'_log.txt'))
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

def load_model(model, load_src):
    if os.path.exists(load_src):
        logger.info(f'Loading checkpoint from {load_src}')
        model.load_state_dict(torch.load(load_src, map_location=torch.device("cpu")))
    return model

def load_data(data_source):
    dataset = pd.read_csv(data_source)

    data = dataset.loc[dataset['actor'] == 'User']  #filter to only include the user utterances of the dataset

    all_classes = [['ack'], ['affirm'], ['bye'], ['deny'], ['inform'], ['repeat'], ['reqalts'], ['request'], ['restart'], ['thank_you'], ['user-confirm'], ['user-hi'], ['user-negate']]    #user dialogue acts from the universal DA schema
    classes = []    #will store dialogue acts present in the training set

    def find_classes(acts):
        for act in acts:
            if [act] not in classes:
                classes.append([act])
        return acts

    data['DAs'].apply(lambda x: find_classes(eval(x)))
    label_encoder = MultiLabelBinarizer().fit(all_classes)

    if ADD_LAYER or REPLACE:
        label_encoder = MultiLabelBinarizer().fit(classes)

    data = data.sample(frac=1).reset_index()

    utterances = data['utterance']
    labels = label_encoder.transform(list(map(eval, data['DAs'])))
    numclasses = len(classes)

    return utterances, labels, label_encoder, numclasses, sorted(all_classes), sorted(classes)

def load_test_data(data_source, label_encoder):
    dataset = pd.read_csv(data_source)

    data = dataset.loc[dataset['actor'] == 'User']

    utterances = data['utterance']
    labels = label_encoder.transform(list(map(eval, data['DAs'])))

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

def make_dataloader(utterances, labels, frac):
    X = utterances
    y = labels

    frac = 1-frac
    indices = set(random.sample(list(range(len(X))), int(frac * len(X))))

    X_data = [n for i,n in enumerate(X) if i not in indices]
    y_data = [n for i,n in enumerate(y) if i not in indices]

    dataset = CustomDataset(X_data, y_data, tokenizer, MAX_LEN)

    params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    }

    data_loader = DataLoader(dataset, **params)

    return data_loader

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


class BERTClassAddLayer(torch.nn.Module):
    def __init__(self, pretrained_model, numclasses):
        super(BERTClassAddLayer, self).__init__()
        self.pretrained = pretrained_model
        self.l4 = torch.nn.Linear(13, numclasses)

    def forward(self, ids, mask, token_type_ids):
        x = self.pretrained(ids, mask, token_type_ids)
        x = self.l4(x)
        return x


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def validation(model, val_loader, device, classes, eval_classes):
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

    if not ADD_LAYER and not REPLACE:
        eval_indices = []
        for c in eval_classes:
            eval_indices.append(classes.index(c))

        targets = torch.index_select(torch.tensor(targets), 1, torch.tensor(eval_indices)).tolist()  # only look at classes present in the target dataset when computing evaluation metrics
        outputs = torch.index_select(torch.tensor(outputs), 1, torch.tensor(eval_indices)).tolist()

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

            if iteration % EVAL_STEP == 0:
                micro, macro = validation(model, test_loader, device, univ_classes, eval_classes)
                logger.info(f'Validation micro: {micro}, macro: {macro}')
                eval_micro[global_step] = micro
                eval_macro[global_step] = macro
                model.train()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            global_step+=1

        micro, macro = validation(model, test_loader, device, univ_classes, eval_classes)
        logger.info(f'Validation micro: {micro}, macro: {macro}')
        eval_micro[global_step] = micro
        eval_macro[global_step] = macro
        model.train()

        if (epoch % MODEL_SAVE_STEP == 0 and epoch != 0) or MODEL_SAVE_STEP == 1:
            save_model(model, OUTPUT_DIR, epoch, 'end', run)

    save_model(model, OUTPUT_DIR, 'final', 'final_run'+str(run), run)
    total_training_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(total_time_str)

    plt.clf()
    plt.plot(list(losses.keys()), list(losses.values()), label='training loss')
    plt.plot(list(eval_micro.keys()), list(eval_micro.values()), label='test micro_f1')
    plt.plot(list(eval_macro.keys()), list(eval_macro.values()), label='test macro_f1')
    plt.xlabel('iterations')
    plt.legend()
    plt.ylim(0,1)
    plt.savefig(OUTPUT_DIR+'/plot_run'+str(run)+'.png', bbox_inches='tight')
    plt.ylim(0.30, 1)
    plt.savefig(OUTPUT_DIR+'/ylim_plot_run'+str(run)+'.png', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Output directory', type=str, default='')
    parser.add_argument('--train_data_src', help='Training data location', type=str, default='')
    parser.add_argument('--test_data_src', help='Test data location', type=str, default='')
    parser.add_argument('--frac', help='Fraction of training data to use', type=float, default=1)
    parser.add_argument('--load_src', help='Pre-trained model location', type=str, default='')
    parser.add_argument('--runs', help='Number of runs', type=int, default=1)
    parser.add_argument('--add_layer',
                        help='Add linear classification layer on top of pre-trained model. Mutually exclusive with --replace',
                        type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y', 'on']), default=False)
    parser.add_argument('--replace',
                        help='Replace linear classification layer of pre-trained model. Mutually exclusive with --add_layer',
                        type=lambda x: (str(x).lower() in ['true', 't', '1', 'yes', 'y', 'on']), default=False)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    TRAIN_DATA_SRC = args.train_data_src
    TEST_DATA_SRC = args.test_data_src
    TRAINING_DATA_FRACTION = args.frac
    LOAD_SRC = args.load_src
    ADD_LAYER = args.add_layer
    REPLACE = args.replace
    RUNS = args.runs

    MAX_LEN = 80
    TRAIN_BATCH_SIZE = 16
    EPOCHS = 4
    LEARNING_RATE = 3e-05
    WEIGHT_DECAY = 0.01
    MODEL_SAVE_STEP = 100 #in epochs
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    device = 'cuda' if cuda.is_available() else 'cpu'

    logger = setup_logger('training', OUTPUT_DIR)
    eval_logger = setup_logger('evaluation', OUTPUT_DIR)

    microf1 = []
    macrof1 = []
    for run in range(RUNS):
        utterances, labels, label_encoder, numclasses, univ_classes, eval_classes = load_data(TRAIN_DATA_SRC)
        test_utterances, test_labels = load_test_data(TEST_DATA_SRC, label_encoder)

        model = BERTClass()

        if LOAD_SRC:    #TODA-BERT-filter
            model = load_model(model, LOAD_SRC)

        if ADD_LAYER:   #TODA-BERT-add
            model = BERTClassAddLayer(model, numclasses)

        elif REPLACE:   #TODA-BERT-replace
            model.l3 = torch.nn.Linear(768, numclasses)

        model.to(device)

        optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        train_loader = make_dataloader(utterances, labels, TRAINING_DATA_FRACTION)
        test_loader = make_dataloader(test_utterances, test_labels, 1)

        num_steps = len(train_loader) * EPOCHS
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

        EVAL_STEP = len(train_loader)//10

        logger.info(f'Labels: {label_encoder.classes_}')
        logger.info(f'\n MAX_LEN: {MAX_LEN} \n TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE} \n EPOCHS: {EPOCHS} \n LEARNING_RATE: {LEARNING_RATE} \n WEIGHT_DECAY: {WEIGHT_DECAY} \n TRAINING_DATA_FRACTION: {TRAINING_DATA_FRACTION} \n MODEL: {model}')

        logger.info('RUN '+str(run))
        train(EPOCHS, run)

        micro, macro = validation(model, test_loader, device, univ_classes, eval_classes)
        eval_logger.info(f'Validation micro: {micro}, macro: {macro}')
        microf1.append(micro)
        macrof1.append(macro)
        model.train()

    eval_logger.info(f'Average validation micro: {sum(microf1)/len(microf1)}, macro: {sum(macrof1)/len(macrof1)}')