# coding=utf-8

import torch
import re
import pandas as pd
import json
from torch.nn.utils.rnn import pad_sequence
from seqeval.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset

def evaluate_(output, labels, ignore_idx):
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != ignore_idx).squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    l = labels.squeeze()[idxs]; o = o_labels[idxs]

    if len(idxs) > 1:
        acc = (l == o).sum().item()/len(idxs)
    else:
        acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

    return acc, (o, l)

def evaluate_results(net, test_loader, pad_id, cuda):
    print("Evaluating test samples...")
    acc = 0; out_labels = []; true_labels = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                          e1_e2_start=e1_e2_start)

            accuracy, (o, l) = evaluate_(classification_logits, labels, ignore_idx=-1)
            out_labels.append([str(i) for i in o]); true_labels.append([str(i) for i in l])
            acc += accuracy

    accuracy = acc/(i + 1)
    results = {
        "accuracy": accuracy,
        "precision": precision_score(true_labels, out_labels),
        "recall": recall_score(true_labels, out_labels),
        "f1": f1_score(true_labels, out_labels)
    }
    print("***** Eval results *****")
    for key in sorted(results.keys()):
        print("  %s = %s" % (key, str(results[key])))

    return results

def process_text(text, mode='train'):
    sents, relations, comments, blanks = [], [], [], []
    for i in range(int(len(text)/4)):
        sent = text[4*i]
        relation = text[4*i + 1]
        comment = text[4*i + 2]
        blank = text[4*i + 3]

        # check entries
        if mode == 'train':
            assert int(re.match("^\d+", sent)[0]) == (i + 1)
        else:
            assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
        assert re.match("^Comment", comment)
        assert len(blank) == 1

        sent = re.findall("\"(.+)\"", sent)[0]
        sent = re.sub('<e1>', '[E1]', sent)
        sent = re.sub('</e1>', '[/E1]', sent)
        sent = re.sub('<e2>', '[E2]', sent)
        sent = re.sub('</e2>', '[/E2]', sent)
        sents.append(sent); relations.append(relation), comments.append(comment); blanks.append(blank)
    return sents, relations, comments, blanks

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}

        print("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in relations:
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1

        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1,\
                 ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])

        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])

        return seqs_padded, labels_padded, labels2_padded, x_lengths, y_lengths, y2_lengths

class semeval_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        print("Tokenizing data...")
        self.df['input'] = self.df.apply(lambda x: tokenizer.encode(x['sents']), axis=1)

        def get_e1e2_start(x, e1_id, e2_id):
            e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                            [i for i, e in enumerate(x) if e == e2_id][0])
            return e1_e2_start

        self.df['e1_e2_start'] = self.df.apply(lambda x: get_e1e2_start(x['input'], e1_id=self.e1_id, e2_id=self.e2_id), axis=1)

    def __len__(self,):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']),\
                torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
                torch.LongTensor([self.df.iloc[idx]['relations_id']])

def preprocess_semeval2010_8(train_data, test_data):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = train_data
    print("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    sents, relations, comments, blanks = process_text(text, 'train')
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})

    data_path = test_data
    print("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    sents, relations, comments, blanks = process_text(text, 'test')
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})

    rm = Relations_Mapper(df_train['relations'])

    df_test['relations_id'] = df_test.apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.apply(lambda x: rm.rel2idx[x['relations']], axis=1)

    return df_train, df_test, rm

def detokenize(text, h, t):
    text_with_ents = []
    for i, token in enumerate(text):
        if i==h['pos'][0]:
            text_with_ents.append('[E1]')
        if i==h['pos'][1]:
            text_with_ents.append('[/E1]')
        if i==t['pos'][0]:
            text_with_ents.append('[E2]')
        if i==t['pos'][1]:
            text_with_ents.append('[/E2]')
        text_with_ents.append(token)
    return ' '.join(text_with_ents)

def process_wiki80_text(text):
    sents, relations = [], []
    for i in range(len(text)):
        text_obj = json.loads(text[i])
        sent = detokenize(text_obj['token'], text_obj['h'], text_obj['t'])
        relation = text_obj['relation']

        sents.append(sent); relations.append(relation)
    return sents, relations

def preprocess_wiki80(train_data, test_data):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = train_data
    print("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    sents, relations = process_wiki80_text(text)
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})

    data_path = test_data
    print("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    sents, relations = process_wiki80_text(text)
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})

    rm = Relations_Mapper(df_train['relations'])

    df_test['relations_id'] = df_test.apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.apply(lambda x: rm.rel2idx[x['relations']], axis=1)

    return df_train, df_test, rm