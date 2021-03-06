# coding=utf-8

import re
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from itertools import permutations, combinations
from torch.utils.data import Dataset

def load_state(net, optimizer, scheduler, checkpoint_path):
    """ Loads saved model and optimizer states if exists """
    start_epoch, best_pred, checkpoint = 0, 0, None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])

        print("Loaded model and optimizer.")
    return start_epoch, best_pred

def process_textlines(text):
    text = [process_sent(sent) for sent in text]
    text = " ".join([t for t in text if t is not None])
    text = re.sub(' {2,}', ' ', text) # remove extra spaces > 1
    return text

def process_sent(sent):
    if sent not in [" ", "\n", ""]:
        sent = sent.strip("\n")
        sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
        sent = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", " ", sent)
        sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
        sent = re.sub("^ +", "", sent) # remove space in front
        sent = re.sub(r"([\.\?,!]){2,}", r"\1", sent) # remove multiple puncs
        sent = re.sub(r" +([\.\?,!])", r"\1", sent) # remove extra spaces in front of punc
        #sent = re.sub(r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), sent) # Replace all CAPS with capitalize
        return sent
    return

def get_subject_objects(sent_):
    ### get subject, object entities by dependency tree parsing
    #sent_ = next(sents_doc.sents)
    root = sent_.root
    subject = None; objs = []; pairs = []
    for child in root.children:
        #print(child.dep_)
        if child.dep_ in ["nsubj", "nsubjpass"]:
            if len(re.findall("[a-z]+",child.text.lower())) > 0: # filter out all numbers/symbols
                subject = child; #print('Subject: ', child)
        elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
            objs.append(child); #print('Object ', child)
    if (subject is not None) and (len(objs) > 0):
        for a, b in permutations([subject] + [obj for obj in objs], 2):
            a_ = [w for w in a.subtree]
            b_ = [w for w in b.subtree]
            pairs.append((a_[0] if (len(a_) == 1) else a_ , b_[0] if (len(b_) == 1) else b_))

    return pairs

def create_pretraining_corpus(raw_text, nlp, window_size=40):
    '''
    Input: Chunk of raw text
    Output: modified corpus of triplets (relation statement, entity1, entity2)
    '''
    #print("Processing sentences...")
    sents_doc = nlp(raw_text)
    ents = sents_doc.ents # get entities

    #print("Processing relation statements by entities...")
    entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
                            "WORK_OF_ART", "LAW", "LANGUAGE"]
    length_doc = len(sents_doc)
    D = []; ents_list = []
    for i in range(len(ents)):
        e1 = ents[i]
        e1start = e1.start; e1end = e1.end
        if e1.label_ not in entities_of_interest:
            continue
        if re.search("[\d+]", e1.text): # entities should not contain numbers
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            e2start = e2.start; e2end = e2.end
            if e2.label_ not in entities_of_interest:
                continue
            if re.search("[\d+]", e2.text): # entities should not contain numbers
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue

            if (1 <= (e2start - e1end) <= window_size): # check if next nearest entity within window_size
                # Find start of sentence
                punc_token = False
                start = e1start - 1
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2end
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]

                ### empty strings check ###
                for token in x:
                    assert len(token) > 0
                assert len(e1.text) > 0
                assert len(e2.text) > 0
                assert e1start != e1end
                assert e2start != e2end
                assert (e2start - e1end) > 0

                r = (x, (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r))
                D.append((r, e1.text, e2.text))
                ents_list.append((e1.text, e2.text))
                #print(e1.text,",", e2.text)
    #print("Processed dataset samples from named entity extraction:")
    #samples_D_idx = np.random.choice([idx for idx in range(len(D))],\
    #                                  size=min(3, len(D)),\
    #                                  replace=False)
    #for idx in samples_D_idx:
    #    print(D[idx], '\n')
    #ref_D = len(D)

    #print("Processing relation statements by dependency tree parsing...")
    doc_sents = [s for s in sents_doc.sents]
    for sent_ in doc_sents:
        if len(sent_) > (window_size + 1):
            continue

        left_r = sent_[0].i
        pairs = get_subject_objects(sent_)

        if len(pairs) > 0:
            for pair in pairs:
                e1, e2 = pair[0], pair[1]

                if (len(e1) > 3) or (len(e2) > 3): # don't want entities that are too long
                    continue

                e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1.text,\
                                    " ".join(w.text for w in e2) if isinstance(e2, list) else e2.text
                e1start, e1end = e1[0].i if isinstance(e1, list) else e1.i, e1[-1].i + 1 if isinstance(e1, list) else e1.i + 1
                e2start, e2end = e2[0].i if isinstance(e2, list) else e2.i, e2[-1].i + 1 if isinstance(e2, list) else e2.i + 1
                if (e1end < e2start) and ((e1text, e2text) not in ents_list):
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r))
                    D.append((r, e1text, e2text))
                    ents_list.append((e1text, e2text))

    #print("Processed dataset samples from dependency tree parsing:")
    #if (len(D) - ref_D) > 0:
    #    samples_D_idx = np.random.choice([idx for idx in range(ref_D, len(D))],\
    #                                      size=min(3,(len(D) - ref_D)),\
    #                                      replace=False)
    #    for idx in samples_D_idx:
    #        print(D[idx], '\n')
    return D

def mtb_evaluate_(lm_logits, blanks_logits, masked_for_pred, blank_labels, tokenizer, print_=True):
    '''
    evaluate must be called after loss.backward()
    '''
    # lm_logits
    lm_logits_pred_ids = torch.softmax(lm_logits, dim=-1).max(1)[1]
    lm_accuracy = ((lm_logits_pred_ids == masked_for_pred).sum().float()/len(masked_for_pred)).item()
    
    if print_:
        print("Predicted masked tokens: \n")
        print(tokenizer.decode(lm_logits_pred_ids.cpu().numpy() if lm_logits_pred_ids.is_cuda else \
                               lm_logits_pred_ids.numpy()))
        print("\nMasked labels tokens: \n")
        print(tokenizer.decode(masked_for_pred.cpu().numpy() if masked_for_pred.is_cuda else \
                               masked_for_pred.numpy()))
    
    '''
    # blanks
    blanks_diff = ((blanks_logits - blank_labels)**2).detach().cpu().numpy().sum() if blank_labels.is_cuda else\
                    ((blanks_logits - blank_labels)**2).detach().numpy().sum()
    blanks_mse = blanks_diff/len(blank_labels)
    
    if print_:
        print("Blanks MSE: ", blanks_mse)
    '''
    blanks_mse = 0
    return lm_accuracy, blanks_mse

class Two_Headed_Loss(nn.Module):
    '''
    Implements LM Loss and matching-the-blanks loss concurrently
    '''
    def __init__(self, lm_ignore_idx, use_logits=False, normalize=False):
        super(Two_Headed_Loss, self).__init__()
        self.lm_ignore_idx = lm_ignore_idx
        self.LM_criterion = nn.CrossEntropyLoss(ignore_index=self.lm_ignore_idx)
        self.use_logits = use_logits
        self.normalize = normalize

        if not self.use_logits:
            self.BCE_criterion = nn.BCELoss(reduction='mean')
        else:
            self.BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def p_(self, f1_vec, f2_vec):
        if self.normalize:
            factor = 1/(torch.norm(f1_vec)*torch.norm(f2_vec))
        else:
            factor = 1.0

        if not self.use_logits:
            p = 1/(1 + torch.exp(-factor*torch.dot(f1_vec, f2_vec)))
        else:
            p = factor*torch.dot(f1_vec, f2_vec)
        return p

    def dot_(self, f1_vec, f2_vec):
        return -torch.dot(f1_vec, f2_vec)

    def forward(self, lm_logits, blank_logits, lm_labels, blank_labels, verbose=False):
        '''
        lm_logits: (batch_size, sequence_length, hidden_size)
        lm_labels: (batch_size, sequence_length, label_idxs)
        blank_logits: (batch_size, embeddings)
        blank_labels: (batch_size, 0 or 1)
        '''
        pos_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 1]
        neg_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 0]

        if len(pos_idxs) > 1:
            # positives
            pos_logits = []
            for pos1, pos2 in combinations(pos_idxs, 2):
                pos_logits.append(self.p_(blank_logits[pos1, :], blank_logits[pos2, :]))
            pos_logits = torch.stack(pos_logits, dim=0)
            pos_labels = [1.0 for _ in range(pos_logits.shape[0])]
        else:
            pos_logits, pos_labels = torch.FloatTensor([]), []
            if blank_logits.is_cuda:
                pos_logits = pos_logits.cuda()

        # negatives
        neg_logits = []
        for pos_idx in pos_idxs:
            for neg_idx in neg_idxs:
                neg_logits.append(self.p_(blank_logits[pos_idx, :], blank_logits[neg_idx, :]))
        neg_logits = torch.stack(neg_logits, dim=0)
        neg_labels = [0.0 for _ in range(neg_logits.shape[0])]

        blank_labels_ = torch.FloatTensor(pos_labels + neg_labels)

        if blank_logits.is_cuda:
            blank_labels_ = blank_labels_.cuda()

        lm_loss = self.LM_criterion(lm_logits, lm_labels)

        blank_loss = self.BCE_criterion(torch.cat([pos_logits, neg_logits], dim=0), \
                                        blank_labels_)

        if verbose:
            print("LM loss, blank_loss for last batch: %.5f, %.5f" % (lm_loss, blank_loss))

        total_loss = lm_loss + blank_loss
        return total_loss

class Mtb_Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=1, label2_pad_value=-1,\
                 label3_pad_value=-1, label4_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        self.label3_pad_value = label3_pad_value
        self.label4_pad_value = label4_pad_value

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

        labels3 = list(map(lambda x: x[3], sorted_batch))
        labels3_padded = pad_sequence(labels3, batch_first=True, padding_value=self.label3_pad_value)
        y3_lengths = torch.LongTensor([len(x) for x in labels3])

        labels4 = list(map(lambda x: x[4], sorted_batch))
        labels4_padded = pad_sequence(labels4, batch_first=True, padding_value=self.label4_pad_value)
        y4_lengths = torch.LongTensor([len(x) for x in labels4])
        return seqs_padded, labels_padded, labels2_padded, labels3_padded, labels4_padded,\
                x_lengths, y_lengths, y2_lengths, y3_lengths, y4_lengths

class pretrain_dataset(Dataset):
    def __init__(self, D, tokenizer, batch_size=None):
        self.internal_batching = True
        self.batch_size = batch_size # batch_size cannot be None if internal_batching == True
        self.alpha = 0.7
        self.mask_probability = 0.15

        self.df = pd.DataFrame(D, columns=['r','e1','e2'])
        self.e1s = list(self.df['e1'].unique())
        self.e2s = list(self.df['e2'].unique())

        self.tokenizer = tokenizer

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.E1_token_id = self.tokenizer.encode("[E1]")[1:-1][0]
        self.E1s_token_id = self.tokenizer.encode("[/E1]")[1:-1][0]
        self.E2_token_id = self.tokenizer.encode("[E2]")[1:-1][0]
        self.E2s_token_id = self.tokenizer.encode("[/E2]")[1:-1][0]
        self.PS = Mtb_Pad_Sequence(seq_pad_value=self.tokenizer.pad_token_id,\
                               label_pad_value=self.tokenizer.pad_token_id,\
                               label2_pad_value=-1,\
                               label3_pad_value=-1,\
                               label4_pad_value=-1)

    def put_blanks(self, D):
        blank_e1 = np.random.uniform()
        blank_e2 = np.random.uniform()
        if blank_e1 >= self.alpha:
            r, e1, e2 = D
            D = (r, "[BLANK]", e2)

        if blank_e2 >= self.alpha:
            r, e1, e2 = D
            D = (r, e1, "[BLANK]")
        return D

    def tokenize(self, D):
        (x, s1, s2), e1, e2 = D
        x = [w.lower() for w in x if x != '[BLANK]'] # we are using uncased model

        ### Include random masks for MLM training
        forbidden_idxs = [i for i in range(s1[0], s1[1])] + [i for i in range(s2[0], s2[1])]
        pool_idxs = [i for i in range(len(x)) if i not in forbidden_idxs]
        masked_idxs = np.random.choice(pool_idxs,\
                                        size=round(self.mask_probability*len(pool_idxs)),\
                                        replace=False)
        masked_for_pred = [token.lower() for idx, token in enumerate(x) if (idx in masked_idxs)]
        #masked_for_pred = [w.lower() for w in masked_for_pred] # we are using uncased model
        x = [token if (idx not in masked_idxs) else self.tokenizer.mask_token \
             for idx, token in enumerate(x)]

        ### replace x spans with '[BLANK]' if e is '[BLANK]'
        if (e1 == '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]' ,'[BLANK]', '[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]

        elif (e1 == '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]' ,'[BLANK]', '[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]

        elif (e1 != '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]'] + x[s1[0]:s1[1]] + ['[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]

        elif (e1 != '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]'] + x[s1[0]:s1[1]] + ['[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]

        e1_e2_start = ([i for i, e in enumerate(x) if e == '[E1]'][0],\
                        [i for i, e in enumerate(x) if e == '[E2]'][0])

        x = self.tokenizer.convert_tokens_to_ids(x)
        masked_for_pred = self.tokenizer.convert_tokens_to_ids(masked_for_pred)
        '''
        e1 = [e for idx, e in enumerate(x) if idx in [i for i in\
              range(x.index(self.E1_token_id) + 1, x.index(self.E1s_token_id))]]
        e2 = [e for idx, e in enumerate(x) if idx in [i for i in\
              range(x.index(self.E2_token_id) + 1, x.index(self.E2s_token_id))]]
        '''
        return x, masked_for_pred, e1_e2_start #, e1, e2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ### implements standard batching
        if not self.internal_batching:
            r, e1, e2 = self.df.iloc[idx]
            x, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks((r, e1, e2)))
            x = torch.tensor(x)
            masked_for_pred = torch.tensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            #e1, e2 = torch.tensor(e1), torch.tensor(e2)
            return x, masked_for_pred, e1_e2_start, e1, e2

        ### implements noise contrastive estimation
        else:
            ### get positive samples
            r, e1, e2 = self.df.iloc[idx] # positive sample
            pool = self.df[((self.df['e1'] == e1) & (self.df['e2'] == e2))].index
            pos_idxs = np.random.choice(pool, \
                                        size=min(int(self.batch_size//2), len(pool)), replace=False)
            ### get negative samples
            '''
            choose from option:
            1) sampling uniformly from all negatives
            2) sampling uniformly from negatives that share e1 or e2
            '''
            if np.random.uniform() > 0.5:
                pool = self.df[((self.df['e1'] != e1) | (self.df['e2'] != e2))].index
                neg_idxs = np.random.choice(pool, \
                                            size=min(int(self.batch_size//2), len(pool)), replace=False)
                Q = 1/len(pool)

            else:
                if np.random.uniform() > 0.5: # share e1 but not e2
                    pool = self.df[((self.df['e1'] == e1) & (self.df['e2'] != e2))].index
                    if len(pool) > 0:
                        neg_idxs = np.random.choice(pool, \
                                                    size=min(int(self.batch_size//2), len(pool)), replace=False)
                    else:
                        neg_idxs = []

                else: # share e2 but not e1
                    pool = self.df[((self.df['e1'] != e1) & (self.df['e2'] == e2))].index
                    if len(pool) > 0:
                        neg_idxs = np.random.choice(pool, \
                                                    size=min(int(self.batch_size//2), len(pool)), replace=False)
                    else:
                        neg_idxs = []

                if len(neg_idxs) == 0: # if empty, sample from all negatives
                    pool = self.df[((self.df['e1'] != e1) | (self.df['e2'] != e2))].index
                    neg_idxs = np.random.choice(pool, \
                                            size=min(int(self.batch_size//2), len(pool)), replace=False)
                Q = 1/len(pool)

            batch = []
            ## process positive sample
            pos_df = self.df.loc[pos_idxs]
            for idx, row in pos_df.iterrows():
                r, e1, e2 = row[0], row[1], row[2]
                x, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks((r, e1, e2)))
                x = torch.LongTensor(x)
                masked_for_pred = torch.LongTensor(masked_for_pred)
                e1_e2_start = torch.tensor(e1_e2_start)
                #e1, e2 = torch.tensor(e1), torch.tensor(e2)
                batch.append((x, masked_for_pred, e1_e2_start, torch.FloatTensor([1.0]),\
                              torch.LongTensor([1])))

            ## process negative samples
            negs_df = self.df.loc[neg_idxs]
            for idx, row in negs_df.iterrows():
                r, e1, e2 = row[0], row[1], row[2]
                x, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks((r, e1, e2)))
                x = torch.LongTensor(x)
                masked_for_pred = torch.LongTensor(masked_for_pred)
                e1_e2_start = torch.tensor(e1_e2_start)
                #e1, e2 = torch.tensor(e1), torch.tensor(e2)
                batch.append((x, masked_for_pred, e1_e2_start, torch.FloatTensor([Q]), torch.LongTensor([0])))
            batch = self.PS(batch)
            return batch