# coding=utf-8

import re
import torch
from itertools import permutations, combinations

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

def process_lines(text):
    text = [process_sent(sent) for sent in text]
    text = " ".join([t for t in text if t is not None])
    text = re.sub(' {2,}', ' ', text) # remove extra spaces > 1
    return text

def label_sent(sent, ent1, ent2):
    labeled_sent = str(sent)
    labeled_sent = labeled_sent.replace(str(ent1), "[E1]" + str(ent1) + "[/E1]")
    labeled_sent = labeled_sent.replace(str(ent2), "[E2]" + str(ent2) + "[/E2]")
    return labeled_sent

def find_entities(sent):
    ents = sent.ents
    pairs = []
    if len(ents) > 1:
        for a, b in permutations([ent for ent in ents], 2):
            pairs.append((a,b))

    labeled_sents = []

    for pair in pairs:
        labeled_sents.append((label_sent(sent, pair[0], pair[1]),pair[0],pair[1]))
    return labeled_sents

class infer_from_trained(object):
    def __init__(self, tokenizer, model, rm):
        self.cuda = torch.cuda.is_available()

        self.net = model
        self.tokenizer = tokenizer
        self.rm = rm
        self.net.resize_token_embeddings(len(self.tokenizer))
        if self.cuda:
            self.net.cuda()

        self.e1_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        self.e2_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        self.pad_id = self.tokenizer.pad_token_id

    def get_e1e2_start(self, x):
        e1_e2_start = ([i for i, e in enumerate(x) if e == self.e1_id][0],\
                        [i for i, e in enumerate(x) if e == self.e2_id][0])
        return e1_e2_start

    def infer_sentence(self, sentence):
        self.net.eval()
        tokenized = self.tokenizer.encode(sentence); #print(tokenized)
        e1_e2_start = self.get_e1e2_start(tokenized); #print(e1_e2_start)
        tokenized = torch.LongTensor(tokenized).unsqueeze(0)
        e1_e2_start = torch.LongTensor(e1_e2_start).unsqueeze(0)
        attention_mask = (tokenized != self.pad_id).float()
        token_type_ids = torch.zeros((tokenized.shape[0], tokenized.shape[1])).long()

        if self.cuda:
            tokenized = tokenized.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

        classification_logits = self.net(tokenized, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                                    e1_e2_start=e1_e2_start)
        predicted = torch.softmax(classification_logits, dim=1).max(1)[1].item()
        #print("Sentence: ", sentence)
        #print("Predicted: ", self.rm.idx2rel[predicted].strip(), '\n')
        return predicted