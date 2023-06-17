from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import copy
import random
import time
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from similarity import sim


def create_attention_mask_from_input_mask(batch_size, seq_length, to_mask):
    to_mask = to_mask[:, np.newaxis, :]
    broadcast_ones = np.ones([batch_size, seq_length, 1], dtype=np.float32)
    mask = broadcast_ones * to_mask
    return mask

def rfunc(triple_list, ent_num, rel_num):
    head = dict()  # head of each relation
    tail = dict()  # tail of each relation
    rel_count = dict()  # count of each relation
    r_mat_ind = list()
    r_mat_val = list()
    head_r = np.zeros((ent_num, rel_num))
    tail_r = np.zeros((ent_num, rel_num))
    for triple in triple_list:
        head_r[triple[0]][triple[1]] = 1
        tail_r[triple[2]][triple[1]] = 1
        r_mat_ind.append([triple[0], triple[2]])
        r_mat_val.append(triple[1])
        if triple[1] not in rel_count:
            rel_count[triple[1]] = 1
            head[triple[1]] = set()
            tail[triple[1]] = set()
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
        else:
            rel_count[triple[1]] += 1
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
    r_mat = tf.SparseTensor(indices=r_mat_ind, values=r_mat_val, dense_shape=[ent_num, ent_num])

    return head, tail, head_r, tail_r, r_mat


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pos_pair, neg_pair):
        # Calculate the contrastive loss between the positive and negative pairs
        pos_loss = torch.mean((pos_pair[0] - pos_pair[1]).pow(2))
        neg_loss = torch.mean(F.relu(self.margin - (neg_pair[0] - neg_pair[1]).pow(2)).pow(2))
        loss = pos_loss + neg_loss
        return loss

class RelationPredictionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dense = nn.Linear(768, num_classes)  # 768 is the size of ClinicalBERT's hidden state

    def forward(self, x):
        return self.dense(x)

class LMModel(nn.Module):
    def __init__(self, ent_num, options, embedding_dim, transformer_encoder_layers,base_model, relation_prediction_head):
        super().__init__()
        self.base_model = base_model
        self.relation_prediction_head = relation_prediction_head
        self._options = options
        self._ent_num = ent_num
        self.ent_embeds = nn.Embedding(ent_num, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=transformer_encoder_layers), 
            num_layers=transformer_encoder_layers)
    
    def getSubclass(file):
        subclass = []
        onto = get_ontology(file)
        onto.load()
    
        classes = []
        excl = ['Thing','DbXref','Definition','ObsoleteClass','Subset','Synonym','SynonymType']

        for cl in onto.classes():
        #if cl.name not in excl:
            classes.append(cl)

        classes = list(set(classes))

        for i in range (len(classes)):
            clss = classes[i].name
            for j in range(len(classes[i].is_a)):
                try:
                    if classes[i].is_a[j].name not in excl:
                        subclass.append((clss,classes[i].is_a[j].name))
                except AttributeError:
                    pass
        return subclass

    def getDijclass(file):
        dijclass = []
        onto = get_ontology(file)
        onto.load()
        classes = []
    #excl = ['DbXref','Definition','ObsoleteClass','Subset','Synonym','SynonymType']

        for cl in onto.classes():
        #if cl.name not in excl:
            classes.append(cl.name)

        classes = list(set(classes))

        try:
            for clss in onto.classes():
                for d in clss.disjoints():
                    if clss.name != d.entities[0].name:
                        dijclass.append((clss.name,d.entities[0].name))
        except AttributeError:
            pass
        return dijclass

    def getTriples(file):
        triples = []
        djw_list = {}
        subclass = getSubclass(file)
        dijclass = getDijclass(file)
        classes,labels = read_ontology(file)

    #excl = ['Thing','DbXref','Definition','ObsoleteClass','Subset','Synonym','SynonymType']

        for sub in subclass:
            triples.append([sub[0],'subclass of',sub[1]])

        for dij in dijclass:
            triples.append([dij[0],'disjoint with', dij[1]])

    
        return triples
    
    def path_encode(self, length=15, reuse=False):
        batch_size = self._options['batch_size']
        ent_seq_len = (length + 1) // 2

        seq = torch.empty((batch_size, ent_seq_len), dtype=torch.int64)
        label = torch.empty((batch_size, self._ent_num))
        attention_mask = torch.empty((batch_size, ent_seq_len, ent_seq_len))
        mask_position = torch.empty((), dtype=torch.int64)

        path_em = self.path_embeds(seq.long())

        outputs = self.transformer_encoder(path_em)

        mask_em = outputs[:, mask_position, :]
        seq_loss = F.binary_cross_entropy_with_logits(mask_em, label)

        return seq_loss, seq, label, mask_position, attention_mask
    
    def triple_train(self, triples):
        start = time.time()
        epoch_loss = 0
        num_batch = len(triples) // self._options.triple_batch_size
        for i in range(num_batch):
            batch_pos = random.sample(triples, self._options.batch_size)
            batch_loss, _ = self.session.run(fetches=[self._triple_loss, self._triple_train_op],
                                             feed_dict={self.pos_hs: [x[0] for x in batch_pos],
                                                        self.pos_rs: [x[1] for x in batch_pos],
                                                        self.pos_ts: [x[2] for x in batch_pos]})
            epoch_loss += batch_loss
        epoch_loss /= len(triples)
        return epoch_loss, time.time() - start

    def head_tail(self, h_i, h_j, t_i, t_j, hi_num, hj_num, ti_num, tj_num):
        h_sim = sim(h_i, h_j, metric='cosine')
        thre_h = min(self.args.thre_h, (np.max(h_sim) + np.mean(h_sim)) / 2)
        t_sim = sim(t_i, t_j, metric='cosine')
        thre_t = min(self.args.thre_t, (np.max(t_sim) + np.mean(t_sim)) / 2)
        h_sim = h_sim.max(axis=0 if h_sim.shape[0] > h_sim.shape[1] else 1)
        h_sim = h_sim * (h_sim > thre_h)  # todo: filter by threshold or not
        t_sim = t_sim.max(axis=0 if t_sim.shape[0] > t_sim.shape[1] else 1)
        t_sim = t_sim * (t_sim > thre_t)
        numerator = sum(h_sim) + sum(t_sim)
        denominator = hi_num + hj_num - sum(h_sim) + ti_num + tj_num - sum(t_sim)
        return numerator / denominator

    def forward(self, input):
        base_model_output = self.base_model(**input)
        logits = base_model_output.logits

        # Use the hidden state corresponding to the [SEP] token for relation prediction
        sep_hidden_state = base_model_output.last_hidden_state[:, input["input_ids"][0].tolist().index(tokenizer.sep_token_id)]
        relation_logits = self.relation_prediction_head(sep_hidden_state)

        return logits, relation_logits

class LMDataset(Dataset):
    def __init__(self, mlm_samples, rp_samples, tokenizer):
        self.mlm_samples = mlm_samples
        self.rp_samples = rp_samples
        self.tokenizer = tokenizer

    def __len__(self):
        return max(len(self.mlm_samples), len(self.rp_samples))

    def __getitem__(self, idx):
        mlm_sample1, mlm_sample2 = self.mlm_samples[idx % len(self.mlm_samples)]  # Loop over the samples if one task has fewer samples
        rp_sample, rp_label = self.rp_samples[idx % len(self.rp_samples)]
        #rp_sample, rp_label = self.rp_samples[idx][0],self.rp_samples[idx][1]
        #print(rp_samples[idx])

        mlm_sample1 = f"{tokenizer.cls_token} {mlm_sample1} {tokenizer.sep_token}"
        mlm_sample2 = f"{tokenizer.cls_token} {mlm_sample2} {tokenizer.sep_token}"

        mlm_encoding1 = self.tokenizer(mlm_sample1, return_tensors='pt', padding=True, truncation=True, max_length=512)
        mlm_encoding2 = self.tokenizer(mlm_sample2, return_tensors='pt', padding=True, truncation=True, max_length=512)
        rp_encoding = self.tokenizer(rp_sample, return_tensors='pt', padding=True, truncation=True, max_length=512)

        return mlm_encoding1, mlm_encoding2, rp_encoding, torch.tensor(rp_label, dtype=torch.long)


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

dataset = LMDataset(mlm_samples, rp_samples, tokenizer)
dataloader = DataLoader(dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = ContrastiveLoss()

base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
relation_prediction_head = RelationPredictionHead(num_classes=2)  # Adjust the number of classes
model = LMModel(base_model, relation_prediction_head)

# You might want to use a different optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Set up a criterion for the relation prediction task
rp_criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(10):  # example epoch number
    for mlm_encoding1, mlm_encoding2, rp_encoding, rp_label in dataloader:
        # Move to device
        mlm_encoding1 = {key: tensor.to(device) for key, tensor in mlm_encoding1.items()}
        mlm_encoding2 = {key: tensor.to(device) for key, tensor in mlm_encoding2.items()}
        rp_encoding = {key: tensor.to(device) for key, tensor in rp_encoding.items()}
        rp_label = rp_label.to(device)

        print(mlm_encoding1)

        mlm_output1, _ = model(mlm_encoding1)
        mlm_output2, _ = model(mlm_encoding2)
        _, rp_output = model(rp_encoding)

        mlm_loss = criterion(mlm_output1, mlm_output2)
        rp_loss = rp_criterion(rp_output, rp_label)

        # Combine the losses. You might want to use different weights for the losses depending on their scale and importance
        loss = mlm_loss + rp_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch: {epoch+1}, MLM Loss: {mlm_loss.item()}, RP Loss: {rp_loss.item()}")
