import logging
import sys
from AlignmentFormat import serialize_mapping_to_tmp_file
from collections import defaultdict
import numpy as np
import random
import json

import os
#from rdflib import Graph, URIRef, RDFS
from bs4 import BeautifulSoup
from owlready2 import onto_path, get_ontology 
import io
from time import sleep
import itertools
from tqdm import tqdm

import urllib.request
from urllib.request import Request, urlopen

import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

from langdetect import detect, detect_langs


def getTrainfileS(source_graph):
    
    classes_source,labels_source = read_ontology(source_graph)
    labels_source_syn = getSyn(source_graph)
    triples_source = getTriples(source_graph)

    entity2id_file = open("ent_ids_source.txt", "w")
    for entity , id in classes_source.items():
        entity = entity.encode('utf8').decode('utf-8')
        entity2id_file.write(str(id) + '\t' + str(entity)+ '\n')
    entity2id_file.close()

    label2id_file = open("label_ids_source.txt", "w")
    for label , id in labels_source_syn.items():
        label = label.encode('utf8').decode('utf-8')
        label2id_file.write(str(id) + '\t' + str(label)+ '\n')
    label2id_file.close()
    
    train2id_file = open("triples_source.txt", "w")
    for triple in triples_source:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[2]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()


def getTrainfileT(target_graph):
    
    classes_target,labels_target = read_ontology(target_graph)
    labels_target_syn = getSyn(target_graph)
    triples_target = getTriples(target_graph)

    entity2id_file = open("ent_ids_target.txt", "w")
    for entity , id in classes_target.items():
        entity = entity.encode('utf8').decode('utf-8')
        entity2id_file.write(str(id) + '\t' + str(entity)+ '\n')
    entity2id_file.close()

    label2id_file = open("label_ids_target.txt", "w")
    for label , id in labels_target_syn.items():
        label = label.encode('utf8').decode('utf-8')
        label2id_file.write(str(id) + '\t' + str(label)+ '\n')
    label2id_file.close()

    train2id_file = open("triples_target.txt", "w")
    for triple in triples_target:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[2]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()

def getTrainList(source_graph,target_graph):
    train_list = []
    labels_source_syn = getSyn(source_graph)
    labels_target_syn = getSyn(target_graph)

    if len(labels_source_syn)==0 or len(labels_target_syn)==0:
        labels_source_syn,label_list = read_ontology(source_graph)
        labels_target_syn,label_list = read_ontology(target_graph)

    for k in sorted(labels_source_syn.keys() & labels_target_syn.keys()):
        train_list.append([labels_source_syn[k],labels_target_syn[k]])

    label_list =[]
    labels = list(itertools.product(list(labels_source_syn.keys()[:6000]), list(labels_target_syn.keys()[:6000])))
    for a, b in tqdm(labels):
        label_list.append([a,b])

    for i in tqdm(range(len(label_list))):
        strings1 = label_list[i][0]
        string1 = strings1.replace('_',' ')
        strings2 = label_list[i][1]
        string2 = strings2.replace('_',' ')
        fuzzs = fuzz.token_sort_ratio(string1,string2)
        a  = labels_source_syn.get(str(strings1))
        b  = labels_target_syn.get(str(strings2))
        if fuzzs >= 97:
            if [a,b] not in train_list:
                train_list.append([a,b])
        # if len(train_list) <=10:
        #     if fuzzsp >=95:
        #         if [a,b] not in train_list:
        #             train_list.append([a,b])
        else:
            pass


    return train_list

def getFile(source_url,target_url):
    getTrainfileS(source_url)
    getTrainfileT(target_url)

    train_list = getTrainList(source_url,target_url)
    train2id_file = open("train.txt", "w")
    for triple in train_list:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()

    #generate negative samples
    negitive_sampling_constrain_source = {}
    sbpt_source = {}
    dis_source = {}
    f_source = open('triples_source.txt', 'r')
    train2id_source = []

    for line in f_source.readlines()[1:]:
        train2id_source.append(line)
    f_source.close()

    for line in train2id_source:
        train2id = line.strip('\n').split('\t')
        leftent, rightent, rel = int(train2id[0]), int(train2id[1]), int(train2id[2])
        #if "subClassOf" in rel_source:
        if rel == 1:
            if str(leftent) not in sbpt_source.keys():
                sbpt_source[str(leftent)] = []
            if rightent not in sbpt_source[str(leftent)]:
                sbpt_source[str(leftent)].append(rightent)
            if str(rightent) not in sbpt_source.keys():
                sbpt_source[str(rightent)] = []
            if leftent not in sbpt_source[str(rightent)]:
                sbpt_source[str(rightent)].append(leftent)
            else:
                pass
        else:
            pass

        #if "disjointWith" in rel_source:
        if rel == 2:
            if str(leftent) not in dis_source.keys():
                dis_source[str(leftent)] = []
            if rightent not in dis_source[str(leftent)]:
                dis_source[str(leftent)].append(rightent)
            if str(rightent) not in dis_source.keys():
                dis_source[str(rightent)] = []
            if leftent not in dis_source[str(rightent)]:
                dis_source[str(rightent)].append(leftent)
            else:
                pass
        else:
            pass
    negitive_sampling_constrain_source['sbc'] = sbpt_source
    negitive_sampling_constrain_source['dij'] = dis_source

    f = open('neg_constrain_source.json', 'w')
    json.dump(negitive_sampling_constrain_source, f)
    f.close()

    negitive_sampling_constrain_target = {}
    sbpt_target = {}
    dis_target = {}
    f_target = open('triples_target.txt', 'r')
    train2id_target = []
    for line in f_target.readlines()[1:]:
        train2id_target.append(line)
    f_target.close()
    for line in train2id_target:
        train2id = line.strip('\n').split('\t')
        leftent, rightent, rel = int(train2id[0]), int(train2id[1]), int(train2id[2])
        #if "subClassOf" in rel_source:
        if rel == 1:
            if str(leftent) not in sbpt_target.keys():
                sbpt_target[str(leftent)] = []
            if rightent not in sbpt_target[str(leftent)]:
                sbpt_target[str(leftent)].append(rightent)
            if str(rightent) not in sbpt_target.keys():
                sbpt_target[str(rightent)] = []
            if leftent not in sbpt_target[str(rightent)]:
                sbpt_target[str(rightent)].append(leftent)
            else:
                pass
        else:
            pass

        #if "disjointWith" in rel_target:
        if rel == 2:
            if str(leftent) not in dis_target.keys():
                dis_target[str(leftent)] = []
            if rightent not in dis_target[str(leftent)]:
                dis_target[str(leftent)].append(rightent)
            if str(rightent) not in dis_target.keys():
                dis_target[str(rightent)] = []
            if leftent not in dis_target[str(rightent)]:
                dis_target[str(rightent)].append(leftent)
            else:
                pass
        else:
            pass
    negitive_sampling_constrain_target['sbc'] = sbpt_target
    negitive_sampling_constrain_target['dij'] = dis_target

    f = open('neg_constrain_target.json', 'w')
    json.dump(negitive_sampling_constrain_target, f)
    f.close()

def embedding_run():
    #Getting translation based embeddings
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    con = Config()
    con.inPath("")
    con.trainTimes(1000)
    con.setBatches(100)
    con.learningRate(0.01)
    con.entDimension(100)
    con.negRate(10)
    con.negativeSampling('unif')
    con.optMethod("SGD")
    con.exportFiles("amd.model.tf", 0)
    con.vecFiles("amd.embedding.vec.json")
    con.model_name("trans")
    con.init()
    con.model(Trans)
    con.run()


def getID():    
    with open('ent_ids_source.txt') as f:
        ent_ids_source = {line.strip().split('\t')[0]: int(line.strip().split('\t')[0]) for line in f.readlines()}
    with open('ent_ids_target.txt') as f:
        ent_ids_target = {line.strip().split('\t')[0]: int(line.strip().split('\t')[0]) for line in f.readlines()}

    f = open("amd.embedding.vec.json", "r")
    embedding = json.load(f)
    f.close()
    source_vecs = embedding["source_ent_embeddings"][:len(ent_ids_source)+1]
    target_vecs = embedding["target_ent_embeddings"][:len(ent_ids_target)+1]

    return ent_ids_source,ent_ids_target,source_vecs,target_vecs

def getList():
    source_list = []
    target_list = []
    with open('ent_ids_source.txt') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            entid = int(params[0])
            ent = str(params[1])
            source_list.append(ent)
        f.close()

    with open('ent_ids_target.txt') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            entid = int(params[0])
            ent = str(params[1])
            target_list.append(ent)
        f.close()

    return source_list,target_list

def getAligns(soure_list, target_list, threshold):
    sim_dict = {}
    vec_alignments = []
    ent_ids_source,ent_ids_target,source_vecs,target_vecs = getID()
    for ent1 in tqdm(soure_list):
        source_vec = source_vecs[soure_list.index(ent1)]
        for ent2 in target_list:
            target_vec = target_vecs[target_list.index(ent2)]
            simility = np.dot(target_vec, source_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(source_vec))
            sim_dict[ent1 + '\t' + ent2] = simility
            if simility >= threshold: 
                vec_alignments.append((ent1, ent2))
    return sim_dict, vec_alignments


def alignmentMatch(source_url, target_url):

    urllib.request.urlretrieve(source_url, "source.owl")
    urllib.request.urlretrieve(target_url, "target.owl")
    source_file = "source.owl"
    target_file = "target.owl"

    onto1 = get_ontology(source_file)
    onto1.load()
    base1 = onto1.base_iri

    onto2 = get_ontology(target_file)
    onto2.load()
    base2 = onto2.base_iri

    relation = '='
    alignments = []

    sentences1 = []
    sourceid = []
    sentences2 = []
    targetid = []

    threshold = 0.924

    entity_syn_source,syn_source = getSyn(source_file)
    entity_syn_target,syn_target = getSyn(target_file)

    for key1,val1 in syn_source.items():
        sentences1.append(key1)
        sourceid.append(val1)

    for key2,val2 in syn_target.items():
        sentences2.append(key2)
        targetid.append(val2)

    model_path = "finetinued.pt"
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    model = BertModel.from_pretrained(model_path)
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1,embeddings2)

    values,indices = torch.max(cosine_scores,1)

    for i in range(len(sentences1)):
        if cosine_scores[i][indices[i]].item() >= threshold:
            a = base1+sourceid[i]
            b = base2+targetid[indices[i]]
            score = round(cosine_scores[i][indices[i]].item(),3)
            if (a,b,relation,score) not in alignments: 
                alignments.append((a,b,relation,score))

    return alignments
