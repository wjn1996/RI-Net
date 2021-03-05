#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:48:55 2019

@author: Majigsuren Enkhsaikhan
"""

import os
import pandas as pd
import numpy as np
import csv
import json
from tqdm import tqdm
import re

from triples_from_text import extract_triples, tagger
# from .triples_from_text import extract_triples

def align(triples, tokens, corefs, paragraph_text):
    # 根据三元组抽取结果、消解列表以及分词情况，每个token要与三元组进行对应

    def search_span(entity, paragraph_text_):
        _in_text = []
        while (True):
            d = paragraph_text_.find(entity)
            if d == -1 or len(paragraph_text_) == 0:
                break
            _in_text.append((d, d + len(entity) - 1))
            paragraph_text_ = paragraph_text_[d + len(entity):]
        return _in_text

    corefs_dict = dict() # 每个指代消解
    token2triple = dict() # 每个token对应的三元组 # key:tokenid, value:[tripleid, 0/1/2]

    for token in tokens:
        token2triple[token[0]] = [[-1, -1]]
    num=0

    for key, value in corefs:
        if value in ['a', 'an', 'the', 'its', 'their', 'his', 'her', 'our', 'all', 'old', 'new', 'latest', 'who', 'that', 'this', 'these', 'those']:
            continue
        corefs_dict[key] = value
    for triple_id, (head, relation, tail) in enumerate(triples): # 对每个三元组，将其与文本进行对齐
        # 头实体和尾实体在原文中的所有索引位置
        head_in_text, tail_in_text = [], []
        # head_in_text = [(i.start(), i.end()) for i in re.finditer(r'' + head, r'' + paragraph_text)]
        # tail_in_text = [(i.start(), i.end()) for i in re.finditer(r'' + tail, r'' + paragraph_text)]

        head_in_text = search_span(head, paragraph_text)
        tail_in_text = search_span(tail, paragraph_text)

        # 将头尾实体在原文中最近的两个（head在tail前且两者距离最近的作为一个三元组与原文对齐的span）
        if not head_in_text or not tail_in_text:
            num+=1
            continue
        triple_char_span = tuple()
        for (hs, he) in reversed(head_in_text):
            for (ts, te) in tail_in_text:
                if he < ts: # 首次出现head在tail的前面，则这个区间可以作为这个三元组在原文的span
                    triple_char_span = (hs, te) # 字符级别上的span
                    break
        if not triple_char_span:
            num += 1
            continue
        # 根据字符级别的span找到对应的token级别的span
        triple_token_span = tuple()
        triple_token_start, triple_token_end = None, None
        start_has = False
        for i in tokens:
            if not start_has and triple_char_span[0] >= i[7] and triple_char_span[0] <= i[8]:
                triple_token_start = i[0]
                start_has = True
                continue
            if start_has and triple_char_span[1] >= i[7] and triple_char_span[1] <= i[8]:
                triple_token_end = i[0]
                break
        if not triple_token_start or not triple_token_end:
            continue
        triple_token_span = (triple_token_start, triple_token_end)
        # head_tokens, relation_tokens, tail_tokens = head.split(' '), relation.split(' '), tail.split(' ')
        for token_i in range(triple_token_span[0], triple_token_span[1] + 1):
            if tokens[token_i][1] in head: # 当前token在三元组的头实体部分
                if token2triple[token_i] == [[-1, -1]]:
                    token2triple[token_i] = [[triple_id, 0]]
                else:
                    token2triple[token_i].append([triple_id, 0])
            elif tokens[token_i][1] in relation: # 当前三元组在关系边上
                if token2triple[token_i] == [[-1, -1]]:
                    token2triple[token_i] = [[triple_id, 1]]
                else:
                    token2triple[token_i].append([triple_id, 1])
            elif tokens[token_i][1] in tail: # 当前token在三元组尾实体
                if token2triple[token_i] == [[-1, -1]]:
                    token2triple[token_i] = [[triple_id, 2]]
                else:
                    token2triple[token_i].append([triple_id, 2])
        # print()
    # 对一些存在指代消解的token，单独进行处理。对所有指代消解的token，直接将指代的实体在所有三元组内查找，并确定其范围
    # for token in tokens:
    #     if token[1] in corefs_dict.keys(): # 如果当前的token是代词，则将其对应指代的词作为与三元组对齐的目标；
    #         target = corefs_dict[token]
    #     else:
    #         target = token
    for key, value in corefs_dict.items(): # 遍历每个指代消解
        # 对于指代消解类的token，启发式地认为只要其指代的词出现在任意一个三元组的头实体或尾实体，则其将对应于对应的三元组
        for ei, (head, _, tail) in enumerate(triples):
            p = 0 if value in head else -1
            p = 2 if value in tail else -1
            if p != -1:
                for token in tokens:
                    if token[1] == key:
                        if token2triple[token[0]] == [[-1, -1]]:
                            token2triple[token[0]] =[[ei, p]]
                        else:
                            token2triple[token[0]].append([ei, p])
    # print(token2triple)
    # print()
    return token2triple


def squad_extractor(train_file, dev_file):
    # 读取SQuAD2.0数据集，并将每个passage转换为knowledge graph，
    # 将knowledge graph中每个结点对应到原始passage的start和end位置，并对每个实体和关系边以及三元组进行编号；
    # 对原始的每个token，生成与knowledge graph对应的map，没有出现在graph的token则做标记；
    # 最终为每个图添加一个额外的结点[CLS]编号为0；
    files_dict = {"dev": dev_file}
    ## 读取SQuAD数据集
    for data_kind, data_file in files_dict.items():
        print("reading {} files ... ".format(data_kind))
        with open(data_file, "r", encoding='utf-8') as reader:
            source = json.load(reader)
            input_data = source["data"]
            version = source["version"]
        structure_knowledge = dict()
        record = dict()
        print("processing {} files ... ".format(data_kind))
        for entry in tqdm(input_data, ncols=50, desc="reading examples:"):
            title = entry["title"]
            if title not in structure_knowledge.keys():
                structure_knowledge[title] = []
            for paragraph in tqdm(entry["paragraphs"]):
                paragraph_text = paragraph["context"]
                triples, df_p_tagged, passage_tokens, corefs = extract_triples(paragraph_text) # 对当前的passage进行信息抽取
                passagetoken2triple = align(triples, passage_tokens, corefs, paragraph_text) # 将passage的每个token与三元组对齐
                record["passagetoken2triple"] = passagetoken2triple
                record["passage_tokens"] = passage_tokens
                record["triples"] = triples
                record["corefs"] = corefs
                structure_knowledge[title].append(record)
                record = dict()
                # for questions in paragraph["qas"]:
                #     question_text = questions["question"]
                #     question_id = questions["id"]
                #     df_q_tagged, _, question_tokens = tagger(question_text)
                #     # questiontoken2triple = align(triples, question_tokens, corefs, question_text) # 将passage的每个token与三元组对齐
                #     record["passagetoken2triple"] = passagetoken2triple
                #     record["passage_tokens"] = passage_tokens
                #     # record["questiontoken2triple"] = questiontoken2triple
                #     record["question_tokens"] = question_tokens
                #     record["triples"] = triples
                #     record["corefs"] = corefs
                #     structure_knowledge[question_id] = record
                #     record = dict()
        np.save("../../data/squad/squad_{}_structure_knowledge.npy".format(data_kind), structure_knowledge)


def newsqa_extractor(data):
    import multiprocessing
    # 读取newsqa数据集，该数据集最终为一个json文件，里面保存了train、dev和test三种类型数据集
    # 数据格式（newsqa数据格式基本仿照SQuAD制作的）
    '''
        {
            "data": [
                {
                    "storyId": "./contoso/stories/2e1d4",
                    "text": "Hyrule (Contoso) -- Tingle, Tingle! Kooloo-Limpah! ...These are the magic words that Tingle created himself. Don't steal them!",
                    "type": "train",
                    "questions": [
                        {
                            "q": "What should you not do with Tingle's magic words?",
                            "consensus": {
                                "s": 115,
                                "e": 125
                            },
                            "isAnswerAbsent": 0.25,
                            "isQuestionBad": 0.25,
                            "answers": [
                                {
                                    "sourcerAnswers": [
                                        {
                                            "s": 115,
                                            "e": 125
                                        }
                                    ]
                                },
                                {
                                    "sourcerAnswers": [
                                        {
                                            "noAnswer": true
                                        }
                                    ]
                                }
                            ],
                            "validatedAnswers": [
                                {
                                    "s": 115,
                                    "e": 125,
                                    "count": 2
                                },
                                {
                                    "noAnswer": true,
                                    "count": 1
                                },
                                {
                                    "badQuestion": true,
                                    "count": 1
                                }
                            ]
                        }
                    ]
                }
            ],
            "version": "1"
        }

    '''

    with open(data_file, "r", encoding='utf-8') as reader:
        source = json.load(reader)
        input_data = source["data"]
        version = source["version"]


    print("processing {} files ... ".format(data_kind))
    procs = []
    n = len(input_data)
    n_cpu = multiprocessing.cpu_count()
    chunk_size = int(n / n_cpu)

    for i in range(0, n_cpu):
        min_i = chunk_size * i

        if i < n_cpu - 1:
            max_i = chunk_size * (i + 1)
        else:
            max_i = n
        digits = [input_data[min_i:max_i], str(i)]
        procs.append(multiprocessing.Process(target=extractor, args=(digits, "parallel")))

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


def extractor(digits, fold="1by1"):
    record = dict()
    input_data = digits[0]
    cupid = digits[1]
    train_structure_knowledge = dict()
    dev_structure_knowledge = dict()
    test_structure_knowledge = dict()

    for entry in tqdm(input_data, ncols=50, desc="reading examples:"):
        storyId = entry["storyId"]
        paragraph_text = entry["text"] # 文章文本内容
        type = entry["type"] # 样本类型（train/dev/test）
        triples, df_p_tagged, passage_tokens, corefs = extract_triples(paragraph_text)  # 对当前的passage进行信息抽取
        passagetoken2triple = align(triples, passage_tokens, corefs, paragraph_text)  # 将passage的每个token与三元组对齐
        record["passagetoken2triple"] = passagetoken2triple
        record["passage_tokens"] = passage_tokens
        record["triples"] = triples
        record["corefs"] = corefs
        if type == "train":
            if storyId not in train_structure_knowledge.keys():
                train_structure_knowledge[storyId] = []
            train_structure_knowledge[storyId].append(record)
        if type == "dev":
            if storyId not in dev_structure_knowledge.keys():
                dev_structure_knowledge[storyId] = []
            dev_structure_knowledge[storyId].append(record)
        if type == "test":
            if storyId not in test_structure_knowledge.keys():
                test_structure_knowledge[storyId] = []
            test_structure_knowledge[storyId].append(record)
        record = dict()

    np.save("../../data/newsqa/newsqa_train_structure_knowledge_{}.npy".format(cupid), train_structure_knowledge)
    np.save("../../data/newsqa/newsqa_dev_structure_knowledge_{}.npy".format(cupid), dev_structure_knowledge)
    np.save("../../data/newsqa/newsqa_test_structure_knowledge_{}.npy".format(cupid), test_structure_knowledge)


# Reads data file and creates the submission.csv file
if __name__ == "__main__":
    print("Start the structure knowledge auto-extractor process.")
    data_kind = 'newsqa'
    if data_kind == 'squad':
        train_file = '../../data/squad/train-v2.0.json'
        dev_file = '../../data/squad/dev-v2.0.json'
        squad_extractor(train_file, dev_file)
        print("Finished the process.")
    if data_kind == 'newsqa':
        data_file = '../../data/newsqa/combined-newsqa-data-v1.json'
        newsqa_extractor(data_file)


# Walter Otto Davis was a Welsh professional footballer who played at centre forward for Millwall for ten years in the 1910s. Millwall Football Club is a professional football club in South East London, which was Founded as Millwall Rovers in 1885.