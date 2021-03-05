import numpy as np
import os
import torch
import copy
import collections
from configure import args
from tqdm import tqdm
import json
import spacy
import pickle
from random import sample, randint
from utils.utils import get_equal_rate, int_list_to_str, str_to_int_list, find_Rel_Path
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer

import Levenshtein
import networkx as nx
import matplotlib
matplotlib.use('AGG')


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id=None,
                 passage_text=None,
                 passage_tokens=None,
                 passagetoken2triple=None,
                 question_text=None,
                 question_tokens=None,
                 questiontoken2triple=None,
                 refine_triples=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 ):
        self.qas_id = qas_id
        self.passage_text = passage_text
        self.passage_tokens = passage_tokens  # passage的token，例子：[11, 'governmental', '', 'O', 'governmental', 'ADJ', 'JJ', 56, 67, 'amod']
        self.passagetoken2triple = passagetoken2triple
        self.question_text = question_text
        self.question_tokens = question_tokens
        self.questiontoken2triple = questiontoken2triple
        self.refine_triples = refine_triples
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position # ground truth的token级别开始位置
        self.end_position = end_position # ground truth的token级别终止位置
        self.is_impossible = is_impossible # 答案是否存在

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join([i[1] for i in self.passage_tokens]))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class NewsqaExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """
    # 关于NewsQA的预处理详见https://github.com/Maluuba/newsqa

    def __init__(self,
                 qas_id=None,
                 passage_text=None,
                 passage_tokens=None,
                 passagetoken2triple=None,
                 question_text=None,
                 question_tokens=None,
                 questiontoken2triple=None,
                 refine_triples=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 ):
        self.qas_id = qas_id
        self.passage_text = passage_text
        self.passage_tokens = passage_tokens  # passage的token，例子：[11, 'governmental', '', 'O', 'governmental', 'ADJ', 'JJ', 56, 67, 'amod']
        self.passagetoken2triple = passagetoken2triple
        self.question_text = question_text
        self.question_tokens = question_tokens
        self.questiontoken2triple = questiontoken2triple
        self.refine_triples = refine_triples
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position # ground truth的token级别开始位置
        self.end_position = end_position # ground truth的token级别终止位置
        self.is_impossible = is_impossible # 答案是否存在

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join([i[1] for i in self.passage_tokens]))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""
    # __slots__ 可以用来降低对象占用的内存大小
    __slots__ = ['unique_id', 'example_index', 'doc_span_index', 'tokens', 'token_to_orig_map', 'token_is_max_context',
                 'input_ids', 'input_mask', 'segment_ids', 'start_position', 'end_position', 'is_impossible',
                 'bert_refine_triples_ids', 'bert_refine_triples_with_cls_ids', 'entity_ids', 'entpair2tripleid', 'tokens_to_triple']

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 # input_rel_path=None,
                 # input_ent_path=None,
                 bert_refine_triples_ids=None,
                 bert_refine_triples_with_cls_ids=None,
                 entity_ids=None,
                 entpair2tripleid=None,
                 tokens_to_triple=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        # self.input_rel_path = input_rel_path
        # self.input_ent_path = input_ent_path
        self.bert_refine_triples_ids = bert_refine_triples_ids,
        self.bert_refine_triples_with_cls_ids = bert_refine_triples_with_cls_ids,
        self.entity_ids = entity_ids,
        self.entpair2tripleid = entpair2tripleid,
        self.tokens_to_triple = tokens_to_triple



class ModelTrainInput(object):
    """A single set of features of data."""
    # __slots__ 可以用来降低对象占用的内存大小
    __slots__ = ['input_ids', 'input_mask', 'input_kg_rel_path', 'input_kg_rel_path_length', 'input_token_to_entity',
                 'input_path_mask', 'segment_ids', 'start_position', 'end_position', 'is_impossible']

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 input_kg_rel_path,
                 input_kg_rel_path_length,
                 input_token_to_entity,
                 input_path_mask,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_kg_rel_path = input_kg_rel_path
        self.input_kg_rel_path_length = input_kg_rel_path_length
        self.input_token_to_entity = input_token_to_entity
        self.input_path_mask = input_path_mask
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

class SquadLoader():
    # 用于加载squad数据集
    def __init__(self):
        self.train_file = args.train_file # 原始训练数据集
        self.predict_file = args.predict_file # 原始验证数据集
        # self.test_file = None # 原始测试数据集
        self.train_struct_file = os.path.join(args.squad_data_dir, 'squad_train_structure_knowledge.npy')
        self.dev_struct_file = os.path.join(args.squad_data_dir, 'squad_dev_structure_knowledge.npy')
        self.nlp = spacy.load("en_core_web_sm")

    def is_whitespace(self, c):  # 判断token是否是空格
        if c == " " or c == " or c ==\t" or c == "\r" "\n" or ord(c) == 0x202F:
            return True
        return False

    def char_to_token_span(self, text, tokens):
        # squad原文给定的answer的区间是字符级别的区间范围，现需要根据分词的情况转换为对应的token级别的区间
        char_to_word_offset = dict()
        for i in range(len(text)):
            char_to_word_offset[i] = -1

        for token in tokens:
            # tokens 样例：[11, 'governmental', '', 'O', 'governmental', 'ADJ', 'JJ', 56, 67, 'amod']
            for j in range(token[7], token[8]+1):
                char_to_word_offset[j] = token[0]
        return char_to_word_offset

    # 读取数据
    # 读取原始的SQuAD2.0数据集，加载每个passage对应的refining的结构化知识，并对对应的每个question进行分词，并与三元组对齐；
    def read_squad_examples(self, is_training=True):
        # simple_nlp = SimpleNlp()
        input_file = self.train_file if is_training == True else self.predict_file
        struct_file = self.train_struct_file if is_training == True else self.dev_struct_file
        """Read a SQuAD json file into a list of SquadExample."""
        ## 读取squad原始数据集
        with open(input_file, "r", encoding='utf-8') as reader:
            source = json.load(reader)
            input_data = source["data"]
            version = source["version"]
        input_tag_data = []
        # 读取抽取的结构化知识
        input_struct_data = np.load(struct_file, allow_pickle=True)[()]


        examples = []
        for entry in tqdm(input_data, ncols=50, desc="reading examples"):
            title = entry['title']
            title_struct_data = input_struct_data[title] # list 保存每个paragraph对应的抽取的结构知识
            paragraphs = entry["paragraphs"]
            assert len(paragraphs) == len(title_struct_data) # 原始数据集每个title对应的段落数要与对应事先抽取的一致
            for pi, paragraph in enumerate(paragraphs):
                '''
                    paragraph_text：str，原始数据集中的passage句子
                    passage_tokens：list，事先已经进行分词的passage，每个token是一个list，包含10个特征，分别对应tokenid，token名称，
                    token类型，IOB标签，lemma标签，词性，标签，字符级别起始位置，字符级别终止位置，依存关系；
                    passagetoken2triple：dict，每个token对应的三元组及其位置（0：head，1：rel，2：tail）
                    triples：list，passage提炼抽取的三元组
                    corefs：list 指代消解列表
                '''
                passage_text = paragraph["context"]
                passage_tokens = title_struct_data[pi]['passage_tokens'] # 事先分词 # 样例：[11, 'governmental', '', 'O', 'governmental', 'ADJ', 'JJ', 56, 67, 'amod']
                passagetoken2triple = title_struct_data[pi]['passagetoken2triple'] # 每个token对应的三元组编号以及位置
                triples = title_struct_data[pi]['triples'] # 当前passage提炼的知识
                corefs = title_struct_data[pi]['corefs'] # 当前passage存在的指代消解
                corefs_dict = dict()
                for i in corefs:
                    corefs_dict[i[0]] = i[1]
                passage_tokens_ = [i[1] for i in passage_tokens]
                char_to_word_offset = self.char_to_token_span(passage_text, passage_tokens)
                #### 分别获取question ####
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    #### 根据question_text与三元组进行对齐，并与passage结合起来计算任意两个token之间的最短路径
                    question_tokens = []
                    questiontoken2triple = dict()
                    for token in self.nlp(question_text):
                        if (token.text).strip() == '':
                            continue
                        question_tokens.append(
                            [token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_,
                             token.idx, token.idx + len(token) - 1, token.dep_])
                        questiontoken2triple[token.i] = [[-1, -1]]
                        # 与passage的每个token进行相似度匹配，匹配最相关的token
                        if len(token.text) > 1:
                            for et, t in enumerate(passage_tokens_):
                                if len(t) > 1 and passagetoken2triple[et] != [[-1, -1]] and get_equal_rate(token.text, t) == True: # 说明当前question中的token与passage很相关
                                    if questiontoken2triple[token.i] == [[-1, -1]]:
                                        # print(et)
                                        questiontoken2triple[token.i] = passagetoken2triple[et]
                                    else:
                                        # print(et)
                                        questiontoken2triple[token.i] += passagetoken2triple[et]

                         # question当前token属于指代类词时：
                    for key, value in corefs_dict.items():  # 遍历每个指代消解
                        # 对于指代消解类的token，启发式地认为只要其指代的词出现在任意一个三元组的头实体或尾实体，则其将对应于对应的三元组
                        for ei, (head, _, tail) in enumerate(triples):
                            p = 0 if value in head else -1
                            p = 2 if value in tail else -1
                            if p != -1:
                                for token in question_tokens:
                                    if token[1] == key:
                                        if token[0] not in questiontoken2triple.keys() or questiontoken2triple[token[0]] == [[-1, -1]]:
                                            questiontoken2triple[token[0]] = [[ei, p]]
                                        else:
                                            questiontoken2triple[token[0]].append([ei, p])

                    #### 提取答案及对应的token级别的start和end ####
                    if is_training:
                        if version == "v2.0":
                            is_impossible = qa["is_impossible"]
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset] # 将字符级别的起始位置转换为token级别
                            end_position = char_to_word_offset[answer_offset + answer_length - 1] # 将字符级别的起始位置转换为token级别
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(passage_tokens_[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                        qas_id=qas_id,
                        passage_text=passage_text,
                        passage_tokens=passage_tokens,
                        passagetoken2triple=passagetoken2triple,
                        question_text=question_text,
                        question_tokens=question_tokens,
                        questiontoken2triple=questiontoken2triple,
                        refine_triples=triples,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                    )  # 相当于c++里的结构体
                    examples.append(example)
                    # print()
        return examples


    # 将数据集转换为模型的输入特征
    def convert_examples_to_features(self, examples, tokenizer, max_seq_length,
                                     doc_stride, max_query_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000
        # unique_id_file = os.path.join(args.squad_data_dir, 'unique_id.npy')
        # if os.path.exists(unique_id_file):
        #     unique_id = np.load(unique_id_file, allow_pickle=True)[()]
        features = []
        flag = True
        # save_every_time = 10000 # 表示每处理一定数量的样本后存入磁盘，以避免内存溢出
        if is_training == True:
            cached_train_features_file = os.path.join(args.squad_data_dir, 'squad_train_features_file.pkl')
        else:
            cached_train_features_file = os.path.join(args.squad_data_dir, 'squad_eval_features_file.pkl')

        for (example_index, example) in enumerate(tqdm(examples, ncols=50, desc="generate features:")):
            # que_span = example.que_span
            # org_que_token = example.token_que
            # org_doc_token = example.token_doc
            # all_doc_span = example.doc_span
            # question_tokens = [i[1] for i in example.question_tokens if i[1].strip() != ''] # question每个token文本
            # question_dep = [i[9] for i in example.question_tokens if i[1].strip() != ''] # question每个token的依存关系
            # passage_tokens = [i[1] for i in example.passage_tokens if i[1].strip() != ''] # passage每个token文本
            # passage_dep = [i[9] for i in example.passage_tokens if i[1].strip() != ''] # passage每个token的依存关系

            question_tokens = [i[1] for i in example.question_tokens] # question每个token文本
            question_dep = [i[9] for i in example.question_tokens] # question每个token的依存关系
            passage_tokens = [i[1] for i in example.passage_tokens] # passage每个token文本
            passage_dep = [i[9] for i in example.passage_tokens] # passage每个token的依存关系



            refine_triples = example.refine_triples # 结构化三元组
            bert_question_tokens = tokenizer.tokenize(example.question_text) # 使用bert的word piece分词工具

            bert_refine_triples = [] # [[['xx','xxx'], ['xx','xxx'], ['xx','xxx']],...]
            # 将三元组内的每个token通过BERT进行分词
            for head, rel, tail in refine_triples:
                head_ = tokenizer.tokenize(head) # word piece list
                rel_ = tokenizer.tokenize(rel) # word piece list
                tail_ = tokenizer.tokenize(tail) # word piece list
                bert_refine_triples.append([head_, rel_, tail_])

            bert_refine_triples_ids = []
            for head, rel, tail in bert_refine_triples:  # 将bert_refine_triples转换为id数值化操作
                head_ = tokenizer.convert_tokens_to_ids(head)
                rel_ = tokenizer.convert_tokens_to_ids(rel)
                tail_ = tokenizer.convert_tokens_to_ids(tail)
                bert_refine_triples_ids.append([head_, rel_, tail_])

            cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
            cls_str = int_list_to_str([cls_id])
            # 统计三元组中所有的实体
            # 新增一个CLS节点，与所有实体节点相连。
            # 目的包括两点：避免图是非连通的；其次，CLS节点可以代表整个input的主节点
            # 特别注意：在计算任意两点之间最短路径时，先不考虑CLS这个结点时候的最短路径，如果此时找不到路径说明这两点不连通，再通过CLS，否则任意两点的
            # 最短路径不会超过三个结点（CLS与所有结点向量，最短路径长度不可能超过3，但这显然在一些推理上不合理）
            bert_refine_triples_with_cls_ids = []
            entity_ids = dict()  # key保存了所有实体，value表示每个实体的邻接实体的列表
            # entity_with_cls_ids = dict()
            entpair2tripleid = dict()  # 给定两个实体确定其所在三元组编号
            for tid, (head, rel, tail) in enumerate(bert_refine_triples_ids):
                bert_refine_triples_with_cls_ids.append([head, rel, tail])
                head, rel, tail = int_list_to_str(head), int_list_to_str(rel), int_list_to_str(
                    tail)  # 列表不能作为字典的key，所以转换一下
                entpair2tripleid[(head, tail)] = tid
                if head not in entity_ids.keys():
                    entity_ids[head] = []
                if tail not in entity_ids.keys():
                    entity_ids[tail] = []
                entity_ids[head].append(tail)
                entity_ids[tail].append(head)
            # entity_with_cls_ids = copy.deepcopy(entity_ids)  # 深度拷贝，完全开辟一个新的内存
            for ent in entity_ids.keys():
                # entity_with_cls_ids[ent].append(cls_str)
                entpair2tripleid[(ent, cls_str)] = len(bert_refine_triples_with_cls_ids)
                bert_refine_triples_with_cls_ids.append([str_to_int_list(ent), [cls_id], [cls_id]])  # 额外添加新的CLS三元组
            # entity_with_cls_ids[cls_str] = [i for i in entity_with_cls_ids.keys()]

            # 此时图以及确定，因此事先求出任意两个实体之间的最短路径，留备后续计算token之间的推理路径
            #（决定：不在此处执行，而转移到batch训练时执行） ents_path_dict = find_Shortest_Path(bert_refine_triples_ids, bert_refine_triples_with_cls_ids, entity_ids, entpair2tripleid, cls_id)


            # sub_que_span = []
            que_org_to_split_map = {} # 原始分词的tokenid对应进行word piece后拆分的token的范围
            pre_tok_len = 0
            # BERT的word piece可能将原始的一个token分成多个token
            for idx, que_token in enumerate(question_tokens):
                if que_token.strip() == '':
                    sub_que_tok = ['']
                else:
                    sub_que_tok = tokenizer.tokenize(que_token) # 将原始的token进行word piece
                # 每个原始的token的id对应BERT的word piece的区间
                # 例如playing（id=10）的word piece为“play（id=15） ##img（id=16）”，则que_org_to_split_map[10]=(15,16)
                que_org_to_split_map[idx] = (pre_tok_len, len(sub_que_tok) + pre_tok_len - 1)
                pre_tok_len += len(sub_que_tok)

            bert_question_dep = [] # BERT分词后question对应每个token的依存关系
            bert_questiontoken2triple = [] # BERT分词后question对应每个token对应的三元组
            # 将原始的每个token对应的特征复制到对应的所有拆分的token
            for idx in range(len(question_tokens)):
                head_start, head_end = que_org_to_split_map[idx]  # 原始token对应新的word piece的区间范围
                for _ in range(head_end - head_start + 1):
                    bert_question_dep.append(question_dep[idx])
                    if idx not in example.questiontoken2triple.keys():
                        bert_questiontoken2triple.append([[-1, -1]])
                    else:
                        bert_questiontoken2triple.append(example.questiontoken2triple[idx])
            # print(len(bert_question_dep), len(bert_question_tokens))
            # assert len(bert_question_dep) == len(bert_question_tokens)
            # assert len(bert_questiontoken2triple) == len(bert_question_tokens)

            tok_to_orig_index = [] # BERT分词后，每个位置对应于原来分词的位置
            orig_to_tok_index = [] # 原始的token对应BERT分词后的索引
            bert_passage_tokens = [] # BERT分词后的token

            for (i, token) in enumerate(passage_tokens):
                orig_to_tok_index.append(len(bert_passage_tokens))
                if token.strip() == '':
                    sub_tokens = ['']
                else:
                    sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    bert_passage_tokens.append(sub_token)

            doc_org_to_split_map = {} # 原始分词的tokenid对应进行word piece后拆分的token的范围
            pre_tok_len = 0
            for idx, doc_token in enumerate(passage_tokens):
                if doc_token.strip() == '':
                    sub_doc_tok = ['']
                else:
                    sub_doc_tok = tokenizer.tokenize(doc_token)
                doc_org_to_split_map[idx] = (pre_tok_len, len(sub_doc_tok) + pre_tok_len - 1)
                pre_tok_len += len(sub_doc_tok)

            bert_passage_dep = [] # BERT分词后passage对应每个token的依存关系
            bert_passagetoken2triple = [] # BERT分词后passage对应每个token对应的三元组
            # 将原始的每个token对应的特征复制到对应的所有拆分的token
            for idx in range(len(passage_tokens)):
                head_start, head_end = doc_org_to_split_map[idx]  # 原始token对应新的word piece的区间范围
                for _ in range(head_end - head_start + 1):
                    bert_passage_dep.append(passage_dep[idx])
                    bert_passagetoken2triple.append(example.passagetoken2triple[idx])

            if len(bert_question_tokens) > max_query_length:
                bert_question_tokens = bert_question_tokens[0:max_query_length]

            tok_start_position = None
            tok_end_position = None
            if is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(passage_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(bert_passage_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    bert_passage_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(bert_question_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _PassSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            bert_passage_spans = []
            start_offset = 0
            # 假设模型最大长度限制是384，减去3个特殊字符，假设question的长度是81，则还有300个字符可以供passage使用
            # 假设此时passage长度是1000，很显然300个字符是放不下的，因此采用滑动窗口，窗口大小为此时的300，滑动的步长是stride=128
            # 最终得到的每个chunk对应的span（起始位置，窗口大小）为[(0, 300), (128, 300), (256, 300), (384, 300), (512, 300), (640, 300), (768, 232)]
            # 每一个chunk与question组合起来作为一个样本，因此原始的passage可能对应多个chunk，即多个样本
            while start_offset < len(bert_passage_tokens):
                length = len(bert_passage_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                bert_passage_spans.append(_PassSpan(start=start_offset, length=length))
                if start_offset + length == len(bert_passage_tokens):
                    break
                start_offset += min(length, doc_stride)

            # 需要说明的是：passage并非直接喂入模型，而是根据上面几行代码划分多个chunk，此处遍历每个chunk，并与question结合
            # 并添加包括CLS和SEP等特殊字符，构成一个新的样本，这样可以保证每个样本的长度在max_seq_len(384)之内
            # 因此我们在划分chunk之后，对新的样本的任意两个token去计算最短路径
            for (doc_span_index, doc_span) in enumerate(bert_passage_spans): # 遍历passage的每个chunk
                tokens = [] # 输入数据的每个token
                tokens_to_triple = [] # 输入数据每个token对齐的三元组
                token_to_orig_map = {}
                token_is_max_context = {}
                # use this idx list to select from doc span mask
                head_select_idx = []
                segment_ids = []
                tokens.append("[CLS]") # BERT分词中的第一个字符默认是[CLS]
                tokens_to_triple.append([[-1, -1]])
                segment_ids.append(0) # BERT的segment embedding所用（不同的句子对应一个segment id）

                # 第一个片段存放passage
                for i in range(doc_span.length): # 当前chunk内token的个数
                    split_token_index = doc_span.start + i
                    # 在新的tokens序列中，每个位置的token对应于最原始的passage的token的id索引
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(bert_passage_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context # 在新的tokens序列中，当前的token是否有最大的context
                    tokens.append(bert_passage_tokens[split_token_index]) # 新的tokens序列中添加一个bert的passage token
                    tokens_to_triple.append(bert_passagetoken2triple[split_token_index]) # 将对应的token对齐的三元组新增到输入token中
                    head_select_idx.append(split_token_index)

                    segment_ids.append(0)

                tokens.append("[SEP]")
                tokens_to_triple.append([[-1, -1]])
                segment_ids.append(0)
                start_doc_ix = head_select_idx[0]
                end_doc_ix = head_select_idx[-1]
                select_doc_len = end_doc_ix - start_doc_ix + 1
                select_que_len = len(bert_question_tokens)
                assert len(head_select_idx) == select_doc_len

                for idx, token in enumerate(bert_question_tokens):
                    tokens.append(token)
                    if idx >= len(bert_questiontoken2triple):
                        tokens_to_triple.append(bert_questiontoken2triple[-1])
                    else:
                        tokens_to_triple.append(bert_questiontoken2triple[idx])
                    segment_ids.append(1)

                tokens.append("[SEP]")
                tokens_to_triple.append([[-1, -1]])
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens) # 将token转换为id数值化操作


                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids) # 用于判断是否是原始值还是padding的部分

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                ### 此处开始两两token遍历，并根据其对齐的结构化三元组获取最短路径
                #（决定：不在此处执行，而转移到batch训练时执行） tokens_rel_path, tokens_ent_path = self.get_tokens_path(bert_refine_triples_ids, bert_refine_triples_with_cls_ids, entity_ids, entpair2tripleid, input_ids, tokens_to_triple, tokenizer, ents_path_dict)

                # assert len(input_ids) == max_seq_length
                # assert len(input_mask) == max_seq_length
                # assert len(segment_ids) == max_seq_length
                # assert len(tokens) == len(tokens_to_triple)
                # assert len(tokens_rel_path) == len(tokens_ent_path)

                start_position = None
                end_position = None
                is_impossible = example.is_impossible

                if is_training and not example.is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    # 判断当前的chunk内是否包含答案的span
                    if (example.start_position < doc_start or
                            example.end_position < doc_start or
                            example.start_position > doc_end or example.end_position > doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        is_impossible = True

                    else:
                        # doc_offset = len(query_tokens) + 2
                        doc_offset = 1  # [CLS]
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                        is_impossible = False

                if is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0
                    is_impossible = True
                # 每个样本对应一个输入特征
                features.append(
                    InputFeatures( # 用类对象很占内存！
                        unique_id=unique_id, # chunk样本的唯一编号
                        example_index=example_index, # 原始样本的唯一编号
                        doc_span_index=doc_span_index, # chunk编号
                        tokens=tokens, # chunk后的输入word piece token序列
                        token_to_orig_map=token_to_orig_map,  # map字典， key表示BERT分词后每个token的id，value表示key对应原始文本token的id
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids, # chunk后的输入word piece token的id序列
                        input_mask=input_mask, # mask矩阵用于判断是否是padding的部分
                        segment_ids=segment_ids, # 分段编号
                        start_position=start_position, # 正确答案的起始位置
                        end_position=end_position, # 正确答案的终止位置
                        is_impossible=is_impossible, # 答案是否不可回答
                        # input_rel_path=tokens_rel_path, # 任意两个token之间在图谱上的推理的关系路径
                        # input_ent_path=tokens_ent_path # 任意两个token之间在图谱上的推理的实体路径
                        ## 经过反复测试，如果直接保存tokens_rel_path和tokens_ent_path，内存占用巨大，因此
                        ## 决定保存生成路径的中间变量，在每次batch训练时进行路径生成。下面为生成路径的中间变量
                        bert_refine_triples_ids=bert_refine_triples_ids,
                        bert_refine_triples_with_cls_ids=bert_refine_triples_with_cls_ids,
                        entity_ids=entity_ids,
                        entpair2tripleid=entpair2tripleid,
                        tokens_to_triple=tokens_to_triple
                    ))

                # features.append(  # 用字典对象，并用np进行序列化
                #     {
                #         'unique_id': unique_id,  # chunk样本的唯一编号
                #         'example_index': example_index,  # 原始样本的唯一编号
                #         'doc_span_index': doc_span_index,  # chunk编号
                #         'tokens': tokens,  # chunk后的输入word piece token序列
                #         'token_to_orig_map': token_to_orig_map,  # map字典， key表示BERT分词后每个token的id，value表示key对应原始文本token的id
                #         'token_is_max_context': token_is_max_context,
                #         'input_ids': input_ids,  # chunk后的输入word piece token的id序列
                #         'input_mask': input_mask,  # mask矩阵用于判断是否是padding的部分
                #         'segment_ids': segment_ids,  # 分段编号
                #         'start_position': start_position,  # 正确答案的起始位置
                #         'end_position': end_position,  # 正确答案的终止位置
                #         'is_impossible': is_impossible,  # 答案是否不可回答
                #         'input_rel_path': tokens_rel_path,  # 任意两个token之间在图谱上的推理的关系路径（为了省空间，转换为dict字符串）
                #         'input_ent_path': tokens_ent_path  # 任意两个token之间在图谱上的推理的实体路径（为了省空间，转换为dict字符串）
                #     })

                unique_id += 1

            # if (example_index + 1) % save_every_time == 0:
            #     if os.path.exists(cached_train_features_file):
            #         with open(cached_train_features_file, "rb") as reader:
            #             train_features = pickle.load(reader)
            #         # train_features = np.load(cached_train_features_file, allow_pickle=True)[()].tolist()
            #     else:
            #         train_features = []
            #     with open(cached_train_features_file, "wb") as writer:
            #         pickle.dump(train_features + features, writer)
            #     # np.save(cached_train_features_file, train_features + features)
            #     np.save(unique_id_file, unique_id)
            #     del train_features
            #     features = []
        np.save('unique_id.npy', unique_id)
        return features

    # def test(self):
    #     cached_train_features_file = os.path.join(args.squad_data_dir, 'squad_train_features_file.pkl')
    #     if os.path.exists(cached_train_features_file):
    #         with open(cached_train_features_file, "rb") as reader:
    #             train_features = pickle.load(reader)
    #
    #         print(train_features[10000].unique_id)
    #         print()
    #         print(train_features[29995].unique_id)
    #         print()
    #         print(train_features[30005].unique_id)
    #         print()
    #
    #         for i in train_features[30000:]:
    #             i.unique_id += 30000
    #
    #         print(train_features[30005].unique_id)
    #
    #         with open(cached_train_features_file, "wb") as writer:
    #             pickle.dump(train_features, writer)

    def get_tokens_path(self, refine_triples_ids, refine_triples_with_cls_ids, entity_ids, entpair2tripleid,
                         input_ids, tokens_to_triple, tokenizer, ents_path_dict):
        '''
            根据三元组以及分词id来为每两个token生成最短路径以表示两两token之间的推理信息
            refine_triples_ids:非结构化文本提炼出来的三元组（每个token都id数值化），[[[23,11],[65],[2,12,33]],...]
            input_ids：输入的token序列，id数值化
            ents_path_dict: 遍历两两token。如果两两token遍历，时间复杂度非常高，不过偶一个trick，有许多word piece由于同属于同一个实体，则它们没有必要重复计算最短路径，因此每次存储一下两个token对应的实体的最短路径即可，这样避免重复的计算最短路径
        '''
        cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
        cls_str = int_list_to_str([cls_id])
        tokens_rel_paths, tokens_ent_paths = [], []

        for batch_i in range(len(refine_triples_ids)):

            # 遍历两两token，如果token
            # input_mask = np.zeros((len(input_ids), len(input_ids)))
            tokens_rel_path = dict() # 保存两两token之间所有的实体路径
            tokens_ent_path = dict() # 保存两两token之间所有的关系路径
            min_path_len = 1000
            # 遍历两两token。如果两两token遍历，时间复杂度非常高，不过偶一个trick，有许多word piece由于同属于同一个实体，则它们没有必要
            # 重复计算最短路径，因此每次存储一下两个token对应的实体的最短路径即可，这样避免重复的计算最短路径
            for token_idx, tokeni2triple in enumerate(tokens_to_triple[batch_i]):
                if tokeni2triple[0] != [[-1, -1]]: # 说明当前的token出现在三元组内，则该token参与attention计算
                    for idx, tokenj2triple in enumerate(tokens_to_triple[batch_i][token_idx:]):
                        token_jdx = token_idx + idx
                        if tokenj2triple != [[-1, -1]]: # 说明此时两个token都对应图中的三元组，一定存在路径，mask为1
                            # input_mask[token_idx, token_jdx] = 1
                            # input_mask[token_jdx, token_idx] = 1
                            # token只会唯一属于某一个实体或关系边，只不过这个实体或关系可能存在于多个三元组，因此只需要随便取一个肯定是对应的
                            # 唯一的三元组
                            posi = tokeni2triple[0][1] if tokeni2triple[0][1] in [0, 2] else 0 # 如果值是0或2，说明其属于实体，否则1为关系，当取1时，我们认为定义其补全至整个三元组，pos为0
                            posj = tokenj2triple[0][1] if tokenj2triple[0][1] in [0, 2] else 2
                            start = int_list_to_str(refine_triples_with_cls_ids[batch_i][0][tokeni2triple[0][0]][posi]) # 路径的起始实体(实体包含的word piece token id序列并以下划线相隔， 例如343_54_3232)
                            end = int_list_to_str(refine_triples_with_cls_ids[batch_i][0][tokenj2triple[0][0]][posj]) # 路径的终点实体
                            if (start, end) in ents_path_dict[batch_i].keys():
                                ent_path = ents_path_dict[batch_i][(start, end)][0]
                                rel_path = ents_path_dict[batch_i][(start, end)][1]
                            elif (end, start) in ents_path_dict[batch_i].keys():
                                ent_path = ents_path_dict[batch_i][(end, start)][0]
                                rel_path = ents_path_dict[batch_i][(end, start)][1]
                            else:
                                ent_path = [str_to_int_list(start), str_to_int_list(end)]
                                rel_path = [[cls_id]]  # CLS作为自环的关系
                            tokens_ent_path[(token_idx, token_jdx)] = ent_path
                            tokens_rel_path[(token_idx, token_jdx)] = rel_path
                            # ent_path.reverse()
                            # rel_path.reverse()
                            # tokens_ent_path[(token_jdx, token_idx)] = ent_path
                            # tokens_rel_path[(token_jdx, token_idx)] = rel_path
            '''
                tokens_rel_path样例：{(71,7):[[1112], [1107], [3152, 1106], [1112], [1982, 1107]]} 表示第71个token与第7个token在图谱中的关系推理路径（5条关系边，每个关系边由对应的word piece token组成）
                tokens_ent_path样例：{(71,7):[[1730, 2483, 1104, 155, 111, 139, 1873, 118, 1372, 16784, 112, 188, 6405], [1523, 3281], [8408], [2027], [1672, 4241, 1105, 5923, 6025], [113, 120, 100, 120, 17775, 118, 162, 11414, 118, 1474, 114]]}
            '''
            tokens_rel_paths.append(tokens_rel_path)
            tokens_ent_paths.append(tokens_ent_path)
        return tokens_rel_paths, tokens_ent_paths

    def find_shortest_path_and_convert_to_tensor(self, refine_triples_ids, refine_triples_with_cls_ids, entity_ids,
                                                 entpair2tripleid, cls_id):
        # graph: [[head, rel, tail], ...]
        cls_str = int_list_to_str([cls_id])
        # path_dicts = []
        entity2id_list = []
        input_rel_paths, input_ent_paths, input_rel_path_lengths, input_ent_path_lengths, input_path_masks = [], [], [], [], []

        for batch_i in range(len(refine_triples_ids)):  # 遍历每个batch样本

            input_rel_path = [0] * args.max_ent_num * args.max_ent_num * args.path_length * args.max_token_size
            input_ent_path = [0] * args.max_ent_num * args.max_ent_num * args.path_length * args.max_token_size
            input_rel_path_length = [0] * args.max_ent_num * args.max_ent_num
            input_ent_path_length = [0] * args.max_ent_num * args.max_ent_num
            input_path_mask = [0] * args.max_ent_num * args.max_ent_num

            second_col_num = args.max_ent_num * args.path_length * args.max_token_size  # 实际的矩阵某一行元素的个数
            third_col_num = args.path_length * args.max_token_size

            ent_pair = [(int_list_to_str(head), int_list_to_str(tail)) for head, _, tail in
                        refine_triples_ids[batch_i][0]]
            # ent_with_cls_pair = [(int_list_to_str(head), int_list_to_str(tail)) for head, _, tail in refine_triples_with_cls_ids]
            G1 = nx.Graph()
            G1.add_edges_from(ent_pair)
            nx.draw(G1, with_labels=True)
            entity2id = dict() # 每个entity对应的编号
            entity_list = [cls_str] # 第一个实体默认是CLS
            entity2id[cls_str] = 0
            for i in list(entity_ids[batch_i][0].keys())[:args.max_ent_num - 1]:
                entity_list.append(i)
                if i not in entity2id.keys():
                    entity2id[i] = len(entity2id)
            if len(entity_list) < args.max_ent_num:
                entity_list += [cls_str] * (args.max_ent_num - len(entity_list))

            assert len(entity_list) == args.max_ent_num

            path_dict = dict()
            for i in range(len(entity_list)):
                for j in range(i, len(entity_list)):
                    if i == j:
                        ent_path = [str_to_int_list(entity_list[i]), str_to_int_list(entity_list[i])]
                        rel_path = [[cls_id]]
                    elif entity_list[i] != cls_str and entity_list[j] != cls_str and nx.has_path(G1, entity_list[i], entity_list[j]):
                        ent_path = nx.shortest_path(G1, source=entity_list[i], target=entity_list[j])
                        rel_path = find_Rel_Path(ent_path, entpair2tripleid[batch_i][0],
                                                 refine_triples_with_cls_ids[batch_i][0])
                        ent_path = [str_to_int_list(k) for k in ent_path]
                    else: # 当两个实体不连通时，再通过额外添加的CLS结点计算路径
                        ent_path = [str_to_int_list(entity_list[i]), [cls_id], str_to_int_list(entity_list[j])]
                        rel_path = [[cls_id], [cls_id]]

                    # ent_path = ent_path[:args.path_length]
                    rel_path = rel_path[:args.path_length]
                    reverse_rel_path = ent_path[-1 * args.path_length:]
                    # reverse_ent_path = rel_path[-1 * args.path_length:]
                    reverse_rel_path.reverse()
                    # reverse_ent_path.reverse()

                    rel_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                                rel_path[:args.path_length]] \
                               + [[0] * args.max_token_size] * max(0, args.path_length - len(rel_path))
                    # ent_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                    #             ent_path[:args.path_length]] \
                    #            + [[0] * args.max_token_size] * max(0, args.path_length - len(ent_path))

                    reverse_rel_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                                        reverse_rel_path[:args.path_length]] \
                                       + [[0] * args.max_token_size] * max(0, args.path_length - len(reverse_rel_path))
                    # reverse_ent_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                    #                     reverse_ent_path[:args.path_length]] \
                    #                    + [[0] * args.max_token_size] * max(0, args.path_length - len(reverse_ent_path))

                    rel_path_ = [m for n in rel_path for m in n]
                    reverse_rel_path_ = [m for n in reverse_rel_path for m in n]
                    # ent_path_ = [m for n in ent_path for m in n]
                    # reverse_ent_path_ = [m for n in reverse_ent_path for m in n]

                    replace_ij_start = i * second_col_num + j * third_col_num
                    replace_ij_end = i * second_col_num + (j + 1) * third_col_num
                    replace_ji_start = j * second_col_num + i * third_col_num
                    replace_ji_end = j * second_col_num + (i + 1) * third_col_num

                    input_rel_path[replace_ij_start:replace_ij_end] = rel_path_
                    input_rel_path[replace_ji_start:replace_ji_end] = reverse_rel_path_
                    # input_ent_path[replace_ij_start:replace_ij_end] = ent_path_
                    # input_ent_path[replace_ji_start:replace_ji_end] = reverse_ent_path_

                    input_path_mask[i * args.max_ent_num + j] = 1
                    input_path_mask[j * args.max_ent_num + i] = 1

                    input_rel_path_length[i * args.max_ent_num + j] = np.sum(
                        np.sum(rel_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                    input_rel_path_length[j * args.max_ent_num + i] = np.sum(
                        np.sum(reverse_rel_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                    # input_ent_path_length[i * args.max_ent_num + j] = np.sum(
                    #     np.sum(ent_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                    # input_ent_path_length[j * args.max_ent_num + i] = np.sum(
                    #     np.sum(reverse_ent_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence

                # path_dict[(entity_list[i], entity_list[j])] = [ent_path, rel_path]
            # path_dicts.append(path_dict)
            input_rel_paths.append(input_rel_path)
            # input_ent_paths.append(input_ent_path)
            input_rel_path_lengths.append(input_rel_path_length)
            # input_ent_path_lengths.append(input_ent_path_length)
            input_path_masks.append(input_path_mask)
            entity2id_list.append(entity2id)

        # input_rel_paths = torch.tensor(input_rel_paths, dtype=torch.long).view(
        #     [-1, args.max_ent_num, args.max_ent_num, args.path_length, args.max_token_size])
        input_rel_paths = np.reshape(input_rel_paths,
                                            [-1, args.max_ent_num, args.max_ent_num, args.path_length, args.max_token_size]).tolist()
        # input_ent_paths = torch.tensor(input_ent_paths, dtype=torch.long).view(
        #     [-1, args.max_ent_num, args.max_ent_num, args.path_length, args.max_token_size])
        input_rel_path_lengths = np.reshape(input_rel_path_lengths,
                                            [-1, args.max_ent_num, args.max_ent_num]).tolist()
        # input_ent_path_lengths = np.reshape(input_ent_path_lengths,
        #                                     [-1, args.max_ent_num, args.max_ent_num]).tolist()
        # input_path_masks = torch.tensor(input_path_masks, dtype=torch.long).view(
        #     [-1, args.max_ent_num, args.max_ent_num])
        input_path_masks = np.reshape(input_path_masks,
                                            [-1, args.max_ent_num, args.max_ent_num]).tolist()

        return input_rel_paths, input_rel_path_lengths, input_path_masks, entity2id_list


    def get_tokens_to_entity(self, refine_triples_with_cls_ids, tokens_to_triple, entity2id_list):
        '''
        根据所有的实体，以及所有token，获得tokeni和tokenj之间的mask值，并将其指向知识图谱中实体对路径的索引
        '''
        input_token_to_entitys, input_path_masks = [], []

        for batch_i in range(len(entity2id_list)):

            # input_rel_path = [[[[0] * args.max_token_size] * args.path_length] * args.max_seq_length] * args.max_seq_length
            # input_ent_path = [[[[0] * args.max_token_size] * args.path_length] * args.max_seq_length] * args.max_seq_length
            # input_rel_path_length = [[0] * args.max_seq_length] * args.max_seq_length
            # input_ent_path_length = [[0] * args.max_seq_length] * args.max_seq_length
            # input_path_mask = [[0] * args.max_seq_length] * args.max_seq_length

            ## trick：按理来说应该初始化一个维度为[384,384,5,5]的矩阵（如上面几行），但是测试发现很耗时间，所以先初始化一个384x384x5x5个元素的以为向量
            ## 这样速度很快，后面赋值时直接通过为止索引推导一下，转换为tensor后再通过view进行维度变换
            input_token_to_entity = [0] * args.max_seq_length * args.max_seq_length
            input_path_mask = [0] * args.max_seq_length * args.max_seq_length

            second_col_num = args.max_seq_length * args.path_length * args.max_token_size # 实际的矩阵某一行元素的个数
            third_col_num = args.path_length * args.max_token_size

            # 用于暂存当前的tokeni与tokenj的信息，如果下一个tokenj与前一个一样与相同的实体存在对齐，则直接从这里读取，避免重复计算
            token_ij_align = None
            for token_idx, tokeni2triple in enumerate(tokens_to_triple[batch_i]):
                if tokeni2triple[0] != [[-1, -1]]: # 说明当前的token出现在三元组内，则该token参与attention计算
                    for idx, tokenj2triple in enumerate(tokens_to_triple[batch_i][token_idx:]):
                        token_jdx = token_idx + idx
                        if tokenj2triple != [[-1, -1]]: # 说明此时两个token都对应图中的三元组，一定存在路径，mask为1

                            # token只会唯一属于某一个实体或关系边，只不过这个实体或关系可能存在于多个三元组，因此只需要随便取一个肯定是对应的
                            # 唯一的三元组
                            posi = tokeni2triple[0][1] if tokeni2triple[0][1] in [0, 2] else 0 # 如果值是0或2，说明其属于实体，否则1为关系，当取1时，我们认为定义其补全至整个三元组，pos为0
                            posj = tokenj2triple[0][1] if tokenj2triple[0][1] in [0, 2] else 2
                            start = int_list_to_str(refine_triples_with_cls_ids[batch_i][0][tokeni2triple[0][0]][posi]) # 路径的起始实体(实体包含的word piece token id序列并以下划线相隔， 例如343_54_3232)
                            end = int_list_to_str(refine_triples_with_cls_ids[batch_i][0][tokenj2triple[0][0]][posj]) # 路径的终点实体
                            startid, endid = 0, 0
                            if start in entity2id_list[batch_i].keys():
                                startid = entity2id_list[batch_i][start]
                            if end in entity2id_list[batch_i].keys():
                                endid = entity2id_list[batch_i][end]

                            input_token_to_entity[token_idx * args.max_seq_length + token_jdx] = startid * args.max_ent_num + endid
                            input_token_to_entity[token_jdx * args.max_seq_length + token_idx] = endid * args.max_ent_num + startid

                            input_path_mask[token_idx * args.max_seq_length + token_jdx] = 1
                            input_path_mask[token_jdx * args.max_seq_length + token_idx] = 1

            input_token_to_entitys.append(input_token_to_entity)
            input_path_masks.append(input_path_mask)
        # input_token_to_entitys = torch.tensor(input_token_to_entitys, dtype=torch.long).view([-1, args.max_seq_length, args.max_seq_length])
        # input_path_masks = torch.tensor(input_path_masks, dtype=torch.long).view([-1, args.max_seq_length, args.max_seq_length])
        input_token_to_entitys = np.reshape(input_token_to_entitys,
                                      [-1, args.max_seq_length, args.max_seq_length]).tolist()
        input_path_masks = np.reshape(input_path_masks,
                                      [-1, args.max_seq_length, args.max_seq_length]).tolist()
        return input_token_to_entitys, input_path_masks




    def get_tokens_path_and_convert_to_tensor(self, refine_triples_ids, refine_triples_with_cls_ids, entity_ids,
                                              entpair2tripleid, input_ids, tokens_to_triple, tokenizer, ents_path_dict):
        '''
        对每个token与token之间推理路径，并转换为张量
        '''
        cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
        cls_str = int_list_to_str([cls_id])
        # tokens_rel_paths, tokens_ent_paths = [], []

        input_rel_paths, input_ent_paths, input_rel_path_lengths, input_ent_path_lengths, input_path_masks = [], [], [], [], []

        for batch_i in range(len(refine_triples_ids)):

            # input_rel_path = [[[[0] * args.max_token_size] * args.path_length] * args.max_seq_length] * args.max_seq_length
            # input_ent_path = [[[[0] * args.max_token_size] * args.path_length] * args.max_seq_length] * args.max_seq_length
            # input_rel_path_length = [[0] * args.max_seq_length] * args.max_seq_length
            # input_ent_path_length = [[0] * args.max_seq_length] * args.max_seq_length
            # input_path_mask = [[0] * args.max_seq_length] * args.max_seq_length

            ## trick：按理来说应该初始化一个维度为[384,384,5,5]的矩阵（如上面几行），但是测试发现很耗时间，所以先初始化一个384x384x5x5个元素的以为向量
            ## 这样速度很快，后面赋值时直接通过为止索引推导一下，转换为tensor后再通过view进行维度变换
            input_rel_path = [0] * args.max_seq_length * args.max_seq_length * args.path_length * args.max_token_size
            input_ent_path = [0] * args.max_seq_length * args.max_seq_length * args.path_length * args.max_token_size
            input_rel_path_length = [0] * args.max_seq_length * args.max_seq_length
            input_ent_path_length = [0] * args.max_seq_length * args.max_seq_length
            input_path_mask = [0] * args.max_seq_length * args.max_seq_length

            second_col_num = args.max_seq_length * args.path_length * args.max_token_size # 实际的矩阵某一行元素的个数
            third_col_num = args.path_length * args.max_token_size

            # 遍历两两token，如果token
            # input_mask = np.zeros((len(input_ids), len(input_ids)))
            # tokens_rel_path = dict() # 保存两两token之间所有的实体路径
            # tokens_ent_path = dict() # 保存两两token之间所有的关系路径
            # min_path_len = 1000
            # 遍历两两token。如果两两token遍历，时间复杂度非常高，不过偶一个trick，有许多word piece由于同属于同一个实体，则它们没有必要
            # 重复计算最短路径，因此每次存储一下两个token对应的实体的最短路径即可，这样避免重复的计算最短路径

            # 用于暂存当前的tokeni与tokenj的信息，如果下一个tokenj与前一个一样与相同的实体存在对齐，则直接从这里读取，避免重复计算
            token_ij_align = None
            for token_idx, tokeni2triple in enumerate(tokens_to_triple[batch_i]):
                if tokeni2triple[0] != [[-1, -1]]: # 说明当前的token出现在三元组内，则该token参与attention计算
                    for idx, tokenj2triple in enumerate(tokens_to_triple[batch_i][token_idx:]):
                        token_jdx = token_idx + idx
                        if tokenj2triple != [[-1, -1]]: # 说明此时两个token都对应图中的三元组，一定存在路径，mask为1
                            state = True
                            if token_ij_align is None: # 第一次
                                token_ij_align = (tokeni2triple[0], tokenj2triple)
                            else:
                                if (tokeni2triple[0], tokenj2triple) == token_ij_align:
                                    # 这里是一个trick优化：说明当前的tokeni和tokenj与上一轮次相同，则无需再重复计算，直接取上次的结果赋值即可
                                    state = False
                                else:
                                    token_ij_align = (tokeni2triple[0], tokenj2triple)

                            if state is True:
                                # token只会唯一属于某一个实体或关系边，只不过这个实体或关系可能存在于多个三元组，因此只需要随便取一个肯定是对应的
                                # 唯一的三元组
                                posi = tokeni2triple[0][1] if tokeni2triple[0][1] in [0, 2] else 0 # 如果值是0或2，说明其属于实体，否则1为关系，当取1时，我们认为定义其补全至整个三元组，pos为0
                                posj = tokenj2triple[0][1] if tokenj2triple[0][1] in [0, 2] else 2
                                start = int_list_to_str(refine_triples_with_cls_ids[batch_i][0][tokeni2triple[0][0]][posi]) # 路径的起始实体(实体包含的word piece token id序列并以下划线相隔， 例如343_54_3232)
                                end = int_list_to_str(refine_triples_with_cls_ids[batch_i][0][tokenj2triple[0][0]][posj]) # 路径的终点实体
                                if (start, end) in ents_path_dict[batch_i].keys():
                                    ent_path = ents_path_dict[batch_i][(start, end)][0]
                                    rel_path = ents_path_dict[batch_i][(start, end)][1]
                                elif (end, start) in ents_path_dict[batch_i].keys():
                                    ent_path = ents_path_dict[batch_i][(end, start)][0]
                                    rel_path = ents_path_dict[batch_i][(end, start)][1]
                                else:
                                    ent_path = [str_to_int_list(start), str_to_int_list(end)]
                                    rel_path = [[cls_id]]  # CLS作为自环的关系

                                ent_path = ent_path[:args.path_length]
                                rel_path = rel_path[:args.path_length]
                                reverse_rel_path = ent_path[-1 * args.path_length:]
                                reverse_ent_path = rel_path[-1 * args.path_length:]
                                reverse_rel_path.reverse()
                                reverse_ent_path.reverse()

                                rel_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in rel_path[:args.path_length]]\
                                           + [[0] * args.max_token_size] * max(0, args.path_length - len(rel_path))
                                ent_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in ent_path[:args.path_length]]\
                                           + [[0] * args.max_token_size] * max(0, args.path_length - len(ent_path))

                                reverse_rel_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in reverse_rel_path[:args.path_length]]\
                                                   + [[0] * args.max_token_size] * max(0, args.path_length - len(reverse_rel_path))
                                reverse_ent_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in reverse_ent_path[:args.path_length]]\
                                                   + [[0] * args.max_token_size] * max(0, args.path_length - len(reverse_ent_path))

                                rel_path_ = [m for n in rel_path for m in n]
                                reverse_rel_path_ = [m for n in reverse_rel_path for m in n]
                                ent_path_ = [m for n in ent_path for m in n]
                                reverse_ent_path_ = [m for n in reverse_ent_path for m in n]

                            replace_ij_start = token_idx * second_col_num + token_jdx * third_col_num
                            replace_ij_end = token_idx * second_col_num + (token_jdx + 1) * third_col_num
                            replace_ji_start = token_jdx * second_col_num + token_idx * third_col_num
                            replace_ji_end = token_jdx * second_col_num + (token_idx + 1) * third_col_num

                            input_rel_path[replace_ij_start:replace_ij_end] = rel_path_
                            input_rel_path[replace_ji_start:replace_ji_end] = reverse_rel_path_
                            input_ent_path[replace_ij_start:replace_ij_end] = ent_path_
                            input_ent_path[replace_ji_start:replace_ji_end] = reverse_ent_path_

                            input_path_mask[token_idx * args.max_seq_length + token_jdx] = 1
                            input_path_mask[token_jdx * args.max_seq_length + token_idx] = 1

                            input_rel_path_length[token_idx * args.max_seq_length + token_jdx] = np.sum(np.sum(rel_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                            input_rel_path_length[token_jdx * args.max_seq_length + token_idx] = np.sum(np.sum(reverse_rel_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                            input_ent_path_length[token_idx * args.max_seq_length + token_jdx] = np.sum(np.sum(ent_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                            input_ent_path_length[token_jdx * args.max_seq_length + token_idx] = np.sum(np.sum(reverse_ent_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence


            '''
                tokens_rel_path样例：{(71,7):[[1112], [1107], [3152, 1106], [1112], [1982, 1107]]} 表示第71个token与第7个token在图谱中的关系推理路径（5条关系边，每个关系边由对应的word piece token组成）
                tokens_ent_path样例：{(71,7):[[1730, 2483, 1104, 155, 111, 139, 1873, 118, 1372, 16784, 112, 188, 6405], [1523, 3281], [8408], [2027], [1672, 4241, 1105, 5923, 6025], [113, 120, 100, 120, 17775, 118, 162, 11414, 118, 1474, 114]]}
            '''
            input_rel_paths.append(input_rel_path)
            input_ent_paths.append(input_ent_path)
            input_rel_path_lengths.append(input_rel_path_length)
            input_ent_path_lengths.append(input_ent_path_length)
            input_path_masks.append(input_path_mask)
            # tokens_rel_paths.append(tokens_rel_path)
            # tokens_ent_paths.append(tokens_ent_path)
        input_rel_paths = torch.tensor(input_rel_paths, dtype=torch.long).view([-1, args.max_seq_length, args.max_seq_length, args.path_length, args.max_token_size])
        input_ent_paths = torch.tensor(input_ent_paths, dtype=torch.long).view([-1, args.max_seq_length, args.max_seq_length, args.path_length, args.max_token_size])
        input_rel_path_lengths = np.reshape(input_rel_path_lengths, [-1, args.max_seq_length, args.max_seq_length]).tolist()
        input_ent_path_lengths = np.reshape(input_ent_path_lengths, [-1, args.max_seq_length, args.max_seq_length]).tolist()
        input_path_masks = torch.tensor(input_path_masks, dtype=torch.long).view([-1, args.max_seq_length, args.max_seq_length])

        return input_rel_paths, input_ent_paths, input_rel_path_lengths, input_ent_path_lengths, input_path_masks


    def convert_path_dict_to_tensor(self, args, input_rel_path_dict, input_ent_path_dict):
        '''
            input_rel_path样例：{(71,7):[[1112], [1107], [3152, 1106], [1112], [1982, 1107]]} 表示第71个token与第7个token在图谱中的关系推理路径（5条关系边，每个关系边由对应的word piece token组成）
            input_ent_path样例：{(71,7):[[1730, 2483, 1104, 155, 111, 139, 1873, 118, 1372, 16784, 112, 188, 6405], [1523, 3281], [8408], [2027], [1672, 4241, 1105, 5923, 6025], [113, 120, 100, 120, 17775, 118, 162, 11414, 118, 1474, 114]]}
            将原始的dict转换为张量形式 [batch_size, sequence_length, sequence_length, path_length, max_token_size]
        '''
        assert len(input_ent_path_dict) == len(input_rel_path_dict)
        input_rel_path = [[[[[0]*args.max_token_size]*args.path_length]*args.max_seq_length]*args.max_seq_length]*len(input_rel_path_dict)
        input_ent_path = [[[[[0]*args.max_token_size]*args.path_length]*args.max_seq_length]*args.max_seq_length]*len(input_ent_path_dict)
        input_rel_path_length = [[[0]*args.max_seq_length]*args.max_seq_length]*len(input_rel_path_dict)
        input_ent_path_length = [[[0]*args.max_seq_length]*args.max_seq_length]*len(input_ent_path_dict)
        input_path_mask = [[[0]*args.max_seq_length]*args.max_seq_length]*len(input_ent_path_dict)

        # input_rel_path = np.zeros([len(input_rel_path_dict), args.max_seq_length, args.max_seq_length, args.path_length,
        #                            args.max_token_size]).tolist()
        # input_ent_path = np.zeros([len(input_ent_path_dict), args.max_seq_length, args.max_seq_length, args.path_length,
        #                            args.max_token_size]).tolist()
        # input_rel_path_length = np.zeros([len(input_rel_path_dict), args.max_seq_length, args.max_seq_length]).tolist()
        # input_ent_path_length = np.zeros([len(input_ent_path_dict), args.max_seq_length, args.max_seq_length]).tolist()
        # input_path_mask = np.zeros([len(input_rel_path_dict), args.max_seq_length, args.max_seq_length]).tolist()

        for example in range(len(input_ent_path_dict)): # 遍历一个batch内的所有样本
            for (tgt, src) in input_ent_path_dict[example].keys():
                rel_path = input_rel_path_dict[example][(tgt, src)][:args.path_length]
                ent_path = input_ent_path_dict[example][(tgt, src)][:args.path_length]
                reverse_rel_path = input_rel_path_dict[example][(tgt, src)][-1 * args.path_length:]
                reverse_ent_path = input_ent_path_dict[example][(tgt, src)][-1 * args.path_length:]
                # reverse_rel_path = (copy.deepcopy(rel_path))
                # reverse_ent_path = (copy.deepcopy(ent_path))
                reverse_rel_path.reverse()
                reverse_ent_path.reverse()

                rel_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                            rel_path[:args.path_length]] + [[0] * args.max_token_size] * max(0, args.path_length - len(rel_path))
                ent_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                            ent_path[:args.path_length]] + [[0] * args.max_token_size] * max(0, args.path_length - len(ent_path))

                reverse_rel_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                                    reverse_rel_path[:args.path_length]] + [[0] * args.max_token_size] * max(0, args.path_length - len(
                                                                                                                 reverse_rel_path))
                reverse_ent_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                                    reverse_ent_path[:args.path_length]] + [[0] * args.max_token_size] * max(0, args.path_length - len(
                                                                                                                 reverse_ent_path))

                input_rel_path[example][tgt][src] = rel_path
                input_rel_path[example][src][tgt] = reverse_rel_path
                input_ent_path[example][tgt][src] = ent_path
                input_ent_path[example][src][tgt] = reverse_ent_path

                input_path_mask[example][tgt][src] = 1
                input_path_mask[example][src][tgt] = 1

                input_rel_path_length = np.sum(np.sum(input_rel_path, -1) != 0) # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                input_ent_path_length = np.sum(np.sum(input_ent_path, -1) != 0) # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
        return torch.tensor(input_rel_path, dtype=torch.long), torch.tensor(input_ent_path, dtype=torch.long), input_rel_path_length, input_ent_path_length, torch.tensor(input_path_mask, dtype=torch.long)



    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, bert_passage_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""
        # 判断给定的token其是对应的上下文context是否是最大的，具体的例子描述如下所示：

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(bert_passage_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index









class NewsqaLoader():
    # 用于加载Newsqa数据集
    def __init__(self):
        self.train_file = args.train_file # 原始训练数据集
        self.predict_file = args.predict_file # 原始验证数据集
        self.test_file = None # 原始测试数据集
        self.train_struct_file = os.path.join(args.newsqa_data_dir, 'newsqa_train_structure_knowledge.npy')
        self.dev_struct_file = os.path.join(args.newsqa_data_dir, 'newsqa_dev_structure_knowledge.npy')
        self.test_struct_file = os.path.join(args.newsqa_data_dir, 'newsqa_test_structure_knowledge.npy')
        self.nlp = spacy.load("en_core_web_sm")

    def is_whitespace(self, c):  # 判断token是否是空格
        if c == " " or c == " or c ==\t" or c == "\r" "\n" or ord(c) == 0x202F:
            return True
        return False

    def char_to_token_span(self, text, tokens):
        # newsqa原文给定的answer的区间是字符级别的区间范围，现需要根据分词的情况转换为对应的token级别的区间
        char_to_word_offset = dict()
        for i in range(len(text)):
            char_to_word_offset[i] = -1

        for token in tokens:
            # tokens 样例：[11, 'governmental', '', 'O', 'governmental', 'ADJ', 'JJ', 56, 67, 'amod']
            for j in range(token[7], token[8]+1):
                char_to_word_offset[j] = token[0]
        return char_to_word_offset

    # 读取数据
    # 读取原始的NewsQA数据集，加载每个passage对应的refining的结构化知识，并对对应的每个question进行分词，并与三元组对齐；
    def read_newsqa_examples(self, status='train'):
        if status == 'train':
            input_file = self.train_file
            struct_file = self.train_struct_file
        elif status == 'dev':
            input_file = self.predict_file
            struct_file = self.dev_struct_file
        else:
            input_file = self.test_file
            struct_file = self.test_struct_file
        """Read a Newsqa json file into a list of NewsqaExample."""
        ## 读取squad原始数据集
        with open(input_file, "r", encoding='utf-8') as reader:
            source = json.load(reader)
            input_data = source["data"]
            version = source["version"]
        input_tag_data = []
        # 读取抽取的结构化知识
        input_struct_data = np.load(struct_file, allow_pickle=True)[()]

        qas_num = 0
        examples = []
        for entry in tqdm(input_data, ncols=50, desc="reading examples"):
            storyId = entry['storyId']
            title_struct_data = input_struct_data[storyId][0] # 保存每个storyId对应的text抽取的结构知识
            passage_text = entry['text']  # 文章文本内容
            type = entry['type']  # 样本类型（train/dev/test）

            '''
                paragraph_text：str，原始数据集中的passage句子
                passage_tokens：list，事先已经进行分词的passage，每个token是一个list，包含10个特征，分别对应tokenid，token名称，
                token类型，IOB标签，lemma标签，词性，标签，字符级别起始位置，字符级别终止位置，依存关系；
                passagetoken2triple：dict，每个token对应的三元组及其位置（0：head，1：rel，2：tail）
                triples：list，passage提炼抽取的三元组
                corefs：list 指代消解列表
            '''
            passage_tokens = title_struct_data['passage_tokens'] # 事先分词 # 样例：[11, 'governmental', '', 'O', 'governmental', 'ADJ', 'JJ', 56, 67, 'amod']
            passagetoken2triple = title_struct_data['passagetoken2triple'] # 每个token对应的三元组编号以及位置
            triples = title_struct_data['triples'] # 当前passage提炼的知识
            corefs = title_struct_data['corefs'] # 当前passage存在的指代消解
            corefs_dict = dict()
            for i in corefs:
                corefs_dict[i[0]] = i[1]
            passage_tokens_ = [i[1] for i in passage_tokens]
            char_to_word_offset = self.char_to_token_span(passage_text, passage_tokens)
            #### 分别获取question ####
            for qa in entry["questions"]:
                qas_id = qas_num
                qas_num += 1
                question_text = qa["q"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                #### 根据question_text与三元组进行对齐，并与passage结合起来计算任意两个token之间的最短路径
                question_tokens = []
                questiontoken2triple = dict()
                for token in self.nlp(question_text):
                    if (token.text).strip() == '':
                        continue
                    question_tokens.append(
                        [token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_,
                         token.idx, token.idx + len(token) - 1, token.dep_])
                    questiontoken2triple[token.i] = [[-1, -1]]
                    # 与passage的每个token进行相似度匹配，匹配最相关的token
                    if len(token.text) > 1:
                        for et, t in enumerate(passage_tokens_):
                            if len(t) > 1 and passagetoken2triple[et] != [[-1, -1]] and get_equal_rate(token.text, t) == True: # 说明当前question中的token与passage很相关
                                if questiontoken2triple[token.i] == [[-1, -1]]:
                                    # print(et)
                                    questiontoken2triple[token.i] = passagetoken2triple[et]
                                else:
                                    # print(et)
                                    questiontoken2triple[token.i] += passagetoken2triple[et]

                # question当前token属于指代类词时：
                for key, value in corefs_dict.items():  # 遍历每个指代消解
                    # 对于指代消解类的token，启发式地认为只要其指代的词出现在任意一个三元组的头实体或尾实体，则其将对应于对应的三元组
                    for ei, (head, _, tail) in enumerate(triples):
                        p = 0 if value in head else -1
                        p = 2 if value in tail else -1
                        if p != -1:
                            for token in question_tokens:
                                if token[1] == key:
                                    if token[0] not in questiontoken2triple.keys() or questiontoken2triple[token[0]] == [[-1, -1]]:
                                        questiontoken2triple[token[0]] = [[ei, p]]
                                    else:
                                        questiontoken2triple[token[0]].append([ei, p])

                #### 提取答案及对应的token级别的start和end ####
                if status == 'train':
                    # 判断当前的问题是否可答
                    isAnswerAbsent = qa['isAnswerAbsent']
                    if isAnswerAbsent > 0.5: # newsqa数据集给定了答案是否可答的一个评分，当超过0.5时认为是不可答的
                        is_impossible = True
                    if not is_impossible:
                        answer = qa['answers'][0]
                        # orig_answer_text = answer["text"]
                        if 's' not in answer['sourcerAnswers'][0].keys():
                            is_impossible = True
                        else:
                            answer_start = answer['sourcerAnswers'][0]['s']
                            answer_end = answer['sourcerAnswers'][0]['e'] - 1 # 原始给的数据集中end-1位置表示最后一个字符
                            orig_answer_text = passage_text[int(answer_start): int(answer_end) + 1]
                            # answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_start] # 将字符级别的起始位置转换为token级别
                            end_position = char_to_word_offset[answer_end] # 将字符级别的起始位置转换为token级别
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(passage_tokens_[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                continue
                    if is_impossible:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    passage_text=passage_text,
                    passage_tokens=passage_tokens,
                    passagetoken2triple=passagetoken2triple,
                    question_text=question_text,
                    question_tokens=question_tokens,
                    questiontoken2triple=questiontoken2triple,
                    refine_triples=triples,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                )  # 相当于c++里的结构体
                examples.append(example)
                # print()
        return examples


    # 将数据集转换为模型的输入特征
    def convert_examples_to_features(self, examples, tokenizer, max_seq_length,
                                     doc_stride, max_query_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000
        # unique_id_file = os.path.join(args.newsqa_data_dir, 'unique_id.npy')
        # if os.path.exists(unique_id_file):
        #     unique_id = np.load(unique_id_file, allow_pickle=True)[()]
        features = []
        flag = True
        # save_every_time = 10000 # 表示每处理一定数量的样本后存入磁盘，以避免内存溢出
        if is_training == True:
            cached_train_features_file = os.path.join(args.newsqa_data_dir, 'newsqa_train_features_file.pkl')
        else:
            cached_train_features_file = os.path.join(args.newsqa_data_dir, 'newsqa_eval_features_file.pkl')

        for (example_index, example) in enumerate(tqdm(examples, ncols=50, desc="generate features:")):
            # que_span = example.que_span
            # org_que_token = example.token_que
            # org_doc_token = example.token_doc
            # all_doc_span = example.doc_span
            # question_tokens = [i[1] for i in example.question_tokens if i[1].strip() != ''] # question每个token文本
            # question_dep = [i[9] for i in example.question_tokens if i[1].strip() != ''] # question每个token的依存关系
            # passage_tokens = [i[1] for i in example.passage_tokens if i[1].strip() != ''] # passage每个token文本
            # passage_dep = [i[9] for i in example.passage_tokens if i[1].strip() != ''] # passage每个token的依存关系

            question_tokens = [i[1] for i in example.question_tokens] # question每个token文本
            question_dep = [i[9] for i in example.question_tokens] # question每个token的依存关系
            passage_tokens = [i[1] for i in example.passage_tokens] # passage每个token文本
            passage_dep = [i[9] for i in example.passage_tokens] # passage每个token的依存关系



            refine_triples = example.refine_triples # 结构化三元组
            bert_question_tokens = tokenizer.tokenize(example.question_text) # 使用bert的word piece分词工具

            bert_refine_triples = [] # [[['xx','xxx'], ['xx','xxx'], ['xx','xxx']],...]
            # 将三元组内的每个token通过BERT进行分词
            for head, rel, tail in refine_triples:
                head_ = tokenizer.tokenize(head) # word piece list
                rel_ = tokenizer.tokenize(rel) # word piece list
                tail_ = tokenizer.tokenize(tail) # word piece list
                bert_refine_triples.append([head_, rel_, tail_])

            bert_refine_triples_ids = []
            for head, rel, tail in bert_refine_triples:  # 将bert_refine_triples转换为id数值化操作
                head_ = tokenizer.convert_tokens_to_ids(head)
                rel_ = tokenizer.convert_tokens_to_ids(rel)
                tail_ = tokenizer.convert_tokens_to_ids(tail)
                bert_refine_triples_ids.append([head_, rel_, tail_])

            cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
            cls_str = int_list_to_str([cls_id])
            # 统计三元组中所有的实体
            # 新增一个CLS节点，与所有实体节点相连。
            # 目的包括两点：避免图是非连通的；其次，CLS节点可以代表整个input的主节点
            # 特别注意：在计算任意两点之间最短路径时，先不考虑CLS这个结点时候的最短路径，如果此时找不到路径说明这两点不连通，再通过CLS，否则任意两点的
            # 最短路径不会超过三个结点（CLS与所有结点向量，最短路径长度不可能超过3，但这显然在一些推理上不合理）
            bert_refine_triples_with_cls_ids = []
            entity_ids = dict()  # key保存了所有实体，value表示每个实体的邻接实体的列表
            # entity_with_cls_ids = dict()
            entpair2tripleid = dict()  # 给定两个实体确定其所在三元组编号
            for tid, (head, rel, tail) in enumerate(bert_refine_triples_ids):
                bert_refine_triples_with_cls_ids.append([head, rel, tail])
                head, rel, tail = int_list_to_str(head), int_list_to_str(rel), int_list_to_str(
                    tail)  # 列表不能作为字典的key，所以转换一下
                entpair2tripleid[(head, tail)] = tid
                if head not in entity_ids.keys():
                    entity_ids[head] = []
                if tail not in entity_ids.keys():
                    entity_ids[tail] = []
                entity_ids[head].append(tail)
                entity_ids[tail].append(head)
            # entity_with_cls_ids = copy.deepcopy(entity_ids)  # 深度拷贝，完全开辟一个新的内存
            for ent in entity_ids.keys():
                # entity_with_cls_ids[ent].append(cls_str)
                entpair2tripleid[(ent, cls_str)] = len(bert_refine_triples_with_cls_ids)
                bert_refine_triples_with_cls_ids.append([str_to_int_list(ent), [cls_id], [cls_id]])  # 额外添加新的CLS三元组
            # entity_with_cls_ids[cls_str] = [i for i in entity_with_cls_ids.keys()]

            # 此时图以及确定，因此事先求出任意两个实体之间的最短路径，留备后续计算token之间的推理路径
            #（决定：不在此处执行，而转移到batch训练时执行） ents_path_dict = find_Shortest_Path(bert_refine_triples_ids, bert_refine_triples_with_cls_ids, entity_ids, entpair2tripleid, cls_id)


            # sub_que_span = []
            que_org_to_split_map = {} # 原始分词的tokenid对应进行word piece后拆分的token的范围
            pre_tok_len = 0
            # BERT的word piece可能将原始的一个token分成多个token
            for idx, que_token in enumerate(question_tokens):
                if que_token.strip() == '':
                    sub_que_tok = ['']
                else:
                    sub_que_tok = tokenizer.tokenize(que_token) # 将原始的token进行word piece
                # 每个原始的token的id对应BERT的word piece的区间
                # 例如playing（id=10）的word piece为“play（id=15） ##img（id=16）”，则que_org_to_split_map[10]=(15,16)
                que_org_to_split_map[idx] = (pre_tok_len, len(sub_que_tok) + pre_tok_len - 1)
                pre_tok_len += len(sub_que_tok)

            bert_question_dep = [] # BERT分词后question对应每个token的依存关系
            bert_questiontoken2triple = [] # BERT分词后question对应每个token对应的三元组
            # 将原始的每个token对应的特征复制到对应的所有拆分的token
            for idx in range(len(question_tokens)):
                head_start, head_end = que_org_to_split_map[idx]  # 原始token对应新的word piece的区间范围
                for _ in range(head_end - head_start + 1):
                    bert_question_dep.append(question_dep[idx])
                    if idx not in example.questiontoken2triple.keys():
                        bert_questiontoken2triple.append([[-1, -1]])
                    else:
                        bert_questiontoken2triple.append(example.questiontoken2triple[idx])
            # print(len(bert_question_dep), len(bert_question_tokens))
            # assert len(bert_question_dep) == len(bert_question_tokens)
            # assert len(bert_questiontoken2triple) == len(bert_question_tokens)

            tok_to_orig_index = [] # BERT分词后，每个位置对应于原来分词的位置
            orig_to_tok_index = [] # 原始的token对应BERT分词后的索引
            bert_passage_tokens = [] # BERT分词后的token

            for (i, token) in enumerate(passage_tokens):
                orig_to_tok_index.append(len(bert_passage_tokens))
                if token.strip() == '':
                    sub_tokens = ['']
                else:
                    sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    bert_passage_tokens.append(sub_token)

            doc_org_to_split_map = {} # 原始分词的tokenid对应进行word piece后拆分的token的范围
            pre_tok_len = 0
            for idx, doc_token in enumerate(passage_tokens):
                if doc_token.strip() == '':
                    sub_doc_tok = ['']
                else:
                    sub_doc_tok = tokenizer.tokenize(doc_token)
                doc_org_to_split_map[idx] = (pre_tok_len, len(sub_doc_tok) + pre_tok_len - 1)
                pre_tok_len += len(sub_doc_tok)

            bert_passage_dep = [] # BERT分词后passage对应每个token的依存关系
            bert_passagetoken2triple = [] # BERT分词后passage对应每个token对应的三元组
            # 将原始的每个token对应的特征复制到对应的所有拆分的token
            for idx in range(len(passage_tokens)):
                head_start, head_end = doc_org_to_split_map[idx]  # 原始token对应新的word piece的区间范围
                for _ in range(head_end - head_start + 1):
                    bert_passage_dep.append(passage_dep[idx])
                    bert_passagetoken2triple.append(example.passagetoken2triple[idx])

            if len(bert_question_tokens) > max_query_length:
                bert_question_tokens = bert_question_tokens[0:max_query_length]

            tok_start_position = None
            tok_end_position = None
            if is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(passage_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(bert_passage_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    bert_passage_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(bert_question_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _PassSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            bert_passage_spans = []
            start_offset = 0
            # 假设模型最大长度限制是384，减去3个特殊字符，假设question的长度是81，则还有300个字符可以供passage使用
            # 假设此时passage长度是1000，很显然300个字符是放不下的，因此采用滑动窗口，窗口大小为此时的300，滑动的步长是stride=128
            # 最终得到的每个chunk对应的span（起始位置，窗口大小）为[(0, 300), (128, 300), (256, 300), (384, 300), (512, 300), (640, 300), (768, 232)]
            # 每一个chunk与question组合起来作为一个样本，因此原始的passage可能对应多个chunk，即多个样本
            while start_offset < len(bert_passage_tokens):
                length = len(bert_passage_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                bert_passage_spans.append(_PassSpan(start=start_offset, length=length))
                if start_offset + length == len(bert_passage_tokens):
                    break
                start_offset += min(length, doc_stride)

            # 需要说明的是：passage并非直接喂入模型，而是根据上面几行代码划分多个chunk，此处遍历每个chunk，并与question结合
            # 并添加包括CLS和SEP等特殊字符，构成一个新的样本，这样可以保证每个样本的长度在max_seq_len(384)之内
            # 因此我们在划分chunk之后，对新的样本的任意两个token去计算最短路径
            for (doc_span_index, doc_span) in enumerate(bert_passage_spans): # 遍历passage的每个chunk
                tokens = [] # 输入数据的每个token
                tokens_to_triple = [] # 输入数据每个token对齐的三元组
                token_to_orig_map = {}
                token_is_max_context = {}
                # use this idx list to select from doc span mask
                head_select_idx = []
                segment_ids = []
                tokens.append("[CLS]") # BERT分词中的第一个字符默认是[CLS]
                tokens_to_triple.append([[-1, -1]])
                segment_ids.append(0) # BERT的segment embedding所用（不同的句子对应一个segment id）

                # 第一个片段存放passage
                for i in range(doc_span.length): # 当前chunk内token的个数
                    split_token_index = doc_span.start + i
                    # 在新的tokens序列中，每个位置的token对应于最原始的passage的token的id索引
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(bert_passage_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context # 在新的tokens序列中，当前的token是否有最大的context
                    tokens.append(bert_passage_tokens[split_token_index]) # 新的tokens序列中添加一个bert的passage token
                    tokens_to_triple.append(bert_passagetoken2triple[split_token_index]) # 将对应的token对齐的三元组新增到输入token中
                    head_select_idx.append(split_token_index)

                    segment_ids.append(0)

                tokens.append("[SEP]")
                tokens_to_triple.append([[-1, -1]])
                segment_ids.append(0)
                start_doc_ix = head_select_idx[0]
                end_doc_ix = head_select_idx[-1]
                select_doc_len = end_doc_ix - start_doc_ix + 1
                select_que_len = len(bert_question_tokens)
                assert len(head_select_idx) == select_doc_len

                for idx, token in enumerate(bert_question_tokens):
                    tokens.append(token)
                    if idx >= len(bert_questiontoken2triple):
                        tokens_to_triple.append(bert_questiontoken2triple[-1])
                    else:
                        tokens_to_triple.append(bert_questiontoken2triple[idx])
                    segment_ids.append(1)

                tokens.append("[SEP]")
                tokens_to_triple.append([[-1, -1]])
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens) # 将token转换为id数值化操作


                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids) # 用于判断是否是原始值还是padding的部分

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                ### 此处开始两两token遍历，并根据其对齐的结构化三元组获取最短路径
                #（决定：不在此处执行，而转移到batch训练时执行） tokens_rel_path, tokens_ent_path = self.get_tokens_path(bert_refine_triples_ids, bert_refine_triples_with_cls_ids, entity_ids, entpair2tripleid, input_ids, tokens_to_triple, tokenizer, ents_path_dict)

                # assert len(input_ids) == max_seq_length
                # assert len(input_mask) == max_seq_length
                # assert len(segment_ids) == max_seq_length
                # assert len(tokens) == len(tokens_to_triple)
                # assert len(tokens_rel_path) == len(tokens_ent_path)

                start_position = None
                end_position = None
                is_impossible = example.is_impossible

                if is_training and not example.is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    # 判断当前的chunk内是否包含答案的span
                    if (example.start_position < doc_start or
                            example.end_position < doc_start or
                            example.start_position > doc_end or example.end_position > doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        is_impossible = True

                    else:
                        # doc_offset = len(query_tokens) + 2
                        doc_offset = 1  # [CLS]
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                        is_impossible = False

                if is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0
                    is_impossible = True
                # 每个样本对应一个输入特征
                features.append(
                    InputFeatures( # 用类对象很占内存！
                        unique_id=unique_id, # chunk样本的唯一编号
                        example_index=example_index, # 原始样本的唯一编号
                        doc_span_index=doc_span_index, # chunk编号
                        tokens=tokens, # chunk后的输入word piece token序列
                        token_to_orig_map=token_to_orig_map,  # map字典， key表示BERT分词后每个token的id，value表示key对应原始文本token的id
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids, # chunk后的输入word piece token的id序列
                        input_mask=input_mask, # mask矩阵用于判断是否是padding的部分
                        segment_ids=segment_ids, # 分段编号
                        start_position=start_position, # 正确答案的起始位置
                        end_position=end_position, # 正确答案的终止位置
                        is_impossible=is_impossible, # 答案是否不可回答
                        # input_rel_path=tokens_rel_path, # 任意两个token之间在图谱上的推理的关系路径
                        # input_ent_path=tokens_ent_path # 任意两个token之间在图谱上的推理的实体路径
                        ## 经过反复测试，如果直接保存tokens_rel_path和tokens_ent_path，内存占用巨大，因此
                        ## 决定保存生成路径的中间变量，在每次batch训练时进行路径生成。下面为生成路径的中间变量
                        bert_refine_triples_ids=bert_refine_triples_ids,
                        bert_refine_triples_with_cls_ids=bert_refine_triples_with_cls_ids,
                        entity_ids=entity_ids,
                        entpair2tripleid=entpair2tripleid,
                        tokens_to_triple=tokens_to_triple
                    ))

                unique_id += 1

        np.save('unique_id.npy', unique_id)
        return features

    def find_shortest_path_and_convert_to_tensor(self, refine_triples_ids, refine_triples_with_cls_ids, entity_ids,
                                                 entpair2tripleid, cls_id):
        # graph: [[head, rel, tail], ...]
        cls_str = int_list_to_str([cls_id])
        # path_dicts = []
        entity2id_list = []
        input_rel_paths, input_ent_paths, input_rel_path_lengths, input_ent_path_lengths, input_path_masks = [], [], [], [], []

        for batch_i in range(len(refine_triples_ids)):  # 遍历每个batch样本

            input_rel_path = [0] * args.max_ent_num * args.max_ent_num * args.path_length * args.max_token_size
            input_ent_path = [0] * args.max_ent_num * args.max_ent_num * args.path_length * args.max_token_size
            input_rel_path_length = [0] * args.max_ent_num * args.max_ent_num
            input_ent_path_length = [0] * args.max_ent_num * args.max_ent_num
            input_path_mask = [0] * args.max_ent_num * args.max_ent_num

            second_col_num = args.max_ent_num * args.path_length * args.max_token_size  # 实际的矩阵某一行元素的个数
            third_col_num = args.path_length * args.max_token_size

            ent_pair = [(int_list_to_str(head), int_list_to_str(tail)) for head, _, tail in
                        refine_triples_ids[batch_i][0]]
            # ent_with_cls_pair = [(int_list_to_str(head), int_list_to_str(tail)) for head, _, tail in refine_triples_with_cls_ids]
            G1 = nx.Graph()
            G1.add_edges_from(ent_pair)
            nx.draw(G1, with_labels=True)
            entity2id = dict() # 每个entity对应的编号
            entity_list = [cls_str] # 第一个实体默认是CLS
            entity2id[cls_str] = 0
            for i in list(entity_ids[batch_i][0].keys())[:args.max_ent_num - 1]:
                entity_list.append(i)
                if i not in entity2id.keys():
                    entity2id[i] = len(entity2id)
            if len(entity_list) < args.max_ent_num:
                entity_list += [cls_str] * (args.max_ent_num - len(entity_list))

            assert len(entity_list) == args.max_ent_num

            path_dict = dict()
            for i in range(len(entity_list)):
                for j in range(i, len(entity_list)):
                    if i == j:
                        ent_path = [str_to_int_list(entity_list[i]), str_to_int_list(entity_list[i])]
                        rel_path = [[cls_id]]
                    elif entity_list[i] != cls_str and entity_list[j] != cls_str and nx.has_path(G1, entity_list[i], entity_list[j]):
                        ent_path = nx.shortest_path(G1, source=entity_list[i], target=entity_list[j])
                        rel_path = find_Rel_Path(ent_path, entpair2tripleid[batch_i][0],
                                                 refine_triples_with_cls_ids[batch_i][0])
                        ent_path = [str_to_int_list(k) for k in ent_path]
                    else: # 当两个实体不连通时，再通过额外添加的CLS结点计算路径
                        ent_path = [str_to_int_list(entity_list[i]), [cls_id], str_to_int_list(entity_list[j])]
                        rel_path = [[cls_id], [cls_id]]

                    # ent_path = ent_path[:args.path_length]
                    rel_path = rel_path[:args.path_length]
                    reverse_rel_path = ent_path[-1 * args.path_length:]
                    # reverse_ent_path = rel_path[-1 * args.path_length:]
                    reverse_rel_path.reverse()
                    # reverse_ent_path.reverse()

                    rel_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                                rel_path[:args.path_length]] \
                               + [[0] * args.max_token_size] * max(0, args.path_length - len(rel_path))
                    # ent_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                    #             ent_path[:args.path_length]] \
                    #            + [[0] * args.max_token_size] * max(0, args.path_length - len(ent_path))

                    reverse_rel_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                                        reverse_rel_path[:args.path_length]] \
                                       + [[0] * args.max_token_size] * max(0, args.path_length - len(reverse_rel_path))
                    # reverse_ent_path = [i[:args.max_token_size] + [0] * max(0, args.max_token_size - len(i)) for i in
                    #                     reverse_ent_path[:args.path_length]] \
                    #                    + [[0] * args.max_token_size] * max(0, args.path_length - len(reverse_ent_path))

                    rel_path_ = [m for n in rel_path for m in n]
                    reverse_rel_path_ = [m for n in reverse_rel_path for m in n]
                    # ent_path_ = [m for n in ent_path for m in n]
                    # reverse_ent_path_ = [m for n in reverse_ent_path for m in n]

                    replace_ij_start = i * second_col_num + j * third_col_num
                    replace_ij_end = i * second_col_num + (j + 1) * third_col_num
                    replace_ji_start = j * second_col_num + i * third_col_num
                    replace_ji_end = j * second_col_num + (i + 1) * third_col_num

                    input_rel_path[replace_ij_start:replace_ij_end] = rel_path_
                    input_rel_path[replace_ji_start:replace_ji_end] = reverse_rel_path_
                    # input_ent_path[replace_ij_start:replace_ij_end] = ent_path_
                    # input_ent_path[replace_ji_start:replace_ji_end] = reverse_ent_path_

                    input_path_mask[i * args.max_ent_num + j] = 1
                    input_path_mask[j * args.max_ent_num + i] = 1

                    input_rel_path_length[i * args.max_ent_num + j] = np.sum(
                        np.sum(rel_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                    input_rel_path_length[j * args.max_ent_num + i] = np.sum(
                        np.sum(reverse_rel_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                    # input_ent_path_length[i * args.max_ent_num + j] = np.sum(
                    #     np.sum(ent_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence
                    # input_ent_path_length[j * args.max_ent_num + i] = np.sum(
                    #     np.sum(reverse_ent_path, -1) != 0)  # 统计每一个路径对应的实际长度，用于GRU的pack_padded_sequence

                # path_dict[(entity_list[i], entity_list[j])] = [ent_path, rel_path]
            # path_dicts.append(path_dict)
            input_rel_paths.append(input_rel_path)
            # input_ent_paths.append(input_ent_path)
            input_rel_path_lengths.append(input_rel_path_length)
            # input_ent_path_lengths.append(input_ent_path_length)
            input_path_masks.append(input_path_mask)
            entity2id_list.append(entity2id)

        input_rel_paths = torch.tensor(input_rel_paths, dtype=torch.long).view(
            [-1, args.max_ent_num, args.max_ent_num, args.path_length, args.max_token_size])
        # input_ent_paths = torch.tensor(input_ent_paths, dtype=torch.long).view(
        #     [-1, args.max_ent_num, args.max_ent_num, args.path_length, args.max_token_size])
        input_rel_path_lengths = np.reshape(input_rel_path_lengths,
                                            [-1, args.max_ent_num, args.max_ent_num]).tolist()
        # input_ent_path_lengths = np.reshape(input_ent_path_lengths,
        #                                     [-1, args.max_ent_num, args.max_ent_num]).tolist()
        input_path_masks = torch.tensor(input_path_masks, dtype=torch.long).view(
            [-1, args.max_ent_num, args.max_ent_num])

        return input_rel_paths, input_rel_path_lengths, input_path_masks, entity2id_list


    def get_tokens_to_entity(self, refine_triples_with_cls_ids, tokens_to_triple, entity2id_list):
        '''
        根据所有的实体，以及所有token，获得tokeni和tokenj之间的mask值，并将其指向知识图谱中实体对路径的索引
        '''
        input_token_to_entitys, input_path_masks = [], []

        for batch_i in range(len(entity2id_list)):

            # input_rel_path = [[[[0] * args.max_token_size] * args.path_length] * args.max_seq_length] * args.max_seq_length
            # input_ent_path = [[[[0] * args.max_token_size] * args.path_length] * args.max_seq_length] * args.max_seq_length
            # input_rel_path_length = [[0] * args.max_seq_length] * args.max_seq_length
            # input_ent_path_length = [[0] * args.max_seq_length] * args.max_seq_length
            # input_path_mask = [[0] * args.max_seq_length] * args.max_seq_length

            ## trick：按理来说应该初始化一个维度为[384,384,5,5]的矩阵（如上面几行），但是测试发现很耗时间，所以先初始化一个384x384x5x5个元素的以为向量
            ## 这样速度很快，后面赋值时直接通过为止索引推导一下，转换为tensor后再通过view进行维度变换
            input_token_to_entity = [0] * args.max_seq_length * args.max_seq_length
            input_path_mask = [0] * args.max_seq_length * args.max_seq_length

            second_col_num = args.max_seq_length * args.path_length * args.max_token_size # 实际的矩阵某一行元素的个数
            third_col_num = args.path_length * args.max_token_size

            # 用于暂存当前的tokeni与tokenj的信息，如果下一个tokenj与前一个一样与相同的实体存在对齐，则直接从这里读取，避免重复计算
            token_ij_align = None
            for token_idx, tokeni2triple in enumerate(tokens_to_triple[batch_i]):
                if tokeni2triple[0] != [[-1, -1]]: # 说明当前的token出现在三元组内，则该token参与attention计算
                    for idx, tokenj2triple in enumerate(tokens_to_triple[batch_i][token_idx:]):
                        token_jdx = token_idx + idx
                        if tokenj2triple != [[-1, -1]]: # 说明此时两个token都对应图中的三元组，一定存在路径，mask为1

                            # token只会唯一属于某一个实体或关系边，只不过这个实体或关系可能存在于多个三元组，因此只需要随便取一个肯定是对应的
                            # 唯一的三元组
                            posi = tokeni2triple[0][1] if tokeni2triple[0][1] in [0, 2] else 0 # 如果值是0或2，说明其属于实体，否则1为关系，当取1时，我们认为定义其补全至整个三元组，pos为0
                            posj = tokenj2triple[0][1] if tokenj2triple[0][1] in [0, 2] else 2
                            start = int_list_to_str(refine_triples_with_cls_ids[batch_i][0][tokeni2triple[0][0]][posi]) # 路径的起始实体(实体包含的word piece token id序列并以下划线相隔， 例如343_54_3232)
                            end = int_list_to_str(refine_triples_with_cls_ids[batch_i][0][tokenj2triple[0][0]][posj]) # 路径的终点实体
                            startid, endid = 0, 0
                            if start in entity2id_list[batch_i].keys():
                                startid = entity2id_list[batch_i][start]
                            if end in entity2id_list[batch_i].keys():
                                endid = entity2id_list[batch_i][end]

                            input_token_to_entity[token_idx * args.max_seq_length + token_jdx] = startid * args.max_ent_num + endid
                            input_token_to_entity[token_jdx * args.max_seq_length + token_idx] = endid * args.max_ent_num + startid

                            input_path_mask[token_idx * args.max_seq_length + token_jdx] = 1
                            input_path_mask[token_jdx * args.max_seq_length + token_idx] = 1

            input_token_to_entitys.append(input_token_to_entity)
            input_path_masks.append(input_path_mask)
        input_token_to_entitys = torch.tensor(input_token_to_entitys, dtype=torch.long).view([-1, args.max_seq_length, args.max_seq_length])
        input_path_masks = torch.tensor(input_path_masks, dtype=torch.long).view([-1, args.max_seq_length, args.max_seq_length])

        return input_token_to_entitys, input_path_masks


    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _check_is_max_context(self, bert_passage_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""
        # 判断给定的token其是对应的上下文context是否是最大的，具体的例子描述如下所示：

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(bert_passage_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index


















