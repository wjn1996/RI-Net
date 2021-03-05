# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 2.0"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from spacy.lang.en import English
from configure import args
import collections
import logging
import json
import math
import spacy
import os
import random
import pickle
from tqdm import tqdm, trange
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertInferenceQuestionAnswerSpanModule
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from DataLoader import SquadExample, SquadLoader
from utils.utils import find_Shortest_Path, int_list_to_str, str_to_int_list

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True



def is_whitespace(c): # 判断token是否是空格
    if c == " " or c == " or c ==\t" or c == "\r" "\n" or ord(c) == 0x202F:
        return True
    return False

# spacy分词工具——专门用来分句
class SimpleNlp(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner', 'textcat'])
        # self.nlp = spacy.load('en_core_web_sm')
        # self.nlp.tokenizer = my_en_tokenizer(self.nlp)

        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        # self.nlp = nltk.data.load('tokenizers/punkt/english.pickle').tokenize

        # self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'), first=True)  # 将分句放在pipeline的开始

    def nlp(self, text):
        return self.nlp(text)


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", "choice_logits"])

# 启发式的判断答案是否存在并记录答案的结果
def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    # all_examples：验证集；all_features：预处理后的验证集（ground-truth）；all_results：模型预测结果
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list) # 创建一个map，每个value是一个初始化的空列表
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples): # 遍历每个样本
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            if pred.feature_index >= len(features):
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.passage_tokens[orig_doc_start:(orig_doc_end + 1)]
                orig_tokens = [i[1] for i in orig_tokens]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest: # nbest为空列表
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

# warmup线性控制学习率：当
def warmup_linear(x, warmup=0.002):
    # 当最初开始训练是x < warmup，
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed) # 统一设置随机种子
    np.random.seed(args.seed) # 统一设置随机种子
    torch.manual_seed(args.seed) # 统一设置随机种子
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    os.makedirs(args.output_dir, exist_ok=True) # 本地创建一个输出目录

    # 下载BERT相关模型和参数文件：args.bert_model表示bert模型类型（例如“bert-base-cased”）
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    cls_str = int_list_to_str([cls_id])

    # print('tokenizer=', tokenizer)
    train_examples = None
    num_train_steps = None
    squadloader = SquadLoader() # SQuAD的数据加载以及提炼三元组
    if args.do_train:
        print("reading train...")
        # 读取训练集，若本地不存在，则重新加载
        train_example_file = os.path.join(args.squad_data_dir, 'squad_train_example_file.pkl')
        if os.path.exists(train_example_file):
            with open(train_example_file, "rb") as reader:
                train_examples = pickle.load(reader)
            # train_examples = np.load(train_example_file, allow_pickle=True)[()]
        else:
            train_examples = squadloader.read_squad_examples(is_training=True)
            with open(train_example_file, "wb") as writer:
                pickle.dump(train_examples, writer)
            # np.save(train_example_file, train_examples)
        # 总共要训练的次数=样本数/每个batch样本数*epoch数
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model 根据下载好的模型，加载预训练的原始模型
    # BertForQuestionAnsweringSpanMask自定义一个基于BERT的模型，其继承了原始的BERT预训练模型，并调用对应的from_pretrained方法
    # model = BertForQuestionAnsweringSpanMask.from_pretrained(args.bert_model,
    #                                                        cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
    #                                                        args.local_rank))


    model = BertInferenceQuestionAnswerSpanModule.from_pretrained("../../../bert-large-cased-whole-word-masking/",
                                                             cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                 args.local_rank))



    # 如果要加载之前训练好的模型继续训练，则执行下面这个代码，否则注释掉
    output_model_file = os.path.join(args.output_dir, "squad_epoch_0_pytorch_model.bin")
    model_state_dict = torch.load(output_model_file)
    model = BertInferenceQuestionAnswerSpanModule.from_pretrained(args.bert_model, state_dict=model_state_dict)


    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size() # 分布式训练
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    # squadloader.test()

    global_step = 0
    TrainLoss = []
    if args.do_train:
        cached_train_features_file = os.path.join(args.squad_data_dir, 'squad_train_features_file.pkl')
        # 如果数据没有保存在本地，则重新进行数据处理，否则加载
        if os.path.exists(cached_train_features_file):
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
            # train_features = np.load(cached_train_features_file, allow_pickle=True)
        else:
            # 将融入结构化知识的数据进行BERT分词，然后将每个分词进行序列化，并根据结构化知识将每个token之间构建最短路径
            train_features = squadloader.convert_examples_to_features(
                examples=train_examples, # 训练集数据
                tokenizer=tokenizer, # BERT的分词器
                max_seq_length=args.max_seq_length, # word piece的总长度
                doc_stride=args.doc_stride, # 将passage分成若干块，每块长度128
                max_query_length=args.max_query_length, # question的总长度
                is_training=True)

            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)

            # if not features:
            #     with open(cached_train_features_file, "rb") as reader:
            #         train_features = pickle.load(reader)
            #     # train_features = np.load(cached_train_features_file, allow_pickle=True)[()].tolist()
            #     with open(cached_train_features_file, "wb") as writer:
            #         pickle.dump(train_features + features, writer)
            #     # np.save(cached_train_features_file, train_features + features)
            #     train_features = train_features + features
            # else:
            #     with open(cached_train_features_file, "rb") as reader:
            #         train_features = pickle.load(reader)
            #     # train_features = np.load(cached_train_features_file, allow_pickle=True)[()]

        train_features = train_features[72000:]

        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long) # 输入word piece token序列
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long) # mask（没课token对应是否是padding）
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long) # segment编号
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long) # answer起始位置
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long) # answer终止位置
        all_is_impossibles = torch.tensor([int(f.is_impossible) for f in train_features], dtype=torch.long) # 是否不可回答
        # all_input_rel_path_dict = [f.input_rel_path for f in train_features] # dict 每两个token之间的关系边路径
        # all_input_ent_path_dict = [f.input_ent_path for f in train_features] # dict 每两个token之间的实体边路径
        ### 以下为为了提炼结构化知识（token与token之间的推理路径）的中间变量
        all_bert_refine_triples_ids = [f.bert_refine_triples_ids for f in train_features]
        all_bert_refine_triples_with_cls_ids = [f.bert_refine_triples_with_cls_ids for f in train_features]
        all_entity_ids = [f.entity_ids for f in train_features]
        all_entpair2tripleid = [f.entpair2tripleid for f in train_features]
        all_tokens_to_triple = [f.tokens_to_triple for f in train_features]
        ### 以上为为了提炼结构化知识（token与token之间的推理路径）的中间变量

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long) # 样本编号
        # dataset类自动加载为训练集对象
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_is_impossibles,
                                   all_example_index)
        # 生成采样器
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        # 使用dataloader类生成训练集的迭代器
        train_dataloader = DataLoader(train_data, shuffle=False, sampler=train_sampler,
                                      batch_size=args.train_batch_size,
                                      num_workers=12, pin_memory=True)
        step = 0
        # 每一轮训练
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            train_loss = 0
            show_loss_every = 100 # 每隔一定次数保存一下当前的训练loss
            loss_trend = []
            save_model_every = 6000 # 每隔一定次数保存一下模型参数
            empty_cache_every = 1200 # 每隔一定次数清理一下缓存

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ncols=50)):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                # input_ids：输入sequence的wordid序列
                # input_mask：输入sequence的mask矩阵
                # segment_ids：BERT对多个句子的输入时对不同的句子设定不同的id
                # start_positions，end_positions：正确答案ground-truth的起始位置
                # is_impossible：答案是否是存在的
                # example_indices：每个样本对应的编号
                input_ids, input_mask, segment_ids, start_positions, end_positions, \
                is_impossibles, example_indices = batch

                #### stage2：前向传播计算
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions, is_impossibles)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    # warmup：一开始使用小的学习率，后逐渐增大学习率。当到达一定程度时，再线性衰减
                    # 参考：https://www.zhihu.com/question/338066667，https://blog.csdn.net/dendi_hust/article/details/104465337
                    # global_step：当前的训练步数，t_total：总共训练的步数（分布式中则平均）
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups: # 动态更新学习率， 参考：https://blog.csdn.net/bc521bc/article/details/85864555
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                train_loss += loss.item()
                if (step + 1) % show_loss_every == 0:
                    loss_trend.append(loss.item())
                if (step + 1) % save_model_every == 0:
                    # Save a trained model
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir,
                                                     "squad_epoch_" + str(epoch) + "_pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    np.save(os.path.join(args.output_dir, "squad_epoch_" + str(epoch) + "_loss_trend.bin"), loss_trend)
                if (step + 1) % empty_cache_every == 0:
                    torch.cuda.empty_cache()

            train_loss = train_loss / (step + 1)
            TrainLoss.append(train_loss)
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "squad_epoch_" + str(epoch) + "_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            np.save(os.path.join(args.output_dir, "squad_epoch_" + str(epoch) + "_loss_trend.bin"), loss_trend)

        with open(os.path.join(args.output_dir, "squad_train_loss.pkl"), 'wb') as f:
            pickle.dump(TrainLoss, f)
    # 预测阶段
    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        print("reading dev...")
        eval_example_file = os.path.join(args.squad_data_dir, 'squad_eval_example_file.pkl')
        cached_eval_features_file = os.path.join(args.squad_data_dir, 'squad_eval_features_file.pkl')

        if os.path.exists(eval_example_file):
            with open(eval_example_file, "rb") as reader:
                eval_examples = pickle.load(reader)
            # train_examples = np.load(train_example_file, allow_pickle=True)[()]
        else:
            eval_examples = squadloader.read_squad_examples(is_training=False)
            with open(eval_example_file, "wb") as writer:
                pickle.dump(eval_examples, writer)

        if os.path.exists(cached_eval_features_file):
            with open(cached_eval_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        else:
            eval_features = squadloader.convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            with open(cached_eval_features_file, "wb") as writer:
                pickle.dump(eval_features, writer)


        logger.info("***** Running evaluating *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)  # 输入word piece token序列
        all_input_mask = torch.tensor([f.input_mask for f in eval_features],
                                      dtype=torch.long)  # mask（没课token对应是否是padding）
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)  # segment编号
        ### 以下为为了提炼结构化知识（token与token之间的推理路径）的中间变量
        all_bert_refine_triples_ids = [f.bert_refine_triples_ids for f in eval_features]
        all_bert_refine_triples_with_cls_ids = [f.bert_refine_triples_with_cls_ids for f in eval_features]
        all_entity_ids = [f.entity_ids for f in eval_features]
        all_entpair2tripleid = [f.entpair2tripleid for f in eval_features]
        all_tokens_to_triple = [f.tokens_to_triple for f in eval_features]
        ### 以上为为了提炼结构化知识（token与token之间的推理路径）的中间变量

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)  # 样本编号
        # dataset类自动加载为训练集对象
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)


        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size,
                                     num_workers=12, pin_memory=True)

        # with open(os.path.join(args.output_dir, "squad_train_loss.pkl"), 'rb') as f:
        #     TrainLoss = pickle.load(f)

        all_results = []
        # 训练阶段是每个epoch保存一次模型，因此在预测阶段也将对每个模型进行一次预测
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # train_loss = TrainLoss[epoch]
            # 将第epoch次训练的模型加载
            output_model_file = os.path.join(args.output_dir, "squad_epoch_" + str(epoch) + "_pytorch_model.bin")
            model_state_dict = torch.load(output_model_file)
            model = BertInferenceQuestionAnswerSpanModule.from_pretrained(args.bert_model, state_dict=model_state_dict)
            model.to(device)
            model.eval()
            model.half()

            logger.info("Start evaluating")
            for batch in tqdm(eval_dataloader, desc="Evaluating",ncols=50):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                if len(all_results) % 1000 == 0:
                    logger.info("Processing example: %d" % (len(all_results)))

                input_ids, input_mask, segment_ids, example_indices = batch
                slices = example_indices.cpu().numpy().tolist()  # 所有feature样本的下标

                bert_refine_triples_ids = [all_bert_refine_triples_ids[i] for i in slices]
                bert_refine_triples_with_cls_ids = [all_bert_refine_triples_with_cls_ids[i] for i in slices]
                entity_ids = [all_entity_ids[i] for i in slices]
                entpair2tripleid = [all_entpair2tripleid[i] for i in slices]
                tokens_to_triple = [all_tokens_to_triple[i] for i in slices]

                #### stage1: Refining Stage

                # 对batch内的每个样本，首先提炼结构化知识，并基于此实现计算每两个结点之间的最短推理路径
                input_kg_rel_path, input_kg_rel_path_length, \
                input_kg_path_mask, entity2id_list = \
                    squadloader.find_shortest_path_and_convert_to_tensor(bert_refine_triples_ids,
                                                                         bert_refine_triples_with_cls_ids,
                                                                         entity_ids, entpair2tripleid, cls_id)

                input_token_to_entity, input_path_mask = squadloader.get_tokens_to_entity(
                    bert_refine_triples_with_cls_ids, tokens_to_triple, entity2id_list)

                input_kg_rel_path = input_kg_rel_path.to(device)
                input_kg_path_mask = input_kg_path_mask.to(device)
                input_token_to_entity = input_token_to_entity.to(device)
                input_kg_rel_path_length = torch.LongTensor(input_kg_rel_path_length).to(device)
                input_path_mask = input_path_mask.to(device)

                with torch.no_grad():
                    # 预测时，不提供ground-truth，直接返回start和end的概率分布
                    batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask, input_kg_rel_path,
                                 input_kg_rel_path_length, input_token_to_entity,
                                 input_path_mask)
                for i, example_index in enumerate(example_indices): # 遍历batch的每个样本
                    # start位置的结果
                    start_logits = batch_start_logits[i].detach().cpu().tolist() # [batch_size, seq_len]
                    # end位置的预测结果
                    end_logits = batch_end_logits[i].detach().cpu().tolist() # [batch_size, seq_len]
                    eval_feature = eval_features[example_index.item()] # 获得对应ground-truth的样本
                    unique_id = int(eval_feature.unique_id)
                    # 每个预测的样本通过unique_id与ground-truth的样本对应
                    all_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits,
                                                 choice_logits=end_logits))
                # print('len(all_results)={}'.format(len(all_results)))

            output_prediction_file = os.path.join(args.output_dir, "squad_epoch_" + str(epoch) + "_predictions.json")
            output_nbest_file = os.path.join(args.output_dir, "squad_epoch_" + str(epoch) + "_nbest_predictions.json")
            output_null_log_odds_file = os.path.join(args.output_dir, "squad_epoch_" + str(epoch) + "_null_odds.json")
            # 根据预测的结果，启发式寻找答案或判断是否无答案，将预测的结果保存到本地
            # eval_examples：验证集；eval_features：预处理后的验证集（ground-truth）；all_results：模型预测结果
            np.save("result.npy", all_results)
            logger.info("Processing example: %d" % (len(all_results)))
            write_predictions(eval_examples, eval_features, all_results,
                              args.n_best_size, args.max_answer_length,
                              args.do_lower_case, output_prediction_file,
                              output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                              True, args.null_score_diff_threshold)

            # result = {'train_loss': train_loss}

            with open(os.path.join(args.output_dir, "result.txt"), "a") as writer:
                writer.write("\t\n***** Eval results Epoch %d  %s *****\t\n" % (epoch, time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                # for key in sorted(result.keys()):
                #     logger.info("  %s = %s", key, str(result[key]))
                #     writer.write("%s = %s\t" % (key, str(result[key])))
                #     writer.write("\t\n")

def check_data():
    # 本函数用于检测需要包含的数据是否存在，不存在则相应执行对应的函数，或提示需要下载文件
    train_struct_file = os.path.join(args.squad_data_dir, 'squad_train_structure_knowledge.npy')
    dev_struct_file = os.path.join(args.squad_data_dir, 'squad_dev_structure_knowledge.npy')
    if not os.path.exists(args.train_file) or not os.path.exists(args.predict_file):
        raise FileExistsError(
            "You Need to download 'train-v2.0.json' and 'dev-v2.0.json' from squad website, and then put them into directory '/data/squad/'")

    if not os.path.exists(train_struct_file) or not os.path.exists(dev_struct_file):
        raise FileExistsError("You Need to generate 'squad_train_structure_knowledge.npy' and 'squad_dev_structure_knowledge.npy', you can refer to directory /utils/auto_extractor/readme.txt")



if __name__ == "__main__":
    check_data()
    main()
