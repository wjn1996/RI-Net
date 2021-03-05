import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from .file_utils import cached_path
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np


class PathEncoder(nn.Module):
    # 对于一个样本，用GRU为relation path进行表示，并得到一个relation embedding
    def __init__(self, config, bidirectional=True):
        super(PathEncoder, self).__init__()
        # self.vocab = vocab  # 词表
        self.hidden_size = config.hidden_size  # 隐状态维度大小(768)
        self.num_layers = 1  # GRU的层数
        self.dropout = 0.2
        self.bidirectional = bidirectional
        # self.bert_embed = BertEmbedding(config) # 根据BERT Encoder输出部分，提取各个token对应的隐状态向量
        self.rel_path_rnn = nn.GRU( # 用于对关系路径进行表示
            input_size=self.hidden_size, # 路径输入部分除了对应的token自身以外，还包括路径上每个部分所有token的均值，所以是两倍的hidden_size
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.,
            bidirectional=bidirectional
        )
        tot_dim = 2 * self.hidden_size if bidirectional else self.hidden_size

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, rel_path_emb, src_lengths):
        # src_tokens和src_lengths的维度是什么？分别代表什么意思？
        # 猜测：src_tokens表示一条token序列组成的relation路径，src_lengths表示对应的长度
        # input: rel_path_emb: [batch_size, max_ent_num, max_ent_num, path_len, hidden_size]
        batch_size, _, __, path_len, hid_dim = rel_path_emb.size()
        rel_path_emb = rel_path_emb.view([-1, path_len, hid_dim]).permute([1, 2, 0])
        src_lengths = src_lengths.view([-1,])
        seq_len, hid_dim, bsz = rel_path_emb.size()
        ###
        sorted_src_lengths, indices = torch.sort(src_lengths, descending=True)  # 最后一个维度上，对所有元素降序排序
        rel_x = rel_path_emb.index_select(2, indices).permute([0, 2, 1])  # 在第3维度上根据索引序列进行检索 [seq_len, bsz, hid_dim]
        ###
        rel_x = F.dropout(rel_x, p=self.dropout, training=self.training)

        # RNN中的pack_padded_sequence
        packed_rel_x = nn.utils.rnn.pack_padded_sequence(rel_x, sorted_src_lengths.data.tolist())

        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        rel_h0 = rel_x.data.new(*state_size).zero_()
        _, rel_final_h = self.rel_path_rnn(packed_rel_x, rel_h0)
        # final_h: [num_layers, batch_size, hidden_size]

        if self.bidirectional:
            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz,
                                                                                                -1)  # [num_layers, batch_size, 2*hidden_size]

            rel_final_h = combine_bidir(rel_final_h)  # [num_layers, batch_size, 2*hidden_size]

        ###
        _, positions = torch.sort(indices)  # 将第一次排序后得到的索引进行排序，相当于恢复原始的索引
        rel_final_h = rel_final_h.index_select(1, positions)  # num_layers x bsz x hidden_size

        # output = 0.5 * self.out_proj(rel_final_h[-1]) + 0.5 * self.out_proj(ent_final_h[-1])  # 取GRU最后一层的输出 [batch_size, hidden_size]
        output = rel_final_h[-1]

        return output  # [batch_size, hidden_size]



# class PathEncoder(nn.Module):
#     # 对于一个样本，用GRU为relation path进行表示，并得到一个relation embedding
#     def __init__(self, config, bidirectional=True):
#         super(PathEncoder, self).__init__()
#         # self.vocab = vocab  # 词表
#         self.hidden_size = config.hidden_size  # 隐状态维度大小(768)
#         self.num_layers = 1  # GRU的层数
#         self.dropout = 0.2
#         self.bidirectional = bidirectional
#         # self.bert_embed = BertEmbedding(config) # 根据BERT Encoder输出部分，提取各个token对应的隐状态向量
#         self.rel_path_rnn = nn.GRU( # 用于对关系路径进行表示
#             input_size=2*self.hidden_size, # 路径输入部分除了对应的token自身以外，还包括路径上每个部分所有token的均值，所以是两倍的hidden_size
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             dropout=self.dropout if self.num_layers > 1 else 0.,
#             bidirectional=bidirectional
#         )
#         self.ent_path_rnn = nn.GRU( # 用于对实体路径进行表示
#             input_size=2*self.hidden_size,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             dropout=self.dropout if self.num_layers > 1 else 0.,
#             bidirectional=bidirectional
#         )
#         tot_dim = 2 * self.hidden_size if bidirectional else self.hidden_size
#         self.out_proj = nn.Linear(2 * tot_dim, self.hidden_size)
#
#     def reset_parameters(self):
#         nn.init.normal_(self.out_proj.weight, std=0.02)
#         nn.init.constant_(self.out_proj.bias, 0.)
#
#     def forward(self, rel_path_emb, src_lengths, ent_path_emb=None):
#         # src_tokens和src_lengths的维度是什么？分别代表什么意思？
#         # 猜测：src_tokens表示一条token序列组成的relation路径，src_lengths表示对应的长度
#         # input: rel_path_emb: [batch_size, max_ent_num, max_ent_num, path_len, hidden_size]
#         batch_size, _, __, path_len, hid_dim = rel_path_emb.size()
#         rel_path_emb = rel_path_emb.view([-1, path_len, hid_dim]).permute([1, 2, 0])
#         if not ent_path_emb:
#             ent_path_emb = ent_path_emb.view([-1, path_len, hid_dim]).permute([1, 2, 0])
#         seq_len, hid_dim, bsz = rel_path_emb.size()
#         ###
#         sorted_src_lengths, indices = torch.sort(src_lengths, descending=True)  # 最后一个维度上，对所有元素降序排序
#         rel_x = rel_path_emb.index_select(2, indices)  # 在第1维度上根据索引序列进行检索
#         if not ent_path_emb:
#             ent_x = ent_path_emb.index_select(2, indices)  # 在第1维度上根据索引序列进行检索
#         ###
#         rel_x = F.dropout(rel_x, p=self.dropout, training=self.training)
#         if not ent_path_emb:
#             ent_x = F.dropout(ent_x, p=self.dropout, training=self.training)
#
#         # RNN中的pack_padded_sequence
#         packed_rel_x = nn.utils.rnn.pack_padded_sequence(rel_x, sorted_src_lengths.data.tolist())
#         if not ent_path_emb:
#             packed_ent_x = nn.utils.rnn.pack_padded_sequence(ent_x, sorted_src_lengths.data.tolist())
#
#         if self.bidirectional:
#             state_size = 2 * self.num_layers, bsz, self.hidden_size
#         else:
#             state_size = self.num_layers, bsz, self.hidden_size
#         rel_h0 = rel_x.data.new(*state_size).zero_()
#         if not ent_path_emb:
#             ent_h0 = ent_x.data.new(*state_size).zero_()
#         _, rel_final_h = self.rel_path_rnn(packed_rel_x, rel_h0)
#         if not ent_path_emb:
#             _, ent_final_h = self.ent_path_rnn(packed_ent_x, ent_h0)
#         # final_h: [num_layers, batch_size, hidden_size]
#
#         if self.bidirectional:
#             def combine_bidir(outs):
#                 return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz,
#                                                                                                 -1)  # [num_layers, batch_size, 2*hidden_size]
#
#             rel_final_h = combine_bidir(rel_final_h)  # [num_layers, batch_size, 2*hidden_size]
#             if not ent_path_emb:
#                 ent_final_h = combine_bidir(ent_final_h)  # [num_layers, batch_size, 2*hidden_size]
#
#         ###
#         _, positions = torch.sort(indices)  # 将第一次排序后得到的索引进行排序，相当于恢复原始的索引
#         rel_final_h = rel_final_h.index_select(1, positions)  # num_layers x bsz x hidden_size
#         if not ent_path_emb:
#             ent_final_h = ent_final_h.index_select(1, positions)  # num_layers x bsz x hidden_size
#
#         # output = 0.5 * self.out_proj(rel_final_h[-1]) + 0.5 * self.out_proj(ent_final_h[-1])  # 取GRU最后一层的输出 [batch_size, hidden_size]
#         if not ent_path_emb:
#             output = self.out_proj(torch.cat([rel_final_h[-1], ent_final_h[-1]], -1))
#         else:
#             output = rel_final_h[-1]
#
#         return output  # [batch_size, hidden_size]

