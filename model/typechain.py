# -*- coding: utf-8 -*-
# @Author: Tao Qian
# @Date:   2020-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 13:22:34
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from .crf import CRF
import torch.nn.functional as F

class TypeChain(nn.Module):
    def __init__(self, lstm_hidden,data , bilstm_flag = True ):
        super(TypeChain, self).__init__()
        self.use_crf = False #data.use_crf

        print("build type prediction net...")

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        type_size = data.type_alphabet_size-1
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=1, batch_first=True, bidirectional=bilstm_flag)
        self.droplstm = nn.Dropout(data.HP_dropout)


        self.hidden2tag_sl_1 = nn.Linear(data.HP_hidden_dim*2, 2)#type_size)
        if self.gpu:
            self.hidden2tag_sl_1 = self.hidden2tag_sl_1.cuda()
            self.droplstm = self.droplstm.cuda()
            self.lstm = self.lstm.cuda()


    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, mask,batch_label):
        lstm_out = pack_padded_sequence(input=word_inputs, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        hidden = None
        lstm_out, hidden = self.lstm(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1,0))
        outs = self.hidden2tag_sl_1(lstm_out)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outs = outs.view(batch_size * seq_len, -1)
        score = F.log_softmax(outs)
        label_loss = loss_function(score, batch_label.view(batch_size * seq_len).long())
        _, tag_seq  = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        if self.average_batch:
            label_loss = label_loss / batch_size
        tag_seq = None
        return label_loss, tag_seq


    def forward(self, word_inputs, word_seq_lengths, mask, batch_label):
        lstm_out = pack_padded_sequence(input=word_inputs, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        hidden = None
        lstm_out, hidden = self.lstm(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)

        outs = self.hidden2tag_sl_1(lstm_out)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        outs = outs.view(batch_size * seq_len, -1)
        _, tag_seq  = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        tag_seq = mask.long() * tag_seq
        return tag_seq