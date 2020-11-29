from __future__ import print_function
#from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordsequence import WordSequence
from .crf import CRF
from .typechain import TypeChain
from .attentionlayer import AttFlowLayer



class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.use_crf = data.use_crf
        print("build network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

        # type embedding
        self.type_size = data.type_alphabet_size-1
        self.type_dim = data.HP_hidden_dim
        self.type_embedding = nn.Embedding(self.type_size, self.type_dim)
        self.type_embedding.weight.data.copy_(torch.from_numpy(
            self.random_embedding_label(self.type_size, self.type_dim, data.label_embedding_scale)))#data.label_alphabet_size

        self.type_embedding = self.type_embedding.cuda()
        self.type_embs = self.type_embedding(torch.LongTensor([i for i in range(self.type_size)]).cuda())

        self.attflowlayer = AttFlowLayer(data.HP_hidden_dim)
        self.typchain = TypeChain(data.HP_hidden_dim,data )

        self.attention2hidden2tag = nn.Linear(data.HP_hidden_dim*2, data.HP_hidden_dim)
        self.lstm = nn.LSTM(data.HP_hidden_dim, data.HP_hidden_dim, num_layers=1, batch_first= True, bidirectional=True)
        self.droplstm = nn.Dropout(data.HP_dropout)

        self.hidden2tag_sl_1 = nn.Linear(data.HP_hidden_dim*2, data.label_alphabet_size)
        if self.gpu:
            self.hidden2tag_sl_1 = self.hidden2tag_sl_1.cuda()
            self.typchain = self.typchain.cuda()
            self.attflowlayer = self.attflowlayer.cuda()
            self.lstm = self.lstm.cuda()
            self.droplstm = self.droplstm.cuda()
            self.attention2hidden2tag = self.attention2hidden2tag.cuda()


    def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, batch_type,mask_type,type_seq_lengths,word_seq_bert_tensor):
        outs, hiddens = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,word_seq_bert_tensor)
        out_context, out_type = self.attflowlayer(outs,self.type_embs)
        type_loss, type_tag_seq = self.typchain.neg_log_likelihood_loss(out_type, type_seq_lengths,mask_type,batch_type)
        out_context = self.attention2hidden2tag(out_context)
        out_context = pack_padded_sequence(input=out_context, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        hidden = None
        lstm_out, hidden = self.lstm(out_context, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        out_context = self.droplstm(lstm_out.transpose(1,0))

        outs = self.hidden2tag_sl_1(out_context)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            label_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs)
            label_loss = loss_function(score, batch_label.view(batch_size * seq_len).long())
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if True:#self.average_batch:
            label_loss = label_loss / batch_size
            #type_loss = type_loss/batch_size
        return label_loss, tag_seq, type_loss,type_tag_seq


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, batch_type,mask_type,type_seq_lengths,word_seq_bert_tensor):
        outs, hiddens = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,word_seq_bert_tensor)


        out_context, out_type = self.attflowlayer(outs,self.type_embs)
        out_context = self.attention2hidden2tag(out_context)

        out_context = pack_padded_sequence(input=out_context, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        hidden = None
        lstm_out, hidden = self.lstm(out_context, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        out_context = self.droplstm(lstm_out.transpose(1,0))


        outs = self.hidden2tag_sl_1(out_context)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            label_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            label_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if True:#self.average_batch:
            label_loss = label_loss / batch_size
        return label_loss,tag_seq


    def random_embedding_label(self, vocab_size, embedding_dim, scale):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        # scale = np.sqrt(3.0 / embedding_dim)
        # scale = 0.025
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb