import torch
import torch.nn as nn
import utils.torch_utils as utils
import utils.constant as constant
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class GCN_BiLSTM(nn.Module):
    def __init__(self, opt, bert_model, bert_tokenizer):
        super(GCN_BiLSTM, self).__init__()
        self.opt = opt
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

        self.pos_embedding = nn.Embedding(len(constant.POS_TO_ID), self.opt['pos_dim']) if self.opt['pos_dim'] > 0 else None
        self.char_embedding = nn.Embedding(86, self.opt['char_dim']) if self.opt['char_dim'] > 0 else None
        self.prot_embedding = nn.Embedding(len(constant.JPROT_TO_ID), self.opt['prot_dim']) if self.opt['prot_dim'] > 0 else None
        self.trig_embedding = nn.Embedding(len(constant.TRIGGER_TO_ID), self.opt['ctrig_dim']) if self.opt['ctrig_dim'] > 0 else None
        if self.opt['char_dim'] > 0:
            self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.opt['char_dim'], kernel_size=(3, self.opt['char_dim']), padding=(2, 0))
        self.in_drop = nn.Dropout(self.opt['input_dropout'])

        input_size = 768
        if self.opt['char_dim'] > 0:
            input_size += self.opt['char_dim']
        if self.opt['pos_dim'] > 0:
            input_size += self.opt['pos_dim']
        if self.opt['prot_dim'] > 0:
            input_size += self.opt['prot_dim']

        self.linear = nn.Linear(input_size, self.opt['rnn_input_dim'])

        # rnn layer
        if self.opt.get('rnn', False):
            self.rnn = nn.LSTM(self.opt['rnn_input_dim'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output

        # transformer layer
        self.encoder_layer = nn.TransformerEncoderLayer(768, 8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 6)
        self.pos_encoder = PositionalEncoding(768, 0.1)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = utils.rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def _get_lstm_features(self, words, char_ids, pos_ids, prot_ids):
        seq_lens = [len(sent.split()) for sent in words]
        cls_embs, word_embs = utils.get_avgword_embs(self.bert_model, self.bert_tokenizer, words, seq_lens)
        embs = [word_embs]
        if self.opt['char_dim'] > 0:
            chars_embeds = self.char_embedding(char_ids)
            chars_embeds = chars_embeds.unsqueeze(1)
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = F.max_pool2d(chars_cnn_out3,  kernel_size=(chars_cnn_out3.size(2), 1)).view(1, chars_cnn_out3.size(0), self.opt['char_dim'])
            embs.append(chars_embeds)
        if self.opt['pos_dim'] > 0:
            pos_embs = self.pos_embedding(pos_ids)
            embs.append(pos_embs)
        if self.opt['prot_dim'] > 0:
            prot_embs = self.prot_embedding(prot_ids)
            embs.append(prot_embs)
        # if self.opt['cjprot_dim'] > 0:
        #     cjprot_embs = self.prot_embedding(cjprot_ids)
        #     embs.append(cjprot_embs)
        # if self.opt['ctrig_dim'] > 0:
        #     ctrig_embs = self.trig_embedding(ctrig_ids)
        #     embs.append(ctrig_embs)

        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        embs  = self.linear(embs)
        # rnn layer
        if self.opt.get('rnn', False):
            lstm_out = self.rnn_drop(self.encode_with_rnn(embs, seq_lens, len(seq_lens)))
        else:
            lstm_out = embs

        return cls_embs, lstm_out

    def _get_transformer_features(self, words):
        seq_lens = [len(sent.split()) for sent in words]
        embs = utils.get_avgword_embs(self.bert_model, self.bert_tokenizer, words, seq_lens)
        embs = self.pos_encoder(embs)
        out = self.transformer_encoder(embs)
        return out