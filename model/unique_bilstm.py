import torch
import torch.nn as nn
import utils.torch_utils as utils
import utils.constant as constant
import torch.nn.functional as F

class Unique_BiLSTM(nn.Module):
    def __init__(self, opt, bert_model, bert_tokenizer):
        super(Unique_BiLSTM, self).__init__()
        self.opt = opt
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.in_drop = nn.Dropout(self.opt['input_dropout'])

        # rnn layer
        self.input_size = 768
        if self.opt.get('rnn', False):
            self.unique_hidden = 115
            self.rnn = nn.LSTM(self.input_size, self.unique_hidden, opt['rnn_layers'], batch_first=True, dropout=opt['rnn_dropout'], bidirectional=True)
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output

        self.linear = nn.Linear(self.input_size, 230)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = utils.rnn_zero_state(batch_size, self.unique_hidden, self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def _get_lstm_features(self, words):
        seq_lens = [len(sent.split()) for sent in words]
        cls_embs, word_embs = utils.get_avgword_embs(self.bert_model, self.bert_tokenizer, words, seq_lens)
        embs = self.in_drop(word_embs)

        # rnn layer
        if self.opt.get('rnn', False):
            lstm_out = self.rnn_drop(self.encode_with_rnn(embs, seq_lens, len(seq_lens)))
        else:
            lstm_out = embs

        return cls_embs, lstm_out

    def forward(self, words):
        seq_lens = [len(sent.split()) for sent in words]
        word_embs = utils.get_avgword_embs(self.bert_model, self.bert_tokenizer, words, seq_lens)
        embs = self.in_drop(word_embs)
        output = self.linear(embs)
        return output