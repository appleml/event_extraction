import torch.nn as nn
import torch
from utils import constant, torch_utils
from model.share_bilstm import Joint_BiLSTM

## Binding event,  判断两个论元是不是在同一个事件中，
class RelaClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, bert_model, bert_tokenizer):
        super().__init__()
        self.opt = opt
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

        self.jbilstm = Joint_BiLSTM(opt, bert_model, bert_tokenizer)
        self.in_drop = nn.Dropout(opt['input_dropout'])
        # rnn layer
        input_size = 768 + self.opt['pos_dim']
        self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, dropout=opt['rnn_dropout'], bidirectional=True)
        self.in_dim = opt['rnn_hidden'] * 2
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output
        self.output = nn.Linear(self.in_dim, 2)

    ## event_exam = (trig_type, trig_typeId, trig_posit, argu1_typeId, prot1_posit, argu2_typeId, prot2_posit)
    def forward(self, words, char_ids, pos_ids, prot_ids, exam_input):
        # rnn layer
        rnn_output = self.jbilstm._get_lstm_features(words, char_ids, pos_ids, prot_ids)
        trig_typeId, trig_posit, argu1_typeId, argu1_posit, argu2_typeId, argu2_posit = exam_input

        trig_mask = trig_posit.eq(0).eq(0).unsqueeze(2)
        argu1_mask = argu1_posit.eq(0).eq(0).unsqueeze(2)
        argu2_mask = argu2_posit.eq(0).eq(0).unsqueeze(2)

        batch_size = argu1_mask.size(0)
        rnn_output = rnn_output.repeat(batch_size, 1, 1)
        trig_emb = torch_utils.pool(rnn_output, trig_mask, type=self.opt['pooling'])
        argu1_emb = torch_utils.pool(rnn_output, argu1_mask, type=self.opt['pooling'])
        argu2_emb = torch_utils.pool(rnn_output, argu2_mask, type=self.opt['pooling'])

        logits = self.output(trig_emb+argu1_emb+argu2_emb)

        return logits

    # def forward(self, words, char_ids, pos_ids, prot_ids, exam_input):
    #     # rnn layer
    #     rnn_output = self.jbilstm._get_lstm_features(words, char_ids, pos_ids, prot_ids)
    #     input_emb = 0
    #     if len(exam_input) == 4:
    #         trig_typeId, trig_posit, argu1_typeId, argu1_posit = exam_input
    #         trig_mask = trig_posit.eq(0).eq(0).unsqueeze(2)
    #         argu1_mask = argu1_posit.eq(0).eq(0).unsqueeze(2)
    #
    #         batch_size = argu1_mask.size(0)
    #         rnn_output = rnn_output.repeat(batch_size, 1, 1)
    #         trig_emb = torch_utils.pool(rnn_output, trig_mask, type=self.opt['pooling'])
    #         argu1_emb = torch_utils.pool(rnn_output, argu1_mask, type=self.opt['pooling'])
    #         input_emb = trig_emb+argu1_emb
    #
    #     elif len(exam_input) == 6:
    #         trig_typeId, trig_posit, argu1_typeId, argu1_posit, argu2_typeId, argu2_posit = exam_input
    #         trig_mask = trig_posit.eq(0).eq(0).unsqueeze(2)
    #         argu1_mask = argu1_posit.eq(0).eq(0).unsqueeze(2)
    #         argu2_mask = argu2_posit.eq(0).eq(0).unsqueeze(2)
    #
    #         batch_size = argu1_mask.size(0)
    #         rnn_output = rnn_output.repeat(batch_size, 1, 1)
    #         trig_emb = torch_utils.pool(rnn_output, trig_mask, type=self.opt['pooling'])
    #         argu1_emb = torch_utils.pool(rnn_output, argu1_mask, type=self.opt['pooling'])
    #         argu2_emb = torch_utils.pool(rnn_output, argu2_mask, type=self.opt['pooling'])
    #         input_emb = trig_emb+argu1_emb+argu2_emb
    #
    #     logits = self.output(input_emb)
    #
    #     return logits

