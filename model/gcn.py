"""
GCN model for relation extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.tree import gen_ret
from utils import constant, torch_utils, common_utils
from model.unique_bilstm import Unique_BiLSTM
from model.gcn_bilstm import GCN_BiLSTM
BertLayerNorm = torch.nn.LayerNorm

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, bert_model, bert_tokenizer, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

        self.gcn_model = GCNRelationModel(opt, bert_model, bert_tokenizer, emb_matrix=emb_matrix)

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    # binding argument
    def bind_argu(self, words, char_ids, pos_ids, prot_ids, parse_inputs, bind_inputs_first, bind_inputs_second):
        logits, pooling_out = self.gcn_model.bind_argument(words, char_ids, pos_ids, prot_ids, parse_inputs, bind_inputs_first, bind_inputs_second)
        return logits, pooling_out

    def forward(self, words, char_ids, pos_ids, prot_ids, parse_inputs, first_entity_pair, second_entity_pair):
        logits, pooling_out = self.gcn_model(words, char_ids, pos_ids, prot_ids, parse_inputs, first_entity_pair, second_entity_pair)
        return logits, pooling_out

class GCNRelationModel(nn.Module):
    def __init__(self, opt, bert_model, bert_tokenizer, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.biffine_hidden = opt['hidden_dim'] + opt['ner_dim'] + 20
        #self.biffine_hidden = opt['hidden_dim']+20
        self.W1 = nn.Parameter(torch.FloatTensor(3, self.biffine_hidden, self.biffine_hidden), requires_grad=True)
        self.weights_init(self.W1)
        self.W2 = nn.Parameter(torch.FloatTensor(2*self.biffine_hidden, 3), requires_grad=True)
        self.weights_init(self.W2)
        self.b = nn.Parameter(torch.FloatTensor(1, 3), requires_grad=True)
        self.weights_init(self.b)

        self.unique_bilstm = Unique_BiLSTM(opt, bert_model, bert_tokenizer)
        # gcn layer
        self.gcn = GCN(opt, bert_model, bert_tokenizer, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim'] * 3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(0.2)
        self.layer_norm = BertLayerNorm(230)

        self.classifier = nn.Linear(230, 200)
        self.func = nn.ReLU()

        self.cls_layer = nn.Linear(768, 230)

        ## bind argument
        self.bind_layer_norm = BertLayerNorm(230)
        self.bind_output = nn.Linear(230, 1)

    def weights_init(self, embed):
        nn.init.kaiming_uniform_(embed.data, mode='fan_in')

    def encode_gcn(self, words, char_ids, pos_ids, prot_ids, parse_inputs):
        _, deprel, head = parse_inputs
        l = [len(x.split()) for x in words]
        maxlen = max(l)

        def inputs_to_tree_reps(head, deprel, l, prune):
            head = head.cpu().numpy()
            deprela = deprel.cpu().numpy()
            adj_set = []
            dep_set = []
            dep_mask_set = []
            for i in range(len(l)):
                sent_adj, sent_depadj, sent_mask = gen_ret(head[i], deprela[i], l[i], directed=False)
                sent_adj = sent_adj.reshape(1, maxlen, maxlen)
                sent_depadj = sent_depadj.reshape(1, maxlen, maxlen)
                sent_mask = sent_mask.reshape(1, maxlen, maxlen)
                adj_set.append(sent_adj)
                dep_set.append(sent_depadj)
                dep_mask_set.append(sent_mask)

            adj = np.concatenate(adj_set, axis=0)
            adj = torch.from_numpy(adj)

            dep_adj = np.concatenate(dep_set, axis=0)
            dep_adj = torch.from_numpy(dep_adj).long()

            dep_mask = np.concatenate(dep_mask_set, axis=0)
            dep_mask = torch.ByteTensor(dep_mask)
            if self.opt['cuda']:
                adj = adj.cuda()
                dep_adj = dep_adj.cuda()
                dep_mask = dep_mask.cuda()

            return adj, dep_adj, dep_mask

        adj, deprel_adj, _ = inputs_to_tree_reps(head.data, deprel.data, l, -1)
        cls_embs, h, pool_mask = self.gcn(adj, deprel_adj, words, char_ids, pos_ids, prot_ids, parse_inputs, l)
        return cls_embs, h, pool_mask

    def bind_argument(self, words, char_ids, pos_ids, prot_ids, parse_inputs, bind_inputs_first, bind_inputs_second):
        cls_embs, h, pool_mask = self.encode_gcn(words, char_ids, pos_ids, prot_ids, parse_inputs)
        first_argu1_type_id, first_argu1_mask, first_argu2_type_id, first_argu2_mask = bind_inputs_first

        batch_size = first_argu1_type_id.size(0)
        h = h.repeat(batch_size, 1, 1)
        cls_embs = cls_embs.repeat(batch_size, 1)
        cls_embs = self.cls_layer(cls_embs)

        h_out = torch_utils.pool(h, pool_mask, type=self.opt['pooling'])
        #first_trig_emb = torch_utils.pool(h, first_trig_mask, type=self.opt['pooling'])
        first_argu1_emb = torch_utils.pool(h, first_argu1_mask, type=self.opt['pooling'])
        first_argu2_emb = torch_utils.pool(h, first_argu2_mask, type=self.opt['pooling'])

        first_argu1_type = self.ner_emb(first_argu1_type_id).squeeze(1)
        first_argu2_type = self.ner_emb(first_argu2_type_id).squeeze(1)

        first_argu1_emb = torch.cat((first_argu1_emb, first_argu1_type), 1)
        first_argu2_emb = torch.cat((first_argu2_emb, first_argu2_type), 1)

        ############################################################
        new_words, second_argu1_mask, second_argu2_mask = bind_inputs_second
        cls_embs2, rnn_output = self.unique_bilstm._get_lstm_features(new_words)

        second_argu1_emb = torch_utils.pool(rnn_output, second_argu1_mask, type=self.opt['pooling'])
        second_argu2_emb = torch_utils.pool(rnn_output, second_argu2_mask, type=self.opt['pooling'])

        argu1_emb = cls_embs + first_argu1_emb + second_argu1_emb
        argu2_emb = cls_embs + first_argu2_emb + second_argu2_emb

        argu1_emb = self.bind_layer_norm(argu1_emb)
        argu1_emb = self.dropout(argu1_emb)

        argu2_emb = self.bind_layer_norm(argu2_emb)
        argu2_emb = self.dropout(argu2_emb)

        argu_emb = argu1_emb + argu2_emb
        logits = self.bind_output(argu_emb)
        return logits, h_out

    '''
    特别注意两个编码器的subj_mask和obj_mask是不同的
    entity_pairs是元组
    '''
    def forward(self, words, char_ids, pos_ids, prot_ids, parse_inputs, first_entity_pairs, second_entity_pairs):
        cls_embs, h, pool_mask = self.encode_gcn(words, char_ids, pos_ids, prot_ids, parse_inputs)
        trig_type_id, first_trig_mask, argu_type_id, first_argu_mask = first_entity_pairs

        batch_size = first_trig_mask.size(0)
        h_embedding = h.repeat(batch_size, 1, 1)
        cls_embs = cls_embs.repeat(batch_size, 1)
        cls_embs = self.cls_layer(cls_embs)

        h_out = torch_utils.pool(h, pool_mask, type=self.opt['pooling'])
        first_subj_emb = torch_utils.pool(h_embedding, first_trig_mask, type=self.opt['pooling'])
        first_obj_emb = torch_utils.pool(h_embedding, first_argu_mask, type=self.opt['pooling'])
        ## 实体的类型很重要
        first_subj_type = self.ner_emb(trig_type_id).squeeze(1)
        first_obj_type = self.ner_emb(argu_type_id).squeeze(1)
        first_subj_emb = torch.cat((first_subj_emb, first_subj_type), 1)
        first_obj_emb = torch.cat((first_obj_emb, first_obj_type), 1)


        ## 第二个编码器
        new_words, second_trig_mask, second_argu_mask = second_entity_pairs
        second_cls_embs, second_rnn_output = self.unique_bilstm._get_lstm_features(new_words)

        second_subj_emb = torch_utils.pool(second_rnn_output, second_trig_mask, type=self.opt['pooling'])
        second_obj_emb = torch_utils.pool(second_rnn_output, second_argu_mask, type=self.opt['pooling'])

        subj_emb = cls_embs + first_subj_emb + second_subj_emb
        obj_emb = cls_embs + first_obj_emb + second_obj_emb

        subj_emb = self.layer_norm(subj_emb)
        subj_emb = self.dropout(subj_emb)

        obj_emb = self.layer_norm(obj_emb)
        obj_emb = self.dropout(obj_emb)

        part1 = torch.einsum('bi,rij,bj->br', [subj_emb, self.W1, obj_emb])
        res2 = torch.cat((subj_emb, obj_emb), 1)
        part2 = torch.mm(res2, self.W2)
        logits = part1 + part2 + self.b

        return logits, h_out

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, opt, bert_model, bert_tokenizer, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bilstm = GCN_BiLSTM(opt, bert_model, bert_tokenizer)
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim

        self.deprela_embs = nn.Embedding(len(constant.DEPREL_TO_ID), self.opt['dep_dim']) if opt['dep_dim'] > 0 else None

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # 计算关系的
        self.WR = nn.Linear(self.opt['dep_dim'], 20)
        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.opt['hidden_dim']*2 if layer == 0 else self.mem_dim+20
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.linear = nn.Linear(768, 400)

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    ## 用了 word 和 pos 特征
    def forward(self, adj, dep_adj, words, char_ids, pos_ids, prot_ids, parse_inputs, seq_lens):
        _, deprel, head = parse_inputs

        cls_embs, gcn_inputs = self.bilstm._get_lstm_features(words, char_ids, pos_ids, prot_ids)
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        # 在此处把依存边类型添加上
        dep_embs = self.deprela_embs(dep_adj)
        Wdep = self.WR(dep_embs)
        Wdep = torch.matmul(adj, Wdep)
        Wdep = Wdep.sum(2)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = torch.cat([AxW, Wdep], dim=2)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return cls_embs, gcn_inputs, mask
