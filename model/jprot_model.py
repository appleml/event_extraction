import torch
import torch.autograd as autograd
import torch.nn as nn
import utils.torch_utils as utils
from model.jprotein_bilstm import JProtein_BiLSTM

class PBiLSTM_CRF(nn.Module):
    def __init__(self, opt, bert_model, bert_tokenizer, jprot2id):
        super(PBiLSTM_CRF, self).__init__()
        self.opt = opt
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bilstm = JProtein_BiLSTM(self.opt, self.bert_model, self.bert_tokenizer)
        self.jprot2id = jprot2id
        self.START_TAG = "START"
        self.STOP_TAG = "STOP"
        self.label_size = len(jprot2id)

        self.in_drop = nn.Dropout(self.opt['input_dropout'])
        self.hidden2tag = nn.Linear(2*self.opt['rnn_hidden'], len(jprot2id))
        self.transformer2tag = nn.Linear(832, len(jprot2id))

        self.transitions = nn.Parameter(torch.randn(len(jprot2id), len(jprot2id)))
        self.transitions.data[jprot2id[self.START_TAG], :] = -10000
        self.transitions.data[:, jprot2id[self.STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        init_alphas = torch.Tensor(1, self.label_size).fill_(-10000.)
        init_alphas[0][self.jprot2id[self.START_TAG]] = 0.
        forward_var = init_alphas.cuda()
        feats = feats.squeeze(0)
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.label_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.label_size)  # 当前词feat被分配为next_tag的分数
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score # forward_var已解析出的序列的得分， trans_score是12个label转移到当前next_tag的得分
                alphas_t.append(utils.log_sum_exp(next_tag_var).unsqueeze(0))

            forward_var = torch.cat(alphas_t, 0)
        terminal_var = forward_var + self.transitions[self.jprot2id[self.STOP_TAG]]
        alpha = utils.log_sum_exp(terminal_var.view(1, -1))

        return alpha

    def _score_sentence(self, feats, tags):
        tags = tags[0]
        score = autograd.Variable(torch.Tensor([0])).cuda()
        tags = torch.LongTensor([self.jprot2id[self.START_TAG]]+tags)
        feats = feats.squeeze(0)
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]] #self.transitions[tags[i + 1], tags[i]] 标签tag[i]转到tag[i+1]的概率

        score = score + self.transitions[self.jprot2id[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.Tensor(1, self.label_size).fill_(-10000.)
        init_vvars[0][self.jprot2id[self.START_TAG]] = 0
        forward_var = init_vvars.cuda()
        feats = feats.squeeze(0)
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.label_size):   # 到达该标签最好的序列，得分是多少
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = utils.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].unsqueeze(0))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.jprot2id[self.STOP_TAG]]
        best_tag_id = utils.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.jprot2id[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, words, char_ids, pos_ids, prot_ids, cjprot_ids, pfea_ids):
        feats = self.bilstm._get_lstm_features(words, char_ids, pos_ids, prot_ids, cjprot_ids)
        feats = self.hidden2tag(feats)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, pfea_ids) # 句子标注序列的打分
        return forward_score - gold_score
        # --- transformer features---------------
        # feats = self.bilstm._get_transformer_features(words, char_ids, pos_ids, prot_ids, cjprot_ids)
        # feats = self.transformer2tag(feats)
        # forward_score = self._forward_alg(feats)
        # gold_score = self._score_sentence(feats, pfea_ids)  # 句子标注序列的打分
        # return forward_score - gold_score

    def forward(self, words, char_ids, pos_ids, prot_ids, cjprot_ids):
        lstm_feats = self.bilstm._get_lstm_features(words, char_ids, pos_ids, prot_ids, cjprot_ids)
        lstm_feats = self.hidden2tag(lstm_feats)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
        # --- transformer features---------------
        # lstm_feats = self.bilstm._get_transformer_features(words, char_ids, pos_ids, prot_ids, cjprot_ids)
        # lstm_feats = self.transformer2tag(lstm_feats)
        # score, tag_seq = self._viterbi_decode(lstm_feats)
        # return score, tag_seq