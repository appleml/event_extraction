"""
Utility functions for torch.
"""

import torch
from torch import nn, optim
from torch.optim import Optimizer
from transformers import AdamW
import utils.constant as constant

### class
class MyAdagrad(Optimizer):
    """My modification of the Adagrad optimizer that allows to specify an initial
    accumulater value. This mimics the behavior of the default Adagrad implementation 
    in Tensorflow. The default PyTorch Adagrad uses 0 for initial acculmulator value.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        init_accu_value (float, optional): initial accumulater value.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value, \
                weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) *\
                        init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss

### torch specific functions
def get_optimizer(name, bert_param, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD([{'params':bert_param, 'lr':1.0e-5}, {'params':parameters}], lr=lr)
            #parameters, lr=lr, weight_decay=l2)
    elif name in ['adagrad', 'myadagrad']:
        # use my own adagrad to allow for init accumulator value
        return MyAdagrad([{'params':bert_param, 'lr':0.00001},
                         {'params':parameters}], lr=lr, init_accu_value=0.1, weight_decay=l2)
    elif name == 'adam':
        return AdamW([{'params':bert_param, 'lr':0.00001},
                      {'params':parameters}], lr=lr, weight_decay=l2)
    elif name == 'adamax':
        return torch.optim.Adamax([{'params':bert_param, 'lr':0.00001},
                                  {'params':parameters}], lr=lr, weight_decay=l2) # use default lr
    elif name == 'adadelta':
        return torch.optim.Adadelta([{'params':bert_param, 'lr':0.00001},
                                    {'params':parameters}], lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat

def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var

def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

### model IO
def save(model, optimizer, opt, filename):
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': opt
    }
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")

def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump['model'])
    if optimizer is not None:
        optimizer.load_state_dict(dump['optimizer'])
    opt = dump['config']
    return model, optimizer, opt

def load_config(filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']

## 对word_seq进行word_piece
'''
sent_lens: 是List, 其中的元素就是句长, 里面个数是batch个
'''
def get_avgword_embs(bert_model, bert_tokenizer, sent_set, sent_lens):
    sent_results = []
    maxlen = max(sent_lens)

    for sent, sent_len in zip(sent_set, sent_lens):
        sent = sent.split()
        assert len(sent) == sent_len
        tokens = []
        token_mask = []
        # tokens.append('<s>') # Roberta
        tokens.append('[CLS]')  # other

        slices = []
        idx = 0
        for word in sent:  # 对于句子中所以的词
            token_set = bert_tokenizer.tokenize(word)
            tokens.extend(token_set)
            if len(token_set) == 1:
                slices.append([idx])
            else:
                slices.append([idx, idx + len(token_set) - 1])
                # slices.append([idx, idx])
            idx += len(token_set)
        if len(sent) < maxlen:
            tokens += ['[PAD]'] * (maxlen - len(sent))
            count = []
            for idd in range(idx, idx + (maxlen - len(sent))):
                mask = [1] * sent_len
                token_mask.append(mask)
                count.append('a')
            assert len(['[PAD]'] * (maxlen - len(sent))) == len(count)

        # tokens.append('</s>') #Roberta
        tokens.append('[SEP]')
        num_subwords = len(tokens) - 2  # 减去[CLS]和[SEP]

        input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids]).cuda()

        word_embs_set = []
        last_hidden_states = bert_model(input_ids)[0]  # Models outputs are now tuples
        last_hidden_states = last_hidden_states[:, 1:-1, :]
        cls_embs = last_hidden_states[:, 0, :]

        for indix in slices:
            mask = [1] * num_subwords
            mask[indix[0]:indix[-1] + 1] = [0] * (indix[-1] - indix[0] + 1)
            mask = torch.ByteTensor(mask).view(1, -1, 1).bool().cuda()
            word_embs = last_hidden_states.masked_fill(mask, 0)
            word_embedding = word_embs.sum(1) / (mask.size(1) - mask.float().sum(1))
            word_embs_set.append(word_embedding)

        sent_embs = torch.cat(word_embs_set, 0).unsqueeze(0)
        sent_results.append(sent_embs)

    result = torch.cat(sent_results, 0).cuda()
    return cls_embs, result


# 1, 取分词后的第一个词或者分词后所有词的和(9月2号, 词向量用average)
# 2, sent_set是batch中句子的集合，sent_lens是所以句子的集合
def get_word_embedding(bert_model, bert_tokenizer, sent_set, sent_lens):
    sent_results = []
    maxlen = max(sent_lens)
    for sent in sent_set:
        sent = sent.split()
        tokens = []
        #tokens.append('<s>') # Roberta
        tokens.append('[CLS]')
        slices = []
        idx = 0
        for word in sent: #对于句子中所有的词
            token_set = bert_tokenizer.tokenize(word)
            tokens.extend(token_set)
            if len(token_set) == 1:
                slices.append([idx])
            else:
                slices.append([idx, idx + len(token_set) - 1])
                #slices.append([idx, idx])
            idx += len(token_set)
        if len(sent) < maxlen:
            tokens += ['[PAD]']*(maxlen-len(sent))
            count = []
            for idd in range(idx, idx +(maxlen-len(sent))):
                slices.append(idd)
                count.append('a')
            assert len(['[PAD]']*(maxlen-len(sent))) == len(count)

        #tokens.append('</s>') #Roberta
        tokens.append('[SEP]')
        input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids]).cuda()
        word_embs_set = []
        with torch.no_grad():
            last_hidden_states = bert_model(input_ids)[0]  # Models outputs are now tuples
            cls_embedding = last_hidden_states[:, 0, :]
            last_hidden_states = last_hidden_states[:, 1:-1, :] #batch, seqence_length, word_dimension
            for indic in slices:
                indices = torch.tensor(indic).cuda()
                tokens_emb = torch.index_select(last_hidden_states, 1, indices)
                word_emb = torch.sum(tokens_emb, dim=1)
                # word_emb = torch.unsqueeze(word_emb, dim=0)
                word_embs_set.append(word_emb)
        sent_embs = torch.cat(word_embs_set, 0).unsqueeze(0)
        sent_results.append(sent_embs)

    result = torch.cat(sent_results, 0).cuda()
    return result, cls_embedding

## 识别trigger的时候用的一些函数
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

###########################################################3
from torch.autograd import Variable
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
## 程序中默认用的 max
def pool(h, mask, type='avg'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)
