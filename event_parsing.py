"""
Train a model on TACRED.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import constant, helper, common_utils
from utils.vocab import Vocab
from metric.rela_metrics import rela_score
from metric.event_metrics import score_event
from metric.bind_metirc import bind_score
import metric.trig_bio_metrics as tmetrics
import metric.jprot_metrics as jpmetrics
import data.data_process as data_process

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/2013_data') #2011_data_one
parser.add_argument('--parse_dir', type=str, default='dataset/2013_genia_parse') #2011_genia_parse_one_split_bio
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--char_dim', type=int, default=30, help='char embedding dimension')
parser.add_argument('--conv_filter_size', type=int, default=2, help='char feature of word')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--prot_dim', type=int, default=2, help='protein embedding dimension')
parser.add_argument('--jprot_dim', type=int, default=2, help='join protein embedding dimension')
parser.add_argument('--cjprot_dim', type=int, default=2, help='context join protein type embedding dimension')
parser.add_argument('--ctrig_dim', type=int, default=8, help='context trigger type embedding dimension')
parser.add_argument('--dep_dim', type=int, default=30, help='dependency relation type dimension')
parser.add_argument('--ner_dim', type=int, default=10, help='NER embedding dimension.')
parser.add_argument('--rnn_input_dim', type=int, default=600, help='linear exchange')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
#parser.set_defaults(lower=False)

parser.add_argument('--context_window', type=int, default=100, help='context word, 增加句子的前后文信息')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='avg', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=1.0e-4, help='Applies to sgd and adagrad.')
parser.add_argument('--jprot_lr', type=float, default=1.0e-4, help='Applies to sgd and adagrad.')
parser.add_argument('--trig_lr', type=float, default=1.0e-4, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')#思考,四个模块是否不一致
parser.add_argument('--num_epoch', type=int, default=80, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()

# make opt
opt = vars(args)
rela2id = constant.RELA_TO_ID
opt['num_class'] = len(rela2id)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

char_file = opt['vocab_dir'] + '/char.pkl'
char2id = vocab.load_charinfo(char_file)
constant.CHAR_TO_ID = char2id
constant.TRIGGER_TO_ID = data_process.gen_trigidx()

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/2013_train', opt['parse_dir'] + '/2013_train', opt['batch_size'], opt, vocab, char2id, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/2013_devel', opt['parse_dir'] + '/2013_devel', opt['batch_size'], opt, vocab, char2id, evaluation=True)

opt['t_total'] = len(train_batch)*opt['num_epoch']

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

trainer = GCNTrainer(opt)

id2jprot = dict([(v, k) for k,v in constant.JPROT_TO_ID.items()])
id2rela = dict([(v,k) for k,v in rela2id.items()])

current_jplr = opt['jprot_lr']
current_tlr = opt['trig_lr']
current_lr = opt['lr']  # lr of relation

dev_jprot_score_history = []
dev_trig_score_history = []
dev_rela_score_history = []
dev_event_score_history = [] # event

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}),jp_loss = {:.6f}, trig_loss = {:.6f}, rela_loss = {:.6f}, bind_loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

# start training
jprot_f1_set = []
trig_f1_set = []
rela_f1_set = []
bind_f1_set = []
event_f1_set = []
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0  # record relation loss, 暂时不考虑其他loss
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        jp_loss, trig_loss, rela_loss, bind_loss = trainer.update(batch)
        train_loss += rela_loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch, opt['num_epoch'], jp_loss, trig_loss, rela_loss, bind_loss, duration, current_lr))

    print("Evaluating on dev set...")
    pred_jprot_seq = list()
    gold_jprot_seq = list()
    pred_trig_seq = list()
    gold_trig_seq = list()
    pred_gold_relas = list()
    pred_gold_binds = list()
    pred_gold_events = list() ## 用于评估预测的事件

    dev_jprot_loss = 0
    dev_trig_loss = 0
    dev_rela_loss = 0
    dev_bind_loss = 0
    dev_loss = 0 #这个一会删除

    prev_filename = ""
    prev_cjprot_set = []
    prev_ctrig_dict = dict()
    for i, batch in enumerate(dev_batch):
        file_name, jprot_loss, trig_loss, jprot_results, trig_results, rela_results, bind_results, event_results = trainer.predict(epoch, batch, prev_filename, prev_cjprot_set, prev_ctrig_dict)
        if file_name != prev_filename:
            prev_filename = file_name
            prev_cjprot_set.clear()
            prev_ctrig_dict.clear()

        #dev_loss += rela_loss
        dev_jprot_loss += jprot_loss
        dev_trig_loss += trig_loss
        #dev_loss += trig_loss
        pred_jprot, gold_jprot = jprot_results
        pred_jprot_seq.append(pred_jprot)
        gold_jprot_seq.append(gold_jprot)
        pred_trigs, gold_trigs = trig_results
        pred_trig_seq.append(pred_trigs)
        gold_trig_seq.append(gold_trigs)

        if rela_results != None:
            pred_gold_relas.append(rela_results)
        if bind_results != None:
            pred_gold_binds.append(bind_results)
        if event_results != None:
            pred_events, gold_events = event_results
            # 对gold_events进行处理
            sent_gold_events = helper.process_gold_events(file_name, gold_events) # 9月5号, 需要修改
            ## 将预测的事件与gold_events进行保存，最后一并评估
            pred_gold_events.append((pred_events, sent_gold_events))

    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size'] ## 此处用的trigger的loss

    print('############-----参与事件的protein的预测情况----#########################')
    dev_jp_p, dev_jp_r, dev_jp_f1 = jpmetrics.eval_jprot(pred_jprot_seq, gold_jprot_seq)
    jprot_f1_set.append(dev_jp_f1)
    dev_jpscore = dev_jp_f1

    print('############----trigger的预测情况----##################################')
    dev_trig_p, dev_trig_r, dev_trig_f1 = tmetrics.eval_trig(pred_trig_seq, gold_trig_seq)
    trig_f1_set.append(dev_trig_f1)
    dev_tscore = dev_trig_f1

    print('#############----trigger 与 argument间----关系判断 ######################')
    dev_rela_p, dev_rela_r, dev_rela_f1 = rela_score(pred_gold_relas)
    rela_f1_set.append(dev_rela_f1)

    print('#############----bind论元情况评估----####################################')
    bind_p, bind_r, bind_f1 = bind_score(pred_gold_binds)
    bind_f1_set.append(bind_f1)

    print("#############----组合事件情况评估----#####################################")
    event_p, event_r, event_f1 = score_event(pred_gold_events)
    event_f1_set.append(event_f1)

    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_rela_f1 = {:.4f}".format(epoch, train_loss, dev_loss, event_f1))
    dev_score = event_f1
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_event_score_history)))
    print(max(event_f1_set), '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    if epoch == 1 or dev_score > max(dev_event_score_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
                                                                .format(epoch, event_p*100, event_r*100, dev_score*100))
        print("最好的预测结果是以及它的epoch:", event_f1, '----',epoch)
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    #
    # lr schedule
    if len(dev_event_score_history) > opt['decay_epoch'] and dev_score <= dev_event_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_event_score_history += [dev_score]
    print("")

    torch.cuda.empty_cache()

print("Training ended with {} epochs.".format(epoch))
