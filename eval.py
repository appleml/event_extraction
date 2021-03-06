"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.test_loader import TESTDataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, constant, helper
from utils.vocab import Vocab
from data.write_a2 import write_event
from collections import Counter
import data.data_process as data_process

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='saved_models/00', help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/2013_data')
parser.add_argument('--parse_dir', type=str, default='dataset/2013_genia_parse')
parser.add_argument('--output_dir', type=str, default='dataset/output_a2')
parser.add_argument('--mode', type=str, default='test', help='experiment on development data or test data')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)

constant.TRIGGER_TO_ID = data_process.gen_trigidx()
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = args.data_dir + '/2013_test'
parse_file = args.parse_dir + '/2013_test'
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))

char_file = opt['vocab_dir'] + '/char.pkl'
char2id = vocab.load_charinfo(char_file)
constant.CHAR_TO_ID = char2id
batch = TESTDataLoader(data_file, parse_file, opt['batch_size'], opt, vocab, constant.CHAR_TO_ID, evaluation=True)

helper.print_config(opt)
rela2id = constant.RELA_TO_ID
id2rela = dict([(v,k) for k,v in rela2id.items()])

## ??????????????????, ????????????
file_trigs = dict()
file_events = dict()
##??????????????????????????????trigger ??? event ???????????????
prev_file_name = ''
prev_trig_idx = 1
prev_event_idx = 1

## ????????????trigger?????????????????????, ??????????????????????????????????????????
prev_context = ""
# pred_event_number?????????????????????,??????????????????????????????????????????, binding??????, regu???????????????, ???????????????????????????
pred_event_number = Counter()
## ?????????????????????????????????????????????join protein ??? trigger
prev_jprot_set = list()
prev_ctrig_dict = dict()

batch_iter = tqdm(batch)
for i, batch in enumerate(batch_iter):
    file_name, last_trig_idx, last_event_idx, last_context, preds_trigs_dict, pred_events = trainer.test(batch,
                                                                                                         prev_jprot_set,
                                                                                                         prev_ctrig_dict,
                                                                                                         prev_file_name,
                                                                                                         prev_trig_idx,
                                                                                                         prev_event_idx,
                                                                                                         prev_context,
                                                                                                         pred_event_number)

    if prev_file_name != file_name:
        prev_jprot_set.clear()
        prev_ctrig_dict.clear()

    prev_file_name = file_name
    prev_trig_idx = last_trig_idx
    prev_event_idx = last_event_idx
    prev_context = last_context

    ## ?????? trigger
    if file_name not in file_trigs.keys():
        file_trigs[file_name] = list(preds_trigs_dict.values())
    else:
        file_trigs[file_name].extend(preds_trigs_dict.values())

    ## ?????? event
    if file_name not in file_events.keys():
        file_events[file_name] = pred_events
    else:
        file_events[file_name].extend(pred_events)


print("?????????????????????:")
print(pred_event_number)
print(sum(pred_event_number.values()))
## ???event_dict??????, ???????????????develment, ??????test??????gold trigger
print("Writing predicted events .....................")
write_event(args.output_dir, file_trigs, file_events)
## ???????????????????????????????????????:
print("??????????????????????????????:")
#outfile_eventnum(args.output_dir)
print("ended ...................")
print("Evaluation ended.")