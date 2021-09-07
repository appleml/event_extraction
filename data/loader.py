"""
Data loader for bioevent files.
"""
import random
import torch
import numpy as np
import data.gcn_input_bio as input_data_bio
import data.data_process as data_process
from utils import constant, common_utils

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, data_path, parser_path, batch_size, opt, vocab, char2id, evaluation=False):#vocab是实例
        self.batch_size = batch_size
        self.opt = opt
        self.char2id = char2id
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.RELA_TO_ID

        data = input_data_bio.read_data(data_path, parser_path)
        data = self.preprocess(data) #以文件为单位

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        #self.num_files = len(data)

        self.num_examples = len(data)
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), data_path))
    '''
    生成句子的训练语料
    不以文件为单位，因为是训练阶段，所以以句子为单位
    将trigger为空，protein为空的样例剔除, event为空的句子暂不剔除
    候选实体对中每对都与句子组成一个候选样例
    以句子为单位： filename, sentId, sent_features, entity_pair, gold_events
    sent_info = sentence(old_sent, new_sent, copy.deepcopy(prot_dict), char_seq, sgenia, sparse)    
    '''

    def preprocess(self, files):
        sent_data = list()
        for file in files:
            file_name = file.file_name
            #print(file_name, "^^^^^^^^^^^^^^^^^^^^^^^^^")
            for sent in file.sent_set:
                tokens = sent.new_sent
                #print(tokens)
                prev_sent = sent.prev_sentence
                next_sent = sent.next_sentence
                sent_char_feats = common_utils.get_char_ids(tokens, self.char2id)
                sent_genia = sent.genia_info
                genia_words = sent_genia.words
                genia_pos_ids = common_utils.map_to_ids(sent_genia.pos_set, constant.POS_TO_ID)
                genia_prot_ids = common_utils.map_to_ids(sent_genia.prot_set, constant.PROT_TO_ID)
                genia_cjprot_ids = common_utils.map_to_ids(sent_genia.cjprot_set, constant.PROT_TO_ID)
                genia_ctrig_ids = common_utils.map_to_ids(sent_genia.ctrig_set, constant.TRIGGER_TO_ID)
                genia_jprot_ids = common_utils.map_to_ids(sent_genia.jprot_set,
                                                          constant.JPROT_TO_ID)  # join protein的gold sequence
                sent_labels = common_utils.map_to_ids(sent_genia.trig_types, constant.TRIGGER_TO_ID)  # trigger的gold sequence
                genia_info = (genia_words, genia_pos_ids, genia_prot_ids, genia_cjprot_ids, genia_ctrig_ids, sent_genia.prot_set,
                              sent_genia.jprot_set, genia_jprot_ids, sent_genia.trig_types, sent_labels)

                # parse information
                sent_parse = sent.parse_info
                parse_pos = common_utils.map_to_ids(sent_parse.pos_set, constant.POS_TO_ID)
                deprel = common_utils.map_to_ids(sent_parse.deptype_set, constant.DEPREL_TO_ID)  # 依存类型
                head = [int(x) for x in sent_parse.head_set]  # 头词
                assert any([x == 0 for x in head])  # list中要有一个0,因为需要有一个词指向ROOT
                parse_info = (parse_pos, deprel, head)

                gold_events = list(sent.event_dict.values())
                # 如果句子中没有trig_dict和prot_dict就过滤掉
                if len(sent.trig_dict) != 0 and len(sent.prot_dict) != 0:
                    entity_pairs, bind_argus, protmod_argus, regu_argus = data_process.gen_sent_pairs(self.opt, tokens, prev_sent, next_sent, sent.trig_dict, sent.prot_dict, sent.event_dict)
                    sent_data += [(file_name, tokens, sent_char_feats, prev_sent, next_sent, sent.prot_dict, genia_info, parse_info, entity_pairs, bind_argus, protmod_argus, regu_argus, gold_events)]
            # if len(sent_data) > 40:
            #   break
        return sent_data

    # 语料中标注的事件集合
    def gold_events(self): #在train.py中用到了
        """ Return gold labels as a list. """
        return self.gold_events

    # 识别的实体对集合
    def gold_seqs(self):
        return self.gold_labels

    def __len__(self):
        return len(self.data)

    # 迭代的方式一次输出一个batch
    # file_name, tokens, sent_char_feats, prev_sent, next_sent, sent.prot_dict, genia_info, parse_info, entity_pairs, bind_argus, protmod_argus, regu_argus, gold_events
    # pair_labels, entity_exams, bind_inputs, bind_labels, bind_exams, gold_events)
    # entity_pair: (event_idx_set, trig, trig_type_idx, trig_posit, prot, argu_type_idx, argu_posit, rela_type)
    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        # batch中个数是20个example, 一个元组就是一个example
        batch = self.data[key]
        batch_size = len(batch)

        batch = list(zip(*batch))  # 元组中有10个元素，比如说第一组元素，就是20个样例的tokenId集合，第二个元素是posId集合，是十个特征
        assert len(batch) == 13

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[1]]
        batch, orig_idx = common_utils.sort_all(batch, lens)

        file_name = batch[0]
        # convert to tensors
        words = batch[1]
        # genia information
        char_feats = batch[2][0]
        char_ids = torch.LongTensor(char_feats)

        prev_sent = batch[3]
        next_sent = batch[4]

        prot_dict = batch[5][0]

        genia_features = list(zip(*batch[6]))
        genia_words, genia_posIds, genia_protIds, genia_cjprot_ids, genia_ctrig_ids, prot_labels, jprot_labels, genia_jprotIds, trig_labels, trig_labelIds = genia_features
        pos_ids = common_utils.get_long_tensor(genia_posIds, batch_size)
        prot_ids = common_utils.get_long_tensor(genia_protIds, batch_size)
        cjprot_ids = common_utils.get_long_tensor(genia_cjprot_ids, batch_size)
        ctrig_ids = common_utils.get_long_tensor(genia_ctrig_ids, batch_size)
        genia_info = (genia_words, pos_ids, prot_ids, cjprot_ids, ctrig_ids, list(prot_labels)[0], list(jprot_labels)[0],
        genia_jprotIds, list(trig_labels)[0], trig_labelIds)

        # parse information
        parse_features = list(zip(*batch[7]))
        parse_pos, deprel, head = parse_features
        parse_posIds = common_utils.get_long_tensor(parse_pos, batch_size)
        deprel = common_utils.get_long_tensor(deprel, batch_size)
        head = common_utils.get_long_tensor(head, batch_size)
        parse_info = (parse_posIds, deprel, head)
        # masks = torch.eq(words, 0)  # torch.uint8

        entity_pairs = batch[8][0]
        bind_argus = batch[9][0]
        protmod_argus = batch[10][0]
        regu_argus = batch[11][0]
        gold_events = batch[-1][0]  ## 加[0]的原始是目前batch=1, 不过此处用法需谨慎
        return file_name, words, char_ids, prev_sent, next_sent, prot_dict, genia_info, parse_info, entity_pairs, bind_argus, protmod_argus, regu_argus, gold_events

    # 在python中实现了__iter__方法的对象是可迭代的
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x for x in tokens]

def process_pair(batch_entitys_pair):
    couple_info = []
    for entitys_couple in batch_entitys_pair: #假设 batch=k, 即有k条语料
        for couple in entitys_couple:  # (trig, trig_type_idx, trig_posit, prot, argu_type_idx, argu_posit, rela_id)
            trig = couple[0]
            argu = couple[3]

            trig_position = torch.LongTensor(couple[2])
            argu_position = torch.LongTensor(couple[5])

            rela_id = couple[-1] #这儿不将其转换成Tensor, 后面还需要再处理
            couple_info.append((trig, trig_position, argu, argu_position, rela_id))
    return couple_info
