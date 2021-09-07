"""
Data loader for bioevent files.
"""
import data.gcn_input_bio as input_data_bio
from utils import constant, common_utils
import torch

class TESTDataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, data_path, parse_path, batch_size, opt, vocab, char2id, evaluation=False):#vocab是实例
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.char2id = char2id
        self.eval = evaluation

        test_data = input_data_bio.read_test_data(data_path, parse_path)
        test_data = self.preprocess(test_data) #以文件为单位
        self.num_examples = len(test_data)

        # chunk into batches
        test_data = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
        self.test_data = test_data
        print("{} batches created for {}".format(len(test_data), data_path))

    '''
    entity_pair = (event_idx_set, trig, trig_type, trig_posit, prot, argu_type, argu_posit, rela_id)     
    (old_sent, new_sent, copy.deepcopy(prot_dict), sentence_genia, sentence_parse)
    '''
    def preprocess(self, files):
        sent_data = list()
        for file in files:
            file_name = file.file_name
            for sent in file.sent_set:
                old_sent = sent.old_sent
                tokens = sent.new_sent
                prev_sent = sent.prev_sentence
                next_sent = sent.next_sentence
                sent_char_feats = common_utils.get_char_ids(tokens, self.char2id)
                ## genia information
                sent_genia = sent.genia_info
                genia_words = sent_genia.words
                genia_pos = common_utils.map_to_ids(sent_genia.pos_set, constant.POS_TO_ID)
                genia_prot = common_utils.map_to_ids(sent_genia.prot_set, constant.PROT_TO_ID)
                genia_info = (genia_words, genia_pos, genia_prot, sent_genia.prot_set)

                # parse information
                sent_parse = sent.parse_info
                parse_pos = common_utils.map_to_ids(sent_parse.pos_set, constant.POS_TO_ID)
                deprel = common_utils.map_to_ids(sent_parse.deptype_set, constant.DEPREL_TO_ID)  # 依存类型
                head = [int(x) for x in sent_parse.head_set]  # 头词
                assert any([x == 0 for x in head])  # list中要有一个0,因为需要有一个词指向ROOT
                parse_info = (parse_pos, deprel, head)
                sent_data += [(file_name, old_sent, tokens, sent_char_feats, prev_sent, next_sent, sent.prot_dict, genia_info, parse_info)]

        return sent_data

    def __len__(self):
        return len(self.test_data)

    #迭代的方式一次输出一个batch
    #(file_name, old_sent, tokens, sent_char_feats, prev_sent, next_sent, sent.prot_dict, genia_info, parse_info)
    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.test_data):
            raise IndexError
        # batch中个数是20个example, 一个元组就是一个example
        batch = self.test_data[key]
        batch_size = len(batch)

        batch = list(zip(*batch))  # 元组中有10个元素，比如说第一组元素，就是20个样例的tokenId集合，第二个元素是posId集合，是十个特征
        assert len(batch) == 9

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[1]]
        batch, orig_idx = common_utils.sort_all(batch, lens)

        file_name = batch[0]
        old_sent = batch[1]
        words = batch[2]
        char_ids = torch.LongTensor(batch[3][0])
        prev_sent = batch[4]
        next_sent = batch[5]

        # genia information
        prot_dict = batch[6][0]
        genia_features = list(zip(*batch[7]))
        genia_words, genia_posIds, genia_protIds, genia_prot_labels = genia_features
        pos_ids = common_utils.get_long_tensor(genia_posIds, batch_size)
        prot_ids = common_utils.get_long_tensor(genia_protIds, batch_size)
        genia_info = (genia_words, pos_ids, prot_ids, genia_prot_labels)

        # parse information
        parse_features = list(zip(*batch[8]))
        parse_pos, deprel, head = parse_features
        parse_posIds = common_utils.get_long_tensor(parse_pos, batch_size)
        deprel = common_utils.get_long_tensor(deprel, batch_size)
        head = common_utils.get_long_tensor(head, batch_size)
        parse_info = (parse_posIds, deprel, head)

        return file_name, old_sent, words, char_ids, prev_sent, next_sent, prot_dict, genia_info, parse_info

    # 在python中实现了__iter__方法的对象是可迭代的
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)