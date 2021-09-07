"""
A trainer class.
"""
import torch
import torch.nn as nn
import numpy as np

from model.trig_model import TBiLSTM_CRF
from model.jprot_model import PBiLSTM_CRF

from model.gcn import GCNClassifier
from utils import constant, torch_utils, common_utils
from data_postprocess.event_generation import gen_event, gen_test_event
from data.bind_example import gen_bind_example
from data.regu_example import gen_regu_example
import data.data_process as data_process
from transformers import BertTokenizer, BertModel

from transformers import get_linear_schedule_with_warmup

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.bert_model.load_state_dict(checkpoint['bert_model'])

        self.jprot_model.load_state_dict(checkpoint['jprot_model'])
        self.jprot_optimizer.load_state_dict(checkpoint['jprot_optimizer'])

        self.trig_model.load_state_dict(checkpoint['trig_model'])
        self.trig_optimizer.load_state_dict(checkpoint['trig_optimizer'])

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['model_optimizer'])

        self.opt = checkpoint['config']
        start_epoch = checkpoint['epoch']

        return start_epoch

    def save(self, filename, epoch):
        params = {
            'epoch': epoch,
            'bert_model': self.bert_model.state_dict(),

            'jprot_model': self.jprot_model.state_dict(),
            'jprot_optimizer': self.jprot_optimizer.state_dict(),

            'trig_model': self.trig_model.state_dict(),
            'trig_optimizer': self.trig_optimizer.state_dict(),

            'model': self.model.state_dict(),
            'model_optimizer': self.optimizer.state_dict(),

            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix

        # model
        self.bert_tokenizer = BertTokenizer.from_pretrained('dataset/scibert_scivocab_cased/')
        self.bert_model = BertModel.from_pretrained('dataset/scibert_scivocab_cased/')

        ADDITIONAL_SPECIAL_TOKENS = common_utils.gen_tags()
        self.bert_tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        self.bert_param_ids = list(map(id, self.bert_model.parameters()))

        # --- joint protein model ---
        self.jprot2id = constant.JPROT_TO_ID
        self.id2jprot = dict([(v, k) for k, v in constant.JPROT_TO_ID.items()])
        self.jprot_model = PBiLSTM_CRF(opt, self.bert_model, self.bert_tokenizer, self.jprot2id)
        self.jprot_base_params = filter(lambda p: p.requires_grad and id(p) not in self.bert_param_ids, self.jprot_model.parameters())
        self.jprot_optimizer = torch_utils.get_optimizer(opt['optim'], self.jprot_model.bert_model.parameters(), self.jprot_base_params, self.opt['jprot_lr'])
        self.jprot_scheduler = get_linear_schedule_with_warmup(self.jprot_optimizer, num_warmup_steps=0, num_training_steps=opt['t_total'])
        if opt['cuda']:
            self.jprot_model.cuda()

        #--- trigger model ---
        self.trig2id = constant.TRIGGER_TO_ID
        self.id2trig = dict([(v, k) for k, v in constant.TRIGGER_TO_ID.items()])
        self.trig_model = TBiLSTM_CRF(opt, self.bert_model, self.bert_tokenizer, self.trig2id)
        self.trig_base_params = filter(lambda p: p.requires_grad and id(p) not in self.bert_param_ids, self.trig_model.parameters())
        self.trig_optimizer = torch_utils.get_optimizer(opt['optim'], self.trig_model.bert_model.parameters(), self.trig_base_params, self.opt['trig_lr'])
        self.trig_scheduler = get_linear_schedule_with_warmup(self.trig_optimizer, num_warmup_steps=0, num_training_steps=self.opt['t_total'])
        if opt['cuda']:
            self.trig_model.cuda()

        #--- relation extraction ---
        self.rela2id = constant.RELA_TO_ID
        self.id2rela = dict([(v, k) for k, v in self.rela2id.items()])
        self.model = GCNClassifier(opt, self.bert_model, self.bert_tokenizer, emb_matrix=emb_matrix)
        self.base_params = filter(lambda p: p.requires_grad and id(p) not in self.bert_param_ids, self.model.parameters())
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.bert_model.parameters(), self.base_params, self.opt['lr'])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=opt['t_total'])
        self.criterion = nn.CrossEntropyLoss()
        self.bind_loss = torch.nn.BCEWithLogitsLoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
            self.bind_loss.cuda()

    '''
    batch = file_name, words, genia_info, parse_info, protrig, entity_pair, gold_events
    genia_info = genia_words, pos_ids, prot_ids, jprot_labels, genia_jprotIds, trig_labels, trig_labelIds
    parse_info = parse_posIds, deprel, head
    trigargu_pair = (event_idx_set, trig, trig_position, argu, argu_position, tp_rela_id)
    '''
    def update(self, batch):
        file_name, words, char_ids, prev_sent, next_sent, prot_dict, genia_input, parse_input, entity_pairs, bind_argus, pmod_argus, regu_argus, gold_events = common_utils.unpack_batch(batch, self.opt['cuda'])
        words, pos_ids, prot_ids, cjprot_ids, ctrig_ids, prot_labels, jprot_labels, jprot_ids, trig_labels, tlabel_ids = genia_input
        # print(file_name[0])
        # print(words[0])
        # print("######################################")

        # step forward
        self.jprot_model.train()
        self.trig_model.train()
        self.jprot_optimizer.zero_grad()
        self.trig_optimizer.zero_grad()

        # --- join protein的识别情况 ---
        jprot_neg_log_likelihood = self.jprot_model.neg_log_likelihood(words, char_ids, pos_ids, prot_ids, cjprot_ids, jprot_ids)
        jprot_neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm_(self.jprot_model.parameters(), self.opt['max_grad_norm'])
        self.jprot_optimizer.step()
        self.jprot_scheduler.step()

        # --- trigger的识别情况 ---
        jprot_ids = common_utils.get_long_tensor(jprot_ids, 1)
        if self.opt['cuda']:
            jprot_ids = jprot_ids.cuda()
        trig_neg_log_likelihood = self.trig_model.neg_log_likelihood(words, char_ids, pos_ids, prot_ids, jprot_ids, ctrig_ids, tlabel_ids)
        trig_neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm_(self.trig_model.parameters(), self.opt['max_grad_norm'])
        self.trig_optimizer.step()
        self.trig_scheduler.step()

        # --- trigger与argument间关系的判断 ---
        self.model.train()
        self.optimizer.zero_grad()
        loss_val = 0
        first_pair_input_set, second_pair_input_set, rela_ids_set = common_utils.process_batch_pair(entity_pairs, self.opt['cuda'], "train")
        for first_pair_input, second_pair_input, rela_ids in zip(first_pair_input_set, second_pair_input_set, rela_ids_set):
            logits_res, pooling_output = self.model(words, char_ids, pos_ids, prot_ids, parse_input, first_pair_input, second_pair_input)
            loss = self.criterion(logits_res, rela_ids)
            #loss = self.focal_loss(logits_res, rela_ids)
            #l2 decay on all conv layers
            if self.opt.get('conv_l2', 0) > 0:
                loss += self.model.conv_l2() * self.opt['conv_l2']
            # l2 penalty on output representations
            if self.opt.get('pooling_l2', 0) > 0:
                loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
            loss_val = loss.item()
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
            self.optimizer.step()
            self.scheduler.step()

        # --- binding argument 是否在同一个事件中 ---
        bind_val = 0
        if len(bind_argus[0]) > 0:
            b_inputs_first_set, b_inputs_second_set, b_labels_set = common_utils.process_batch_br2(bind_argus, self.opt['cuda'], "train")
            for b_inputs_first, b_inputs_second, b_labels in zip( b_inputs_first_set, b_inputs_second_set, b_labels_set):
                bind_logits, pooling_output = self.model.bind_argu(words, char_ids, pos_ids, prot_ids, parse_input, b_inputs_first, b_inputs_second)
                bind_loss = self.bind_loss(bind_logits.squeeze(1), b_labels.float().cuda())
                if self.opt.get('pooling_l2', 0) > 0:
                    bind_loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
                bind_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                bind_val = bind_loss.item()

        pmode_val = 0
        if len(pmod_argus[0]) > 0:
            pmod_inputs_first_set, pmod_inputs_second_set, pm_labels_set = common_utils.process_batch_br2(pmod_argus, self.opt['cuda'], "train")
            for pmod_inputs_first, pmod_inputs_second, pm_labels in zip(pmod_inputs_first_set, pmod_inputs_second_set, pm_labels_set):
                pmod_logits, pooling_output = self.model.bind_argu(words, char_ids, pos_ids, prot_ids, parse_input, pmod_inputs_first, pmod_inputs_second,)
                pmod_loss = self.bind_loss(pmod_logits.squeeze(1), pm_labels.float().cuda())
                if self.opt.get('pooling_l2', 0) > 0:
                    pmod_loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
                pmod_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                pmode_val = pmod_loss.item()

        regu_val = 0
        if len(regu_argus[0]) > 0:
            r_inputs_first_set, r_inputs_second_set, r_labels_set = common_utils.process_batch_br2(regu_argus, self.opt['cuda'], "train")
            for r_inputs_first, r_inputs_second, r_labels in zip(r_inputs_first_set, r_inputs_second_set, r_labels_set):
                regu_logits, pooling_output = self.model.bind_argu(words, char_ids, pos_ids, prot_ids, parse_input, r_inputs_first, r_inputs_second)
                regu_loss = self.bind_loss(regu_logits.squeeze(1), r_labels.float().cuda())
                if self.opt.get('pooling_l2', 0) > 0:
                    regu_loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
                regu_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
                self.optimizer.step()
                self.scheduler.step()
                regu_val = regu_loss.item()
        torch.cuda.empty_cache()
        return jprot_neg_log_likelihood.item(), trig_neg_log_likelihood.item(), loss_val, bind_val

    '''
    batch = file_name, words, genia_info, parse_info, protrig, entity_pair, gold_events
    genia_info = genia_words, pos_ids, prot_ids, jprot_labels, genia_jprotIds, trig_labels, trig_labelIds
    parse_info = parse_posIds, deprel, head
    '''
    def predict(self, epoch, batch, prev_filename, prev_jprot_set, prev_ctrig_dict, unsort=True):
        file_name, words, char_ids, prev_sent, next_sent, prot_dict, genia_input, parse_input, entity_pairs, bind_argus, pmod_argus, regu_argus, gold_events = common_utils.unpack_devbatch(batch, prev_jprot_set, prev_ctrig_dict, self.opt['cuda'])
        words, pos_ids, prot_ids, pred_cjprot_ids, pred_ctrig_ids, prot_labels, jprot_labels, jprot_ids, trig_labels, tlabel_ids = genia_input

        # forward
        self.jprot_model.eval()
        self.trig_model.eval()
        self.model.eval()

        #--- join protein 的识别情况 ---
        with torch.no_grad():
            dev_jprot_loss = self.jprot_model.neg_log_likelihood(words, char_ids, pos_ids, prot_ids, pred_cjprot_ids, jprot_ids)
            _, pred_jprot_ids = self.jprot_model(words, char_ids, pos_ids, prot_ids, pred_cjprot_ids)
        pred_jprot_labels = [self.id2jprot[jp] for jp in pred_jprot_ids]
        pred_jprotname_set = data_process.get_prot_name(words, pred_jprot_labels)

        #--- trigger 词的识别情况 ---
        pred_jprot_ids = common_utils.get_long_tensor([pred_jprot_ids], 1)
        if self.opt['cuda']:
            pred_jprot_ids = pred_jprot_ids.cuda()
        with torch.no_grad():
            dev_trig_loss = self.trig_model.neg_log_likelihood(words, char_ids, pos_ids, prot_ids, pred_jprot_ids, pred_ctrig_ids, tlabel_ids)
            _, pred_trig_ids = self.trig_model(words, char_ids, pos_ids, prot_ids, pred_jprot_ids, pred_ctrig_ids)

        #--- trigger 与 argument 关系识别 ---
        # 生成候选的 entity_pairs (trigger 与 argument 的关系)
        pred_trigs = [self.id2trig[t] for t in pred_trig_ids]
        trig_idx = common_utils.gen_trigidx(prot_dict)
        pred_trig_dict, predtrig_nametype_dict = data_process.gen_trig_dict_BIO(words, pred_trigs, trig_idx) ## 这里的gen_trig_dict有错误
        pred_entity_pairs_first, pred_entity_pairs_second, pred_entity_pairs_exams = data_process.gen_entity_pairs(self.opt, words[0], prev_sent[0], next_sent[0], pred_trig_dict, prot_dict)
        #-------------------------------------------------------------------------------------------------------
        ## 将预测出的jprot, trig转换出来存储到 prev_jprot_set, prev_ctrig_set中
        if file_name == prev_filename:
            prev_jprot_set.extend(pred_jprotname_set)
            prev_ctrig_dict.update(predtrig_nametype_dict)

        if len(pred_entity_pairs_first) > 0:
            dev_first_pair_inputs_set, dev_second_pair_input_set = common_utils.process_batch_pair((pred_entity_pairs_first, pred_entity_pairs_second), self.opt['cuda'], "devel")
            with torch.no_grad():
                pred_rela_seq = []
                for dev_first_pair_inputs, dev_second_pair_input in zip(dev_first_pair_inputs_set,dev_second_pair_input_set):
                    logits, h = self.model(words, char_ids, pos_ids, prot_ids, parse_input, dev_first_pair_inputs, dev_second_pair_input)
                    pred_rela = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
                    pred_rela_seq.extend(pred_rela)

                # 将预测结果生成事件
                pred_pair_relas = [self.id2rela[p] for p in pred_rela_seq]
            # --- binding argument是否在同一个事件中 ---
            pred_bind_argus_first, pred_bind_argus_second, pred_bind_argus_exams = gen_bind_example(self.opt, words[0], prev_sent[0], next_sent[0], pred_entity_pairs_exams, pred_pair_relas, "devel")  ## pred_res是预测出的序列
            pmod_argus_info, regu_argus_info = gen_regu_example(self.opt, words[0], prev_sent[0], next_sent[0], pred_entity_pairs_exams, pred_pair_relas, "devel")
            pred_pmod_argus_first, pred_pmod_argus_second, pred_pmod_argus_exams = pmod_argus_info
            pred_regu_argus_first, pred_regu_argus_second, pred_regu_argus_exams = regu_argus_info

            pred_bind_labels = []
            if len(pred_bind_argus_first) > 0:
                pred_bind_labels = self.pred_argu_rela2("devel", words, char_ids, pos_ids, prot_ids, parse_input, pred_bind_argus_first, pred_bind_argus_second)

            pred_pmod_labels = []
            if len(pred_pmod_argus_first) > 0:
                pred_pmod_labels = self.pred_argu_rela2("devel", words, char_ids, pos_ids, prot_ids, parse_input, pred_pmod_argus_first, pred_pmod_argus_second)

            pred_regu_labels = []
            if len(pred_regu_argus_first) > 0:
                pred_regu_labels = self.pred_argu_rela2("devel", words, char_ids, pos_ids, prot_ids, parse_input, pred_regu_argus_first, pred_regu_argus_second)

            pred_sent_events = gen_event(file_name, pred_entity_pairs_exams, pred_pair_relas, pred_bind_argus_exams, pred_bind_labels,
                                                     pred_pmod_argus_exams, pred_pmod_labels, pred_regu_argus_exams, pred_regu_labels)



            jprot_eval = (pred_jprot_labels, jprot_labels)
            trig_eval = (pred_trigs, trig_labels)
            rela_eval = (entity_pairs[3], entity_pairs[2], pred_entity_pairs_exams, pred_pair_relas)
            bind_eval = (bind_argus[3], bind_argus[2], pred_bind_argus_exams, pred_bind_labels)
            event_eval = (pred_sent_events, gold_events)
            torch.cuda.empty_cache()
            return file_name, dev_jprot_loss.item(), dev_trig_loss.item(), jprot_eval, trig_eval, rela_eval, bind_eval, event_eval

        else:
            jprot_eval = (pred_jprot_labels, jprot_labels)
            trig_eval = (pred_trigs, trig_labels)
            torch.cuda.empty_cache()
            return file_name, 10000, 10000, jprot_eval, trig_eval, None, None, None

    ## jprot, trigger, relation, binding, event
    ## PMID-2283805.a2
    def test(self, batch, prev_jprot_set, prev_ctrig_dict, prev_file_name, curr_trig_idx, curr_event_idx, curr_context, pred_event_num, unsort=True):
        file_name, old_sent, words, char_ids, prev_sent, next_sent, prot_dict, genia_input, parse_input = common_utils.unpack_testbatch(batch, prev_jprot_set, prev_ctrig_dict, self.opt['cuda'])
        genia_words, pos_ids, prot_ids, pred_cjprot_ids, pred_ctrig_ids, prot_labels = genia_input

        ## 判断是否要读取a1文件，得到protein的个数用于计算trigger的idx
        trig_idx, context = common_utils.is_read_a1_txt(prev_file_name, file_name, curr_trig_idx, curr_context)
        ## 计算event_idx
        event_idx = common_utils.record_event_idx(prev_file_name, file_name, curr_event_idx)
        if len(prot_dict) == 0:
            return file_name, trig_idx, event_idx, context, {}, []

        # forward
        self.jprot_model.eval()
        self.trig_model.eval()
        self.model.eval()

        # join protein 的识别情况
        with torch.no_grad():
            _, pred_jprot_ids = self.jprot_model(words, char_ids, pos_ids, prot_ids, pred_cjprot_ids)
        pred_jprot_labels = [self.id2jprot[jp] for jp in pred_jprot_ids]
        pred_jprotname_set = data_process.get_prot_name(words, pred_jprot_labels)

        # trigger 词的识别情况
        pred_jprot_ids = common_utils.get_long_tensor([pred_jprot_ids], 1)
        if self.opt['cuda']:
            pred_jprot_ids = pred_jprot_ids.cuda()
        with torch.no_grad():
            _, pred_trig_ids = self.trig_model(words, char_ids, pos_ids, prot_ids, pred_jprot_ids, pred_ctrig_ids)

        # 生成候选的 entity_pairs (trigger 与 argument 的关系)
        pred_trigs = [self.id2trig[t] for t in pred_trig_ids]
        pred_trig_dict, predtrig_nametype_dict, last_trig_idx = data_process.gen_testtrig_dict_bio(words, pred_trigs, trig_idx)
        common_utils.trig_context_posit(file_name, old_sent[0], words[0], prot_dict, pred_trig_dict, context)  ## 定位trigger在原句子中的位置, 进而计算其中全文中的位置
        pred_entity_pairs_first, pred_entity_pairs_second, pred_entity_pairs_exam = data_process.gen_entity_pairs(self.opt,
                                                                                                                  words[0],
                                                                                                                  prev_sent[0],
                                                                                                                  next_sent[0],
                                                                                                                  pred_trig_dict,
                                                                                                                  prot_dict)
        # -------------------------------------------------------------------------------------------------------
        ## 将预测出的jprot, trig转换出来存储到 prev_jprot_set, prev_ctrig_set中
        if file_name == prev_file_name:
            prev_jprot_set.extend(pred_jprotname_set)
            prev_ctrig_dict.update(predtrig_nametype_dict)

        ## pair_input, rela_Ids = process_pair(entity_pairs, pair_label, self.opt['cuda'])
        if len(pred_entity_pairs_first) > 0:
            test_first_pair_inputs_set, test_second_pair_input_set = common_utils.process_batch_pair((pred_entity_pairs_first, pred_entity_pairs_second), self.opt['cuda'], "test")
            with torch.no_grad():
                pred_rela_seq = []
                for test_first_pair_inputs, test_second_pair_input in zip(test_first_pair_inputs_set, test_second_pair_input_set):
                    logits, h = self.model(words, char_ids, pos_ids, prot_ids, parse_input, test_first_pair_inputs, test_second_pair_input)
                    pred_rela = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
                    pred_rela_seq.extend(pred_rela)

                # 将预测结果生成事件
                pred_pair_relas = [self.id2rela[p] for p in pred_rela_seq]
            # --- binding argument是否在同一个事件中 ---
            pred_bind_argus_first, pred_bind_argus_second, pred_bind_exams = gen_bind_example(self.opt,
                                                                                              words[0],
                                                                                              prev_sent[0],
                                                                                              next_sent[0],
                                                                                              pred_entity_pairs_exam,
                                                                                              pred_pair_relas,
                                                                                              "test")  ## pred_res是预测出的序列
            pred_pmod_argus, pred_regu_argus = gen_regu_example(self.opt,
                                                                words[0],
                                                                prev_sent[0],
                                                                next_sent[0],
                                                                pred_entity_pairs_exam,
                                                                pred_pair_relas,
                                                                "test")
            pred_pmod_argus_first, pred_pmod_argus_second, pred_pmod_exams = pred_pmod_argus
            pred_regu_argus_first, pred_regu_argus_second, pred_regu_exams = pred_regu_argus

            pred_bind_labels = []
            if len(pred_bind_argus_first) > 0:
                pred_bind_labels = self.pred_argu_rela2("test", words, char_ids, pos_ids, prot_ids, parse_input, pred_bind_argus_first, pred_bind_argus_second)

            pred_pmod_labels = []
            if len(pred_pmod_argus_first) > 0:
                pred_pmod_labels = self.pred_argu_rela2("test", words, char_ids, pos_ids, prot_ids, parse_input, pred_pmod_argus_first, pred_pmod_argus_second)

            pred_regu_labels = []
            if len(pred_regu_argus_first) > 0:
                pred_regu_labels = self.pred_argu_rela2("test", words, char_ids, pos_ids, prot_ids, parse_input, pred_regu_argus_first, pred_regu_argus_second)

            pred_write_events, _, last_event_idx = gen_test_event(file_name, pred_entity_pairs_exam, pred_pair_relas,
                                                                  pred_bind_exams, pred_bind_labels,
                                                                  pred_pmod_exams, pred_pmod_labels,
                                                                  pred_regu_exams, pred_regu_labels,
                                                                  event_idx, pred_event_num)
            return file_name, last_trig_idx, last_event_idx, context, pred_trig_dict, pred_write_events
        else:
            return file_name, last_trig_idx, event_idx, context, pred_trig_dict, None
    ## GPU 32G 的显存可以运行
    def pred_argu_rela(self, mode, words, char_ids, pos_ids, prot_ids, parse_input, pred_argus_first, pred_argus_second):
        inputs_first, inputs_second = common_utils.process_batch_br((pred_argus_first, pred_argus_second), self.opt['cuda'], mode)
        with torch.no_grad():
            logits, _ = self.model.bind_argu(words, char_ids, pos_ids, prot_ids, parse_input, inputs_first, inputs_second)
            output = torch.sigmoid(logits.squeeze(1))
            pred_labels = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
            pred_labels = pred_labels.cpu().numpy().tolist()
        return pred_labels

    ## GPU 12G 的显存可以运行
    def pred_argu_rela2(self, mode, words, char_ids, pos_ids, prot_ids, parse_input, pred_argus_first, pred_argus_second):
        inputs_first_set, inputs_second_set = common_utils.process_batch_br2((pred_argus_first, pred_argus_second), self.opt['cuda'], mode)
        with torch.no_grad():
            pred_labels = []
            for input_first, input_second in zip(inputs_first_set, inputs_second_set):
                bind_logits, _ = self.model.bind_argu(words, char_ids, pos_ids, prot_ids, parse_input, input_first, input_second)
                output = torch.sigmoid(bind_logits.squeeze(1))
                pred_label_ids = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
                pred_label_ids = pred_label_ids.cpu().numpy().tolist()
                pred_labels.extend(pred_label_ids)
        return pred_labels