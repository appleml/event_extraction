from data_prepare.entity import Trigger, Protein, Event
from data_prepare.file import sent_event
import copy
import utils.constant as constant
from metric.metric_util import trig_isequal, event_isequal
'''
针对句子中实体对预测的结果构建event
trigargu_pair：一个句子中所以实体对集合: (event_idx_set, trig, trig_type_idx, trig_posit, prot, argu_type_idx, argu_posit, rela_type
pair_preds: 是针对 trigargu_pair 预测的结果序列
exam_input = (trig_type_str, [trig_typeId], trig_position1, [argu1_typeId], argu_position1, [argu2_typeId], argu_position2)
exam_seq 是针对 exam_input的 预测序列
arguargu_pair = (trig_idx+"&"+argu1.prot_idx+"&"+argu2.prot_idx, trig1, argu1, label1, argu2, label2)
'''
def gen_event(file_name, entity_pairs, pair_preds, pred_bind_exams, pred_bind_labels, pred_pmod_exams, pred_pmod_labels, pred_regu_exams, pred_regu_labels):
    simp_pairs = list()
    bind_pairs = list()
    pmod_theme_pairs = list()
    regu_theme_pairs = list()

    for trigargu_pair, pred_label in zip(entity_pairs, pair_preds):
        _, _, trig, argu, _, _ = trigargu_pair
        trig_type = trig.trig_type
        assert isinstance(trig_type, str) and isinstance(pred_label, str)
        if pred_label == 'Theme' and trig_type in constant.SIMP_TYPE:
            simp_pairs.append((trig, argu, pred_label))
        elif pred_label == "Theme" and trig_type in constant.BIND_TYPE:
            bind_pairs.append((trig, argu, pred_label))
        elif pred_label == "Theme" and trig_type in constant.PMOD_TYPE:
            pmod_theme_pairs.append(((trig, argu, pred_label)))
        elif pred_label == "Theme" and trig_type in constant.REGU_TYPE:
            regu_theme_pairs.append((trig, argu, pred_label))


    ## 感觉有问题啊?? 如果trigger是预测出的, 那么是没有trig_idx
    simp_event_dict = build_simple(simp_pairs)
    bind_event_dict = build_binding(bind_pairs, pred_bind_exams, pred_bind_labels)
    pmod_event_dict = build_protmode(pmod_theme_pairs, pred_pmod_exams, pred_pmod_labels)
    regu_event_dict = build_regulation2(regu_theme_pairs, simp_event_dict, bind_event_dict, pmod_event_dict, pred_regu_exams, pred_regu_labels)

    sent_evet = sent_event(file_name)
    sent_evet.simp_event_set = get_event(simp_event_dict)
    sent_evet.bind_event_set = get_event(bind_event_dict)
    sent_evet.pmod_event_set = get_event(pmod_event_dict)
    sent_evet.regu_event_set = get_event(regu_event_dict)

    return sent_evet

## exam_input = (trig_type_str, [trig_typeId], trig_position1, [argu1_typeId], argu_position1, [argu2_typeId], argu_position2)
## pair_event = (trig1, argu1, label1, argu2, label2)

# 6月3号所写
def build_binding(bind_pairs, arguargu_pair, arguargu_label):
    sent_bind_events = dict()
    bind_two_events = dict()
    if len(arguargu_label) > 0:
        for bind_pair, pred_tag in zip(arguargu_pair, arguargu_label):
            trig1, argu1, label1, argu2, label2 = bind_pair
            trig_type = trig1.trig_type
            assert trig_type == 'Binding'
            if pred_tag == 1:
                b_event = Event()
                b_event.add_fargu_entity(trig1, argu1, label1)
                b_event.add_sargu_entity(argu2, label2)
                mark = trig1.trig_idx+"&"+argu1.prot_idx+"&"+argu2.prot_idx
                bind_two_events[mark] = b_event
                store_event(trig1.trig_idx, b_event, sent_bind_events)

    for bind_pair in bind_pairs:
        trig, argu, pred_label = bind_pair
        trig_idx = trig.trig_idx
        argu_idx = argu.prot_idx
        flag = False
        for idx_sign, _ in bind_two_events.items():
            sign_seg = idx_sign.split("&")
            # 要判断这里是否有问题
            if trig_idx == sign_seg[0] and argu_idx in [sign_seg[1], sign_seg[2]]:
                flag = True
                break
        if flag == False:
            b_event = Event()
            b_event.add_fargu_entity(trig, argu, pred_label)
            store_event(trig_idx, b_event, sent_bind_events)
    return sent_bind_events

def build_protmode(pmod_theme_pairs, pred_pmod_exams, pred_pmod_labels):
    sent_pmod_events = dict()
    pmod_two_events = dict() # 中介
    if len(pred_pmod_labels) > 0:
        for pmod_exam, pmod_label in zip(pred_pmod_exams, pred_pmod_labels):
            trig1, argu1, label1, argu2, label2 = pmod_exam
            assert label1 == "Theme" and label2 == "Cause"
            trig_type = trig1.trig_type
            assert trig_type in constant.PMOD_TYPE
            if pmod_label == 1:
                r_event = Event()
                r_event.add_fargu_entity(trig1, argu1, label1)
                r_event.add_sargu_entity(argu2, label2)
                if isinstance(argu1, Protein) and isinstance(argu2, Protein):
                    mark = trig1.trig_idx + "&" + argu1.prot_idx + "&" + argu2.prot_idx
                elif isinstance(argu1, Protein) and isinstance(argu2, Trigger):
                    mark = trig1.trig_idx + "&" + argu1.prot_idx + "&" + argu2.trig_idx
                elif isinstance(argu1, Trigger) and isinstance(argu2, Protein):
                    mark = trig1.trig_idx + "&" + argu1.trig_idx + "&" + argu2.prot_idx
                elif isinstance(argu1, Trigger) and isinstance(argu2, Trigger):
                    mark = trig1.trig_idx + "&" + argu1.trig_idx + "&" + argu2.trig_idx

                pmod_two_events[mark] = r_event
                store_event(trig1.trig_idx, r_event, sent_pmod_events)

    for pmod_pair in pmod_theme_pairs:
        trig, argu, pred_label = pmod_pair
        trig_idx = trig.trig_idx
        if isinstance(argu, Protein):
            argu_idx = argu.prot_idx
        elif isinstance(argu, Trigger):
            argu_idx = argu.trig_idx

        flag = False
        for idx_sign, _ in pmod_two_events.items():
            sign_seg = idx_sign.split("&")
            if trig_idx == sign_seg[0] and argu_idx in [sign_seg[1], sign_seg[2]]:
                flag = True
                break
        if flag == False:
            r_event = Event()
            r_event.add_fargu_entity(trig, argu, pred_label)
            store_event(trig_idx, r_event, sent_pmod_events)

    return sent_pmod_events

#简单事件只能是Theme类型
def build_simple(pair_set):
    sent_simp_events = dict() # key是trig_idx, value是list
    for pair in pair_set:
        trig, argu, pred_label = pair
        assert isinstance(argu, Protein)
        simp_event = Event()
        simp_event.add_fargu_entity(trig, argu, pred_label)
        trig_idx = trig.trig_idx
        store_event(trig_idx, simp_event, sent_simp_events)
    return sent_simp_events

def store_event(trig_idx, event, event_dict):
    if trig_idx in event_dict.keys():
        events_set = event_dict[trig_idx]
        events_set.append(event)
    else:
        event_dict[trig_idx] = [event]

## 构建复杂事件，注意的点是：论元可能是trigger,要小心应对
## simp_event_dict, bind_event_dict 是已经构建好 simp events 和 bind events
def build_regulation(regu_theme_pairs, regu_cause_pairs, simp_event_dict, bind_event_dict):
    sent_regu_events = dict()
    for trig_idx, regu_theme_set in regu_theme_pairs.items():
        if trig_idx not in regu_cause_pairs.keys():
            for regu_theme in regu_theme_set:
                trig, argu, label = regu_theme
                assert label == "Theme"
                regu_evet = Event()
                regu_evet.add_fargu_entity(trig, argu, label)
                store_event(trig_idx, regu_evet, sent_regu_events)
        else:
            regu_cause_set = regu_cause_pairs[trig_idx]
            for regu_theme in regu_theme_set:
                trig1, argu1, label1 = regu_theme
                for regu_cause in regu_cause_set:
                    trig2, argu2, label2 = regu_cause
                    assert trig_isequal(trig1, trig2) # 其实这里只需要比较trig_idx即可
                    assert label1 == "Theme" and label2 == "Cause"
                    r_event = Event()
                    r_event.add_fargu_entity(trig1, argu1, label1)
                    r_event.add_sargu_entity(argu2, label2)
                    store_event(trig_idx, r_event, sent_regu_events)

    ##sent_regu_events 中的 argu 有可能是trigger, 将其挑选出来
    regu_event_dict, unwell_regu_event = classify_regu_event(sent_regu_events)
    # unwell_regu_event里的event, 论元需要由trigger替换为event
    rectify(simp_event_dict, bind_event_dict, regu_event_dict, unwell_regu_event)

    return regu_event_dict

## sent_regu_events 中的argu 有可能是trigger, 将其挑选出来
def classify_regu_event(sent_regu_events):
    well_regu_events = dict()
    unwell_regu_events = dict()  # 此处里面的事件论元是trigger, 需要将其修正
    for trig_idx, regu_event_set in sent_regu_events.items():
        for regu_event in regu_event_set:
            argu1 = regu_event.first_argu
            argu2 = regu_event.second_argu
            if argu2 == None:
                if isinstance(argu1, Protein):
                    store_event(trig_idx, regu_event, well_regu_events)
                else:
                    assert isinstance(argu1, Trigger)
                    store_event(trig_idx, regu_event, unwell_regu_events)
            else:
                if isinstance(argu1, Protein) and isinstance(argu2, Protein):
                    store_event(trig_idx, regu_event, well_regu_events)
                else:
                    assert isinstance(argu1, Trigger) or isinstance(argu2, Trigger)
                    store_event(trig_idx, regu_event, unwell_regu_events)
    return well_regu_events, unwell_regu_events

## simp_event_dict, bind_event_dict, regu_event_dict是已经完善好的事件
## unwell_regu_events 中的event需要修正论元
## 修正成功的存放到 regu_event_dict，没有修正成功的存放到left_regu_event
def rectify(simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, unwell_regu_events):
    i = 0
    while len(unwell_regu_events) > 0 and i < 1:
        left_regu_event = dict() #修正不了的存放在该字典中
        for trig_idx, r_event_set in unwell_regu_events.items():
            for r_event in r_event_set:
                argu1 = r_event.first_argu
                argu2 = r_event.second_argu
                if argu2 == None:
                    assert isinstance(argu1, Trigger)
                    argutrig1_idx = argu1.trig_idx
                    left_regu_dict = can_replace(trig_idx, r_event, argutrig1_idx, simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, True)
                    merge_dict(left_regu_event, left_regu_dict)

                else:
                    if isinstance(argu1, Protein) and isinstance(argu2, Trigger):
                        argutrig2_idx = argu2.trig_idx
                        left_regu_dict = can_replace(trig_idx, r_event, argutrig2_idx, simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, False)
                        merge_dict(left_regu_event, left_regu_dict)

                    elif isinstance(argu1, Trigger) and isinstance(argu2, Protein):
                        argutrig1_idx = argu1.trig_idx
                        left_regu_dict = can_replace(trig_idx, r_event, argutrig1_idx, simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, True)
                        merge_dict(left_regu_event, left_regu_dict)

                    elif isinstance(argu1, Trigger) and isinstance(argu2, Event):
                        argutrig1_idx = argu1.trig_idx
                        left_regu_dict = can_replace(trig_idx, r_event, argutrig1_idx, simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, True)
                        merge_dict(left_regu_event, left_regu_dict)

                    elif isinstance(argu1, Event) and isinstance(argu2, Trigger):
                        argutrig2_idx = argu2.trig_idx
                        left_regu_dict = can_replace(trig_idx, r_event, argutrig2_idx, simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, False)
                        merge_dict(left_regu_event, left_regu_dict)

                    elif isinstance(argu1, Trigger) and isinstance(argu2, Trigger):# 总觉得有问题
                        ## 先替换第一个论元，再替换第二个论元
                        argutrig1_idx = argu1.trig_idx
                        left_regu_dict1 = can_replace(trig_idx, r_event, argutrig1_idx, simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, True)
                        # 第一个论元替换成功，替换第二个论元
                        if len(left_regu_dict1) > 1:
                            for trig_idx, left_regu_set in left_regu_dict1.items():
                                for left_regu in left_regu_set:
                                    argutrig2_idx = argu2.trig_idx
                                    left_regu_dict2 = can_replace(trig_idx, left_regu, argutrig2_idx, simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, False)
                                    merge_dict(left_regu_event, left_regu_dict2)

                        else: # 第一个论元没替换成功，直接替换第二个论元
                            argutrig2_idx = argu2.trig_idx
                            left_regu_dict = can_replace(trig_idx, r_event, argutrig2_idx, simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, False)
                            merge_dict(left_regu_event, left_regu_dict)
                    else:
                        print("应该没有这种问题吧")

        i += 1
        unwell_regu_events = copy.deepcopy(left_regu_event)


## r_event是unwell_regu_events中的元素，根据is_first替换第一个论元还是第二个论元
## simp_events_dict, bind_events_dict, regu_events_dict都是完美的事件
## trig_idx 是 r_event 中 trigger的 idx
'''
trig_idx 是当前事件
argutrig_idx 是当前事件的论元的
'''
def can_replace(trig_idx, r_event, argutrig_idx, simp_events_dict, bind_events_dict, pmod_event_dict, regu_events_dict, is_first):
    bad_events = dict() #替换不成功的放在这里
    assert trig_idx != argutrig_idx
    if argutrig_idx in simp_events_dict.keys():
        arguevent_set = simp_events_dict[argutrig_idx]
        well_events, unwell_events = argu_replace(trig_idx, r_event, arguevent_set, is_first)
        merge_dict(regu_events_dict, well_events)
        merge_dict(bad_events, unwell_events)
    elif argutrig_idx in bind_events_dict.keys():
        arguevent_set = bind_events_dict[argutrig_idx]
        well_events, unwell_events = argu_replace(trig_idx, r_event, arguevent_set, is_first)
        merge_dict(regu_events_dict, well_events)
        merge_dict(bad_events, unwell_events)
    elif argutrig_idx in pmod_event_dict.keys():
        arguevent_set = pmod_event_dict[argutrig_idx]
        well_events, unwell_events = argu_replace(trig_idx, r_event, arguevent_set, is_first)
        merge_dict(regu_events_dict, well_events)
        merge_dict(bad_events, unwell_events)
    elif argutrig_idx in regu_events_dict.keys():
        argu_event_set = regu_events_dict[argutrig_idx]
        well_events, unwell_events= argu_replace(trig_idx, r_event, argu_event_set, is_first)
        merge_dict(regu_events_dict, well_events)
        merge_dict(bad_events, unwell_events)
    else:
        store_event(trig_idx, r_event, bad_events)

    return bad_events

# trig_idx是r_event中trigger的idx
# is_first是替换r_event中第一个论元还是第二个论元，
def argu_replace(trig_idx, r_event, arguevent_set, is_first):
    well_events = dict()
    unwell_events = dict()
    for argu_event in arguevent_set:
        if is_first == True: ## 虽然替换的是第一个论元，但也不能丢掉第二个论元
            new_event = Event()
            new_event.add_fargu_entity(r_event.event_trig, argu_event, r_event.first_argu_type)
            second_argu = r_event.second_argu
            if second_argu == None:
                store_event(trig_idx, new_event, well_events)
            else:
                new_event.add_sargu_entity(second_argu, r_event.second_argu_type)
                if isinstance(second_argu, Protein) or isinstance(second_argu, Event):
                    store_event(trig_idx, new_event, well_events)
                else:
                    assert isinstance(second_argu, Trigger)
                    store_event(trig_idx, new_event, unwell_events)
        else: #替换第二个论元
            new_event = Event()
            new_event.add_fargu_entity(r_event.event_trig, r_event.first_argu, r_event.first_argu_type)
            new_event.add_sargu_entity(argu_event, r_event.second_argu_type)
            if isinstance(r_event.first_argu, Protein) or isinstance(r_event.first_argu, Event):
                store_event(trig_idx, new_event, well_events)
            else:
                assert isinstance(r_event.first_argu, Trigger)
                store_event(trig_idx, new_event, unwell_events)
    return well_events, unwell_events

## 两个dict合并
def merge_dict(origin_event_dict, new_event_dict):
    for new_trig_idx, new_event_set in new_event_dict.items():
        if new_trig_idx in origin_event_dict.keys():
            origin_event_set = origin_event_dict[new_trig_idx]
            origin_event_set.extend(new_event_set)
        else:
            origin_event_dict[new_trig_idx] = new_event_set

##从字典中得到event集合
## 验证该函数是否有问题
def get_event(sent_event_dict):
    event_set = list()
    for _, evet_set in sent_event_dict.items():
        event_set.extend(evet_set)
    return event_set

###################################################################
'''
参数 pred_event_num 为了统计一个文件中总共有简单事件, binding事件和regu事件的个数
与devel不同的时, 需要考虑 event_idx 和 pred_event_num
返回值: write_events, pred_sent_events 有什么区别
write_events : 用于写出到.a2文件中
pred_sent_events : 那这个呢?
'''
def gen_test_event(file_name, entity_pairs, pair_labels, pred_bind_exam, pred_bind_labels, pred_pmod_exams, pred_pmod_labels, pred_regu_exam, pred_regu_labels, event_idx, pred_event_num):
    simp_pairs = list()
    bind_pairs = list()
    pmod_theme_pairs = list()
    regu_theme_pairs = list() ## key 是 trig_idx, value要么是event, 要么是 (trig, argu, pred_label)

    for trigargu_pair, pred_label in zip(entity_pairs, pair_labels):
        _, _, trig, argu, _, _ = trigargu_pair
        trig_type = trig.trig_type
        assert isinstance(trig_type, str) and isinstance(pred_label, str)
        if pred_label == 'Theme' and trig_type in constant.SIMP_TYPE:
            simp_pairs.append((trig, argu, pred_label))
        elif pred_label == "Theme" and trig_type in constant.BIND_TYPE:
            bind_pairs.append((trig, argu, pred_label))
        elif pred_label == "Theme" and trig_type in constant.PMOD_TYPE:
            pmod_theme_pairs.append((trig, argu, pred_label))
        elif pred_label == "Theme" and trig_type in constant.REGU_TYPE:
            regu_theme_pairs.append((trig, argu, pred_label))

    simp_event_dict = build_simple(simp_pairs)
    bind_event_dict = build_binding(bind_pairs, pred_bind_exam, pred_bind_labels,)
    pmod_event_dict = build_protmode(pmod_theme_pairs, pred_pmod_exams, pred_pmod_labels)
    regu_event_dict = build_regulation2(regu_theme_pairs, simp_event_dict, bind_event_dict, pmod_event_dict, pred_regu_exam, pred_regu_labels)

    simp_event_set = get_event(simp_event_dict)
    bind_event_set = get_event(bind_event_dict)
    pmod_event_set = get_event(pmod_event_dict)
    regu_event_set = get_event(regu_event_dict)

    pred_event_num['simp'] += len(simp_event_set)
    pred_event_num['bind'] += len(bind_event_set)
    pred_event_num['pmod'] += len(pmod_event_set)
    pred_event_num['regu'] += len(regu_event_set)
    pred_sent_events = gen_sent_event(file_name, simp_event_set, bind_event_set, pmod_event_set, regu_event_set)

    # 完善事件的各种信息， event的id号， argu的id号
    all_events = dict()
    merge_dict(all_events, simp_event_dict)
    merge_dict(all_events, bind_event_dict)
    merge_dict(all_events, pmod_event_dict)
    merge_dict(all_events, regu_event_dict)
    ## 按照key值进行排序, 给 all event都赋值event_idx
    last_event_idx = sorted_event(all_events, event_idx)
    ## 遍历所有event, 论元的idx赋值
    assign_arguidx(all_events) ## 这里就有问题了
    ## 用于写出的话，没必要分simp_event, bind_event, regu_event
    write_events = get_event(all_events)

    assert len(write_events) == len(simp_event_set) + len(bind_event_set) + len(pmod_event_set) + len(regu_event_set)
    return write_events, pred_sent_events, last_event_idx

from data_prepare import file
def gen_sent_event(file_name, simp_events, bind_events, pmod_events, regu_events):
    sent_events = file.sent_event(file_name)
    sent_events.add_simp(simp_events)
    sent_events.add_bind(bind_events)
    sent_events.add_pmod(pmod_events)
    sent_events.add_regu(regu_events)
    return sent_events

## 思路：将所有的事件的event_idx都赋值，
def sorted_event(all_events, event_idx):
    for key in sorted(all_events.keys()):
        event_set = all_events[key]
        for event in event_set:
            event.event_idx = "E"+str(event_idx)
            event_idx += 1
    return event_idx

## 论元的idx赋值
## 检查是否赋值成功
def assign_arguidx(all_events):
    for trig_idx, event_set in all_events.items():
        for evet in event_set:
            first_argu = evet.first_argu
            second_argu = evet.second_argu
            if evet.event_trig_idx == "":
                evet.event_trig_idx = evet.event_trig.trig_idx
            # 为first_argu_idx赋值
            if isinstance(first_argu, Event):
                first_arguevent_idx = arguidx(first_argu, all_events)
                evet.first_argu_idx = first_arguevent_idx
            elif isinstance(first_argu, Protein):
                first_arguprot_idx = first_argu.prot_idx
                evet.first_argu_idx = first_arguprot_idx
            # 为second_argu_idx赋值
            if isinstance(second_argu, Event): ## first_argu和second_argu是独立的
                second_arguevent_idx = arguidx(second_argu, all_events)
                evet.second_argu_idx = second_arguevent_idx
            elif isinstance(second_argu, Protein):
                second_arguprot_idx = second_argu.prot_idx
                evet.second_argu_idx = second_arguprot_idx

            # 检验下first_argu_idx 和 second_argu_idx：
            if second_argu != None:
                assert evet.first_argu_idx != ""
                assert evet.second_argu_idx != ""
            else:
                assert evet.first_argu_idx != ""

## 待检验
def arguidx(arguevent, all_event):
    arguevent_trigidx = arguevent.event_trig_idx
    assert arguevent_trigidx != ""
    maybe_event_set = all_event[arguevent_trigidx]
    res = ""
    for maybe_event in maybe_event_set:
        flag = event_isequal(maybe_event, arguevent)
        if flag == True:
            arguevent.event_idx = maybe_event.event_idx
            res = maybe_event.event_idx
            break
    assert res != ""
    return res

###################################################################
'''
暂且放弃两个参数: regu_theme_pairs 和 regu_cause_pairs
'''
def build_regulation2(regu_theme_pairs, simp_event_dict, bind_event_dict, pmod_event_dict, pred_regu_exam, pred_regu_labels):
    sent_regu_events = dict()
    regu_two_events = dict()
    if len(pred_regu_labels) > 0:
        for pred_regu, pred_label in zip(pred_regu_exam, pred_regu_labels):
            trig1, argu1, label1, argu2, label2 = pred_regu
            assert label1 == "Theme" and label2 == "Cause"
            trig_type = trig1.trig_type
            assert trig_type in constant.REGU_TYPE
            if pred_label == 1:
                r_event = Event()
                r_event.add_fargu_entity(trig1, argu1, label1)
                r_event.add_sargu_entity(argu2, label2)
                if isinstance(argu1, Protein) and isinstance(argu2, Protein):
                    mark = trig1.trig_idx + "&" + argu1.prot_idx + "&" + argu2.prot_idx
                elif isinstance(argu1, Protein) and isinstance(argu2, Trigger):
                    mark = trig1.trig_idx + "&" + argu1.prot_idx + "&" + argu2.trig_idx
                elif isinstance(argu1, Trigger) and isinstance(argu2, Protein):
                    mark = trig1.trig_idx + "&" + argu1.trig_idx + "&" + argu2.prot_idx
                elif isinstance(argu1, Trigger) and isinstance(argu2, Trigger):
                    mark = trig1.trig_idx + "&" + argu1.trig_idx + "&" + argu2.trig_idx

                regu_two_events[mark] = r_event
                store_event(trig1.trig_idx, r_event, sent_regu_events)

    for regu_pair in regu_theme_pairs:
        trig, argu, pred_label = regu_pair
        trig_idx = trig.trig_idx
        if isinstance(argu, Protein):
            argu_idx = argu.prot_idx
        elif isinstance(argu, Trigger):
            argu_idx = argu.trig_idx

        flag = False
        for idx_sign, _ in regu_two_events.items():
            sign_seg = idx_sign.split("&")
            # 要判断这里是否有问题
            if trig_idx == sign_seg[0] and argu_idx in [sign_seg[1], sign_seg[2]]:
                flag = True
                break
        if flag == False:
            r_event = Event()
            r_event.add_fargu_entity(trig, argu, pred_label)
            store_event(trig_idx, r_event, sent_regu_events)

    ##sent_regu_events 中的 argu 有可能是trigger, 将其挑选出来
    regu_event_dict, unwell_regu_event = classify_regu_event(sent_regu_events)
    # unwell_regu_event里的event, 论元需要由trigger替换为event
    rectify(simp_event_dict, bind_event_dict, pmod_event_dict, regu_event_dict, unwell_regu_event)

    return regu_event_dict