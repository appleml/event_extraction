from data_prepare.entity import Protein, Trigger, Event
from utils import constant
'''
将 string 形式的 protein 信息转换为实体
# T1 Protein S1 19 49 19 49 interferon regulatory factor 4
(prot_idx, prot_name, prot_oldchar_start, prot_oldchar_end, prot_newchar_start, prot_newchar_end, prot_start, prot_end):
'''
def prot_str2entity(old_sent, new_sent, str_prot_set):
    prot_dict = dict()
    for str_prot in str_prot_set:
        prot_info = str_prot.split()
        new_start, new_end, word_start, word_end = relocate(old_sent, new_sent, int(prot_info[4]), int(prot_info[5]))
        prot_entity = Protein(prot_info[1], " ".join(prot_info[8:]), int(prot_info[4]), int(prot_info[5]), new_start, new_end, word_start, word_end)
        prot_entity.prot_context_start = int(prot_info[6])
        prot_entity.prot_context_end = int(prot_info[7])
        prot_entity.prot_mark = prot_info[3]
        prot_dict[prot_info[1]] = prot_entity
    return prot_dict

'''
@ T20 Regulation S2 160 172 306 318 deregulation
(trig_idx, trig_name, trig_type, trig_oldchar_start, trig_oldchar_end, trig_newchar_start, trig_newchar_end, trig_start, trig_end)
'''
def trig_str2entity(old_sent, new_sent, str_trig_set):
    trig_dict = dict()
    for str_trig in str_trig_set:
        trig_info = str_trig.split()
        new_start, new_end, word_start, word_end = relocate(old_sent, new_sent, int(trig_info[4]), int(trig_info[5]))
        trig_entity = Trigger(trig_info[1], " ".join(trig_info[8:]), trig_info[2], int(trig_info[4]), int(trig_info[5]), new_start, new_end, word_start, word_end)
        trig_entity.trig_context_start = int(trig_info[6])
        trig_entity.trig_context_end = int(trig_info[7])
        trig_entity.trig_mark = trig_info[3]
        trig_dict[trig_info[1]] = trig_entity
    return trig_dict

## % E14 Gene_expression:T31 Theme:T14
from data_prepare import file
def event_str2entity(file_name, strevent_set, trig_dict, prot_dict):
    event_dict = dict()
    for evet in strevent_set:
        evet_info = evet.split()
        etrig_info = evet_info[2].split(":")
        fargu_info = evet_info[3].split(":")
        if len(evet_info) == 4:  # 只有一个论元
            event_entity = Event()
            event_entity.add_basic_info(evet_info[1], etrig_info[1], etrig_info[0], fargu_info[1], fargu_info[0])
            event_dict[evet_info[1]] = event_entity

        elif len(evet_info) == 5: # 两个论元的情况
            event_entity = Event()
            event_entity.add_basic_info(evet_info[1], etrig_info[1], etrig_info[0], fargu_info[1], fargu_info[0])
            sargu_info = evet_info[4].split(":")
            if sargu_info[0].startswith("Theme"):
                event_entity.second_argu_idx = sargu_info[1]
                event_entity.second_argu_type = "Theme"

            elif sargu_info[0].startswith("Cause"):
                event_entity.second_argu_idx = sargu_info[1]
                event_entity.second_argu_type = sargu_info[0]
            else:
                print(sargu_info[0], "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            event_dict[evet_info[1]] = event_entity

        else: # 有一个情况, binding带了多个论元(不止两个)
            event_entity = Event()
            ## 第一个论元赋值
            event_entity.add_basic_info(evet_info[1], etrig_info[1], etrig_info[0], fargu_info[1], fargu_info[0])

            ## 第二个论元赋值
            sargu_info = evet_info[4].split(":")
            if sargu_info[0].startswith("Theme"):
                event_entity.second_argu_idx = sargu_info[1]
                event_entity.second_argu_type = "Theme"

            elif sargu_info[0].startswith("Cause"):
                event_entity.second_argu_idx = sargu_info[1]
                event_entity.second_argu_type = sargu_info[0]
            else:
                print(sargu_info[0], "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

            ## 多于两个论元的其他论元
            for other_argu in evet_info[5:]:
                other_arguinfo = other_argu.split(":")
                other_argu_idx = other_arguinfo[1]
                other_argu_type = other_arguinfo[0]
                event_entity.other_argu_info[other_argu_idx] = other_argu_type ## 怎么记录论元呢

            event_dict[evet_info[1]] = event_entity

    simp_event_set = list()
    bind_event_set = list()
    pmod_event_set = list()
    regu_event_set = list()
    # 添加event的trig信息以及论元信息
    for event_idx, evet in event_dict.items():
        etrig_idx = evet.event_trig_idx
        etrig = trig_dict[etrig_idx]
        evet.event_trig = etrig
        fargu_idx = evet.first_argu_idx
        if fargu_idx.startswith('T'):
            fargu = prot_dict[fargu_idx]
            evet.first_argu = fargu
        elif fargu_idx.startswith('E'):
            fargu = event_dict[fargu_idx]
            evet.first_argu = fargu

        sargu_idx = evet.second_argu_idx
        if sargu_idx != '' and sargu_idx.startswith('T'):
            sargu = prot_dict[sargu_idx]
            evet.second_argu = sargu
        elif sargu_idx != '' and sargu_idx.startswith('E'):
            sargu = event_dict[sargu_idx]
            evet.second_argu = sargu

        ## other_argu_entity
        if len(evet.other_argu_info) > 0:
            for argu_idx, argu_type in evet.other_argu_info.items(): ## 论元一定是蛋白质
                oargu = prot_dict[argu_idx]
                evet.other_argu_entity[argu_idx] = oargu

        evet_type = evet.event_type
        if evet_type in constant.SIMP_TYPE:
            simp_event_set.append(evet)
        elif evet_type in constant.BIND_TYPE:
            bind_event_set.append(evet)
        elif evet_type in constant.PMOD_TYPE:
            pmod_event_set.append(evet)
        elif evet_type in constant.REGU_TYPE:
            regu_event_set.append(evet)

    gold_sent_events = file.sent_event(file_name)
    gold_sent_events.add_simp(simp_event_set)
    gold_sent_events.add_bind(bind_event_set)
    gold_sent_events.add_pmod(pmod_event_set)
    gold_sent_events.add_regu(regu_event_set)

    return event_dict, gold_sent_events

'''
将 string 形式的 trigger 信息转换为实体
@ T18 Negative_regulation S1 0 15 0 15 Down-regulation
# (trig_idx, trig_name, trig_type, trig_oldchar_start, trig_oldchar_end, trig_newchar_start, trig_newchar_end, trig_start, trig_end):
trig_dict中key是trigger名, value是trig_entity
该方法不用于处理多个词组成的trigger
'''
def process_strtrig_one(old_sent, new_sent, str_trig_set):
    trig_dict = dict()
    multi_trigid_set =[]
    for str_trig in str_trig_set:
        trig_info = str_trig.split()
        specialTrigType = ["Entity", 'Anaphora']
        if len(trig_info) <= 9:
            if trig_info[2] not in specialTrigType:
                new_start, new_end, word_startId, word_endId = relocate(old_sent, new_sent, int(trig_info[4]), int(trig_info[5]))
                trig_entity = Trigger(trig_info[1], trig_info[8], trig_info[2], int(trig_info[4]), int(trig_info[5]), new_start, new_end, word_startId, word_endId)
                trig_dict[trig_info[1]] = trig_entity

        elif len(trig_info) > 9:
            if trig_info[2] not in specialTrigType:
                multi_trigid_set.append(trig_info[1])
    return trig_dict, multi_trigid_set

'''
一个词的trigger 和 多次词的trigger都要进行处理
'''
specialTrigType = ["Entity", 'Anaphora']
def process_strtrig_two(old_sent, new_sent, str_trig_set):
    trig_dict = dict()
    for str_trig in str_trig_set:
        trig_info = str_trig.split()
        if trig_info[2] not in specialTrigType:
            new_start, new_end, word_startId, word_endId = relocate(old_sent, new_sent, int(trig_info[4]), int(trig_info[5]))
            trig_entity = Trigger(trig_info[1], ' '.join(trig_info[8:]), trig_info[2], int(trig_info[4]), int(trig_info[5]), new_start, new_end, word_startId, word_endId)
            trig_dict[trig_info[1]] = trig_entity
    assert len(trig_dict) == len(str_trig_set)
    return trig_dict

# 根据trigger或protein在old_sent中的位置确定在new_sent中的位置，并且确定出在new_sent中的单词位置
def relocate(old_sent, new_sent, old_start, old_end):
    # 字符的位置
    new_start = locating(old_sent, new_sent, old_start, is_start=True)
    new_end = locating(old_sent, new_sent, old_end, is_start=False)
    assert old_sent[old_start:old_end].replace(" ", "") == new_sent[new_start:new_end].replace(" ", "")

    new_list = new_sent.split()
    # 单词的位置
    if new_start != 0 and new_sent[new_start-1] != " ":
        word_startId = len(new_sent[:new_start].split())-1
    else:
        word_startId = len(new_sent[:new_start].split())

    word_endId = len(new_sent[:new_end].split())-1
    # if new_sent[new_start-1] != " " and new_sent[new_end] != " ":
    #     assert " ".join(new_list[word_startId:word_endId + 1])[:new_end - new_start] == new_sent[new_start:new_end]
    #
    # elif new_sent[new_end] != " ":
    #     assert " ".join(new_list[word_startId:word_endId+1])[:new_end-new_start] == new_sent[new_start:new_end]
    # elif new_start != 0 and new_sent[new_start-1] != " ":
    #     print(new_end-new_start)
    #     print(" ".join(new_list[word_startId:word_endId + 1])[-(new_end - new_start):])
    #     assert " ".join(new_list[word_startId:word_endId + 1])[-(new_end - new_start):] == new_sent[new_start:new_end]
    assert word_startId <=word_endId
    return new_start, new_end, word_startId, word_endId

# 根据字符在old_sent中的位置确定在new_sent中的位置
# is_start是标示目前是计算单词的开始字符位置还是结束字符位置
def locating(old_sent, new_sent, locate, is_start):
    old_str = old_sent[:locate]
    new_str = new_sent[:locate]
    old_str1 = old_sent[:locate].replace(" ", "")
    new_str1 = new_sent[:locate].replace(" ", "")
    new_locate = locate
    if old_str1 == new_str1 and old_str != new_str:
        new_locate += 1

    while old_str1 != new_str1:
        new_locate += 1
        new_str1 = new_sent[:new_locate].replace(" ", "")

    new_str = new_sent[:new_locate]
    if old_str.endswith(" ") and not new_str.endswith(" "):
        new_locate += 1
        new_str1 = new_sent[:new_locate].replace(" ", "")

    if is_start == True:
        if new_sent[new_locate:new_locate+1] == " " and new_sent[:new_locate].replace(" ", "") == new_sent[:new_locate+1].replace(" ", ""):
            new_locate += 1
            new_str1 = new_sent[:new_locate].replace(" ", "")

    assert old_str1 == new_str1
    return new_locate

'''
对source event作处理，处理后可能存在重复，这里对重复事件不作处理
此处没有将Theme2修改称Theme
有一个小小的问题, 有些binding事件可能带不止两个论元
% E11 Positive_regulation:T60 Theme:E9 Cause:T19
'''
def process_strevent(str_event): #注意后面是否有"\n"
    event_info = str_event.strip().split(' ')
    newevent_idx = event_info[1]
    newevent = ''
    if len(event_info) == 4: #事件只有一个论元
        newevent = str_event

    elif len(event_info) == 5:
        if event_info[4].startswith("Theme") or event_info[4].startswith("Cause"):
            newevent = str_event

        else:
            newevent = event_info[0] + " " + event_info[1] + " " + event_info[2] + " " + event_info[3]
            newevent1 = " ".join(event_info[0:4])
            assert newevent == newevent1

    elif len(event_info) >= 6: # 考虑到Binding事件可能不止两个论元,所以
        newevent = event_info[0] + " " + event_info[1] + " " + event_info[2]
        for argu_info in event_info[3:]:
            argu_type = argu_info.split(":")[0]
            argu_idx = argu_info.split(":")[1]
            if argu_type.startswith("Cause"):
                newevent += " " + argu_type+":"+argu_idx
            elif argu_type.startswith("Theme"):
                newevent += " Theme:"+argu_idx
    return newevent, newevent_idx

'''
生成序列：Protein和trigger类型以及NoTrigger
prot_trig
'''
def gen_seq(new_sent, prot_dict, trig_dict):
    sent_list = new_sent.split()
    sent_len = len(sent_list)
    seq_result = ["NoTP"]*sent_len # "NoTP"

    prott_len = 0
    trigg_len = 0
    for protein in prot_dict.values():
        prot_start = protein.prot_start #单词的开始位置
        prot_end = protein.prot_end

        prott_len += prot_end-prot_start+1

        for i in range(prot_start, prot_end+1):
            seq_result[i] = "Prot"

        assert sent_len == len(seq_result)

    for trigger in trig_dict.values():
        trig_start = trigger.trig_start
        trig_end = trigger.trig_end
        trig_type = trigger.trig_type
        trigg_len += (trig_end-trig_start)+1
        #assert trig_name == t_name
        for j in range(trig_start, trig_end+1):
            if seq_result[j] == "Prot":
                type = seq_result[j]
                seq_result[j] = type+"_Trig"  #应该是Prot_Trig
            else:
                seq_result[j] = trig_type

        assert sent_len == len(sent_list)

    return seq_result
