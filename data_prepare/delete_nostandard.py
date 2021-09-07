import copy
# 主要是删除跨句子的无效事件 和 multiword_trigger引发的事件
# 重复事件已经删除过了
def delete_invalid_event_one(all_event, prot_idx, event_idx, multiword_trigidx):
    assert len(all_event) == len(event_idx)
    sent_events = list()
    prev_num = len(event_idx)
    epoch = 0
    while epoch == 0 or prev_num != len(event_idx):
        prev_num = len(event_idx)
        epoch += 1
        sent_events.clear()
        for sevent in all_event:
            sevent_info = sevent.split()
            sevent_idx = sevent_info[1]
            if len(sevent_info) == 4:
                first_argu_idx = sevent_info[3].split(":")[1]
                if first_argu_idx not in prot_idx+event_idx: # 说明该事件是跨句子事件
                    event_idx.remove(sevent_idx) # 将event_idx从event_idx中去掉
                else:
                    sent_events.append(sevent)

            elif len(sevent_info) == 5:
                first_argu_idx = sevent_info[3].split(":")[1]
                second_argu_idx = sevent_info[4].split(":")[1]
                if first_argu_idx not in prot_idx+event_idx or second_argu_idx not in prot_idx+event_idx:
                    event_idx.remove(sevent_idx)
                else:
                    sent_events.append(sevent)
        all_event = copy.deepcopy(sent_events)

    assert len(sent_events) == len(event_idx)

    ## 判断sent_events里有多少个multi-word trigger引发的事件
    multi_event_num = 0
    for evet in sent_events:
        event_trigidx = evet.split()[2].split(":")[1]
        if event_trigidx in multiword_trigidx:
            multi_event_num += 1

    return sent_events, multi_event_num
'''
(1) 删除掉跨句子的事件: 事件中的蛋白质论元没有出现在prot_idx中
(2) 重复事件已经删除了
(3) 多词构成的trigger引发的事件保留(不要删除)
这个方法还没哟验证
% E31 Protein_modification:T102 Theme:T26
'''
def delete_invalid_event_two(all_event, prot_idx, trig_idx, event_idx):
    assert len(all_event) == len(event_idx)
    sent_events = list()
    prev_num = len(event_idx)
    epoch = 0
    while epoch == 0 or prev_num != len(event_idx):
        prev_num = len(event_idx)
        epoch += 1
        sent_events.clear()
        for sevent in all_event:
            sevent_info = sevent.split()
            sevent_idx = sevent_info[1]
            event_trig_idx = sevent_info[2].split(":")[1]
            if event_trig_idx in trig_idx:
                event_idx.remove(sevent_idx)
            else:
                #------------------------------
                if len(sevent_info) == 4: # % E12 Positive_regulation:T29 Theme:T12
                    first_argu_idx = sevent_info[3].split(":")[1]
                    if first_argu_idx not in prot_idx+event_idx: # 说明该事件是跨句子事件
                        event_idx.remove(sevent_idx) # 将event_idx从event_idx中去掉
                    else:
                        sent_events.append(sevent)

                elif len(sevent_info) == 5:
                    first_argu_idx = sevent_info[3].split(":")[1]
                    second_argu_idx = sevent_info[4].split(":")[1]
                    if first_argu_idx not in prot_idx+event_idx or second_argu_idx not in prot_idx+event_idx:
                        event_idx.remove(sevent_idx)
                    else:
                        sent_events.append(sevent)
                else: #每一个论元都没有跨句子
                    for argu_info in sevent_info[3:]:
                        argu_id = argu_info.split(":")[1]
                        if argu_id not in prot_idx+event_idx:
                            event_idx.remove(sevent_idx)
                            break
                    sent_events.append(sevent) ## 没有break, 所以论元条件都满足

        all_event = copy.deepcopy(sent_events)
    assert len(sent_events) == len(event_idx)
    return sent_events