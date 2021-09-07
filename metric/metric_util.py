from data_prepare.entity import Protein, Event

#比较两个trigger是否相同
def trig_isequal(trig1, trig2):
    if trig1.trig_name == trig2.trig_name and trig1.trig_start == trig2.trig_start and \
                    trig1.trig_end == trig2.trig_end and trig1.trig_type == trig2.trig_type:
        return True
    return False

#比较两个protein是否相同
def prot_isequal(prot1, prot2):
    if prot1.prot_name == prot2.prot_name and prot1.prot_start == prot2.prot_start and \
                    prot1.prot_end == prot2.prot_end:
        return True
    return False

def event_isequal(event1, event2):
    trig1 = event1.event_trig
    fargu1 = event1.first_argu
    sargu1 = event1.second_argu

    trig2 = event2.event_trig
    fargu2 = event2.first_argu
    sargu2 = event2.second_argu

    if sargu1 == None and sargu2 == None: #只有一个论元
        if trig_isequal(trig1, trig2) and argu_isequal(fargu1, fargu2):
            return True

    elif sargu1 != None and sargu2 != None: #两个事件的第二个论元都不为空
        if trig_isequal(trig1, trig2) and argu_isequal(fargu1, fargu2) and argu_isequal(sargu1, sargu2):
            return True
        elif trig_isequal(trig1, trig2) and argu_isequal(fargu1, sargu2) and argu_isequal(sargu1, fargu2):
            return True

    return False

# 事件论元的比较，第一个参数类型是event, 迭戈论元类型是example #　要明确的是论元其实是example，可能是有一个argu，也可能有两个argu
#不包含论元的类型比较
def argu_isequal(argu1, argu2):
    if isinstance(argu1, Protein) and isinstance(argu2, Protein):
        return prot_isequal(argu1, argu2)

    elif isinstance(argu1, Event) and isinstance(argu2, Event):
        return event_isequal(argu1, argu2)

    return False