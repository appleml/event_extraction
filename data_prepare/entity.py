class Protein(object):
    def __init__(self, prot_idx, prot_name, prot_oldchar_start, prot_oldchar_end, prot_newchar_start, prot_newchar_end, prot_start, prot_end):
        self.prot_idx = prot_idx
        self.prot_name = prot_name

        self.prot_oldchar_start = prot_oldchar_start  # 老句子中字符的位置
        self.prot_oldchar_end = prot_oldchar_end

        self.prot_newchar_start = prot_newchar_start  # 新句子字符的位置
        self.prot_newchar_end = prot_newchar_end

        self.prot_start = prot_start  # 在新句子中单词的位置
        self.prot_end = prot_end

        self.context_start = 0  #在原文中的位置
        self.context_end = 0
        # S0是前一句, S1是当前句,
        self.prot_mark = ""

class Trigger(object):
    def __init__(self, trig_idx, trig_name, trig_type, trig_oldchar_start, trig_oldchar_end, trig_newchar_start, trig_newchar_end, trig_start, trig_end):
        self.trig_idx = trig_idx
        self.trig_name = trig_name
        self.trig_type = trig_type

        self.trig_oldchar_start = trig_oldchar_start #字符的位置
        self.trig_oldchar_end = trig_oldchar_end

        self.trig_newchar_start = trig_newchar_start
        self.trig_newchar_end = trig_newchar_end

        self.trig_start = trig_start #在新句子中单词的位置
        self.trig_end = trig_end
        # 为了跨句子事件, 将前一句的sent, join protein, trigger合并到当前句S0是前一句, S1是当前句
        self.trig_mark = 0

# 读取string infomation时有用
class Event(object):
    def __init__(self):
        self.event_idx = ''
        self.event_trig_idx = ''
        self.event_type = ''

        self.first_argu_idx = ''
        self.first_argu_type = ''

        self.second_argu_idx = ''
        self.second_argu_type = ''

        self.other_argu_info = dict() # key是argu_idx, value是argu_type
        self.other_argu_entity = dict() # key是argu_idx, value是argu_entity
        #附加信息,便于评估
        self.event_trig = None
        self.first_argu = None
        self.second_argu = None

    ## 读语料的时候转换信息用
    def add_basic_info(self, event_idx, event_trig_idx, event_type, first_argu_idx, first_argu_type):
        self.event_idx = event_idx
        self.event_trig_idx = event_trig_idx
        self.event_type = event_type

        self.first_argu_idx = first_argu_idx
        self.first_argu_type = first_argu_type

    def add_fargu_entity(self, event_trig, first_argu, first_type): #一定要记得添加event_type
        self.event_trig = event_trig
        self.event_type = event_trig.trig_type
        self.event_trig_idx = event_trig.trig_idx

        self.first_argu = first_argu
        if self.first_argu_type == '':
            self.first_argu_type = first_type
        elif self.first_argu_type != '' and first_type == self.first_argu_type:
            pass
        else:
            print('第一个论元类型赋值时出现问题')

    def add_sargu_entity(self, second_argu, second_type):
        self.second_argu = second_argu
        if self.second_argu_type == '':
            self.second_argu_type = second_type
        elif second_type != '' and second_type == self.second_argu_type:
            pass
        else:
            print('第二个论元类型赋值时出现问题')
    # 之前的论元赋值的trigger, 现在将其修改为trigger引发的Event
    # 注意这里修改的是first_argu, 第二个论元在赋值时已经修正过了
    def modify_fargu(self, argu):
        assert isinstance(argu, Event) == True
        self.first_argu = argu