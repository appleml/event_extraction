class biofile(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.sent_set = list()

    def add_sent(self, sentence):
        self.sent_set.append(sentence)

## 读数据时处理后的数据
class sentence(object):
    def __init__(self, old_sent, new_sent, prot_dict, genia_info, parse_info):
        self.old_sent = old_sent
        self.new_sent = new_sent
        self.prot_dict = prot_dict

        self.prev_sentence = ""
        self.next_sentence = ""

        # genia and parse information
        self.genia_info = genia_info
        self.parse_info = parse_info

        self.jprot_dict = None
        self.trig_dict = None
        self.event_dict = None
        self.event_set = None

# 句子的 genia 信息
class sent_genia():
    def __init__(self, words, pos_set, prot_set):
        self.words = words
        self.pos_set = pos_set
        self.prot_set = prot_set
        self.jprot_set = None
        self.trig_types = None

# 句子的 parse 信息
class sent_parse():
    def __init__(self, words, pos_set, head_set, deptype_set):
        self.words = words
        self.pos_set = pos_set
        self.head_set = head_set
        self.deptype_set = deptype_set

## 这个类是经过处理后用来保存信息的
class sent_info():
    def __init__(self, file_name, tokenIds, posIds, protrigIds, deprelIds, headIds, entity_pair, gold_events):
        self.file_name = file_name
        self.tokenIds = tokenIds
        self.posIds = posIds
        self.protrigIds = protrigIds
        self.deprelIds = deprelIds
        self.headIds = headIds
        self.entity_pair = entity_pair
        self.gold_events = gold_events

# 存储句子中存在的事件
class sent_event():
    def __init__(self, file_name):
        self.file_name = file_name
        self.simp_event_set = list()
        self.bind_event_set = list()
        self.pmod_event_set = list()
        self.regu_event_set = list()

    def add_event(self, event_set):
        for event in event_set:
            event_type = event.event_type
            assert event_type != ''
            if event_type in ['Gene_expression', 'Transcription', 'Protein_catabolism', 'Localization']:
                self.simp_event_set.append(event)
            elif event_type in ['Binding']:
                self.bind_event_set.append(event)
            elif event_type in ['Protein_modification', 'Phosphorylation', 'Ubiquitination',  'Acetylation', 'Deacetylation']:
                self.pmod_event_set.append(event)
            elif event_type in ['Regulation', 'Positive_regulation', 'Negative_regulation']:
                self.regu_event_set.append(event)
            else:
                print("应该没有这种情况")

    def add_simp(self, simp_events):
        self.simp_event_set = self.simp_event_set+simp_events

    def add_bind(self, bind_events):
        self.bind_event_set.extend(bind_events)

    def add_pmod(self, pmod_events):
        self.pmod_event_set.append(pmod_events)

    def add_regu(self, regu_events):
        self.regu_event_set.extend(regu_events)