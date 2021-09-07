"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NoTP': 2, 'Protein': 3, 'Prot_Trig': 4, 'Gene_expression': 5, 'Transcription': 6, 'Protein_catabolism': 7, 'Localization': 8, 'Binding': 9, 'Protein_modification':10, 'Phosphorylation': 11, 'Ubiquitination':12, 'Acetylation':13, 'Deacetylation':14, 'Regulation': 15, 'Positive_regulation': 16, 'Negative_regulation': 17}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '(': 24, ')': 25, 'VBP': 26, 'MD': 27, 'NNPS': 28, 'WP': 29, 'WDT': 30, 'WRB': 31, 'RP': 32, 'JJR': 33, 'JJS': 34, 'HYPH': 35, 'FW': 36, 'RBR': 37, 'SYM': 38, 'EX': 39, 'RBS': 40, 'WP$': 41, 'PDT': 42, 'LS': 43, '#': 44}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'csubjpass': 40}

# CHAR_TO_ID = {} #在event_parsing.py中进行赋值
#
TRIGGER_TO_ID = {}

PROT_TO_ID = {'Other': 0, 'B-Protein': 1, 'I-Protein': 2}

JPROT_TO_ID = {'Other': 0, 'B-Protein': 1, 'I-Protein': 2, 'START': 3, 'STOP': 4}

# 识别trigger时的类别
#TRIGGER_TO_ID = {'Other': 0, 'Gene_expression': 1, 'Transcription': 2, 'Phosphorylation': 3, 'Protein_catabolism': 4, 'Localization': 5, 'Binding': 6, 'Regulation': 7, 'Positive_regulation': 8, 'Negative_regulation': 9, 'START': 10, 'STOP': 11}

RELA_TO_ID = {'Other': 0, 'Theme': 1, 'Cause': 2} # no_relation不如换成other

INFINITY_NUMBER = 1e12 # 一乘以十的十二次方

SIMP_TYPE = ['Gene_expression', 'Transcription', 'Protein_catabolism', 'Localization']

BIND_TYPE = ['Binding']

PMOD_TYPE = ['Protein_modification', 'Ubiquitination',  'Acetylation', 'Phosphorylation', 'Deacetylation'] # Phosphorylation最后两个有Cause论元, 且论元是protein

REGU_TYPE = ['Regulation', 'Positive_regulation', 'Negative_regulation']

