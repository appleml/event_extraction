"""
Prepare vocabulary and initial word vectors.
## 也应该在此处准备char
"""
import pickle
import argparse
import numpy as np
from collections import Counter, OrderedDict
import os
from utils import vocab, constant, helper


def parse_args():
    parser = argparse.ArgumentParser(description='`Prepare` vocab for relation extraction.')
    parser.add_argument('--data_dir', default='dataset/2013_genia_parse', help='GE4 directory.')
    parser.add_argument('--vocab_dir', default='dataset/vocab', help='Output vocab directory.')
    parser.add_argument('--glove_dir', default='dataset/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')  # 只要运行时该变量有传参就将该变量设为True

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # input files
    train_file = args.data_dir + '/2011_train'
    dev_file = args.data_dir + '/2011_devel'
    #test_file = args.data_dir + '/2011_test'
    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    vocab_file = args.vocab_dir + '/vocab.pkl'  # 词字典，从 train语料中收集
    char_file = args.vocab_dir + "/char.pkl"
    emb_file = args.vocab_dir + '/embedding.npy'

    # load files
    print("loading files...")
    train_tokens, train_char_dict = load_tokens(train_file)  # train_tokens中的词没有去重
    dev_tokens, _ = load_tokens(dev_file)
    #test_tokens, _ = load_tokens(test_file)
    if args.lower:
        train_tokens, dev_tokens = [[t.lower() for t in tokens] for tokens in (train_tokens, dev_tokens)]

    # load glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)  # 收集glove中出现的词
    print("{} words loaded from glove.".format(len(glove_vocab)))

    print("building vocab...")
    v = build_vocab(train_tokens, glove_vocab, args.min_freq)  # train_data中最少出现 min_freq 这么多次的词

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)  # 注意这里的d是有重复的, v 应该是没有重复的
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov * 100.0 / total))

    print("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")  ## 词汇集合vocab以及生成的embedding存储起来备用，为什么一个用pickle存储一个用numpy存储呢
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)  # vocab用pickle存储
    np.save(emb_file, embedding)  # embedding用numpy存储

    ## 将character 存储
    with open(char_file, 'wb') as char_outfile:
        pickle.dump(train_char_dict, char_outfile)
    print("all done.")


def load_tokens(file_path):
    sent_tokens = []
    char_dict = OrderedDict()
    char_dict['<PAD>'] = 0
    char_dict['<UNK>'] = 1
    char_idx = 2

    files = os.listdir(file_path)
    for file_name in files:
        with open(file_path + "/" + file_name, 'r') as infile:
            for line in infile:
                line = line.strip()
                if len(line) > 0:
                    line_info = line.split("\t")
                    word = line_info[0]
                    sent_tokens.append(word)
                    for c in word:
                        if c not in char_dict.keys():
                            char_dict[c] = char_idx
                            char_idx += 1
    # print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return sent_tokens, char_dict

def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)  # 如果没有限制词频，则选用了glove_vocab
    # add special tokens and entity mask tokens
    v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v


# 不care词的重复
def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)  # 统计tokens的重复次数
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total - matched


if __name__ == '__main__':
    main()