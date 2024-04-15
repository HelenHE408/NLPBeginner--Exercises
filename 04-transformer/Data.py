## exmaple of usage: can get coupus(tensor), vocab_size, dicts of encoder/ decoder
# corpus = Data.Data('fra_clean.txt',2000)
# corpus_en, vocab_size_en, word2idx_en, idx2word_en = corpus.encoder()

import numpy as np
import string
import pandas as pd
import torch
# import re
# import nltk
# import collections

# global var
chi_string = '？！“”。，《》[]〖〗'

# functions

def dePunctuation(line):
    line = line.translate(str.maketrans('', '', string.punctuation))
    line = line.translate(str.maketrans('', '', chi_string))
    return line

def clean(text):
    text_new = dePunctuation(text)
    return text_new

def get_vocab(corpus):
    vocab = [word for line in corpus for word in line]
    vocab = set(vocab)
    # 需要保证position的维度能被2整除
    if len(set(vocab))%2 == 0:
        vocab_size = len(vocab)
    else:
        vocab.add('#')
        vocab_size = len(vocab)
        
    return vocab, vocab_size

def make_dic(vocab):
    word2idx = {word:i for i, word in enumerate(vocab)}
    idx2word = {i:word for i, word in enumerate(vocab)}
    return word2idx, idx2word

def make_tensor(corpus, vocab_size, word2idx):
    # get size
    batch_size = len(corpus)
    seq_length = max(len(line) for line in corpus)
    
    # make tensor
    df = pd.DataFrame(index=range(seq_length), columns=range(batch_size))
    df = pd.DataFrame(corpus)
    df = df.map(lambda x: word2idx.get(x) if x in word2idx else vocab_size).T

    array = df.to_numpy()
    if vocab_size % 2 == 0:
        vocab_size += 2
        new_array = np.zeros((seq_length, batch_size, vocab_size))
    else:
        vocab_size += 1
        new_array = np.zeros((seq_length, batch_size, vocab_size))
        
    
    for a in range(len(array)):
        new_array[a] = np.eye(vocab_size)[array[a]]

    corpus_tensor = torch.from_numpy(new_array).to(torch.float32).transpose(0,1)
    # # 删除最后一个维度的最后一列，因为最后一列是-1填充的nan值；更正：本来就是表示空值，不用删
    # new_column = torch.zeros(batch_size, seq_length, vocab_size+1)
    # if corpus_tensor.shape[-1]%2 == 0:
    #     pass
    # else:
    #     corpus_tensor = torch.cat((corpus_tensor, new_column), dim=-1)

    # if corpus_tensor.shape[-1]%2 == 0:
    #     corpus_tensor[:, :, -1] = 0
    # else:
    #     corpus_tensor = corpus_tensor[:, :, :-1]

    return corpus_tensor

def make_target(corpus, vocab_size, word2idx):
    # get size
    batch_size = len(corpus)
    seq_length = max(len(line) for line in corpus)
    
    # make tensor
    df = pd.DataFrame(index=range(seq_length), columns=range(batch_size))
    df = pd.DataFrame(corpus)
    df = df.map(lambda x: word2idx.get(x) if x in word2idx else vocab_size).T

    array = df.to_numpy()
    corpus_tensor = torch.from_numpy(array).to(torch.float32).transpose(0,1)

    return corpus_tensor

# class

class Data:

    def __init__(self, file_name, text_length):
        self.file_name = file_name
        self.text_length = text_length
        self.corpus_en, self.corpus_de = self.corpus()
    
    def text_clean(self):

        file_name = self.file_name
        text_length = self.text_length

        file = open(file_name, "r")
        ori_text = []
        for line in file:
            ori_text.append(line.strip())
        file.close()

        # raw_text = [ori_text[i] for i in range(2000)]
        raw_text = ori_text[:text_length]
        lines = '\n'.join(raw_text)

        text_clean = clean(lines)
        text_clean = text_clean.split('\n')

        return text_clean
    
    def corpus(self):
        text_clean = self.text_clean()
        corpus = [line.split('\t') for line in text_clean]
        corpus = [word.split() for line in corpus for word in line]

        corpus_en = corpus[0::2]
        corpus_de = corpus[1::2]

        return corpus_en, corpus_de
        
    def encoder(self):
        corpus_en = self.corpus_en
        vocab, vocab_size = get_vocab(corpus_en)
        word2idx, idx2word = make_dic(vocab)
        corpus_tensor = make_tensor(corpus_en, vocab_size, word2idx)
        return corpus_tensor, vocab_size, word2idx, idx2word
    
    def decoder(self):
        corpus_de = self.corpus_de
        vocab, vocab_size = get_vocab(corpus_de)
        word2idx, idx2word = make_dic(vocab)
        corpus_tensor = make_tensor(corpus_de, vocab_size, word2idx)
        return corpus_tensor, vocab_size, word2idx, idx2word
    
    def target(self):
        corpus_de = self.corpus_de
        vocab, vocab_size = get_vocab(corpus_de)
        word2idx, _ = make_dic(vocab)
        target = make_target(corpus_de, vocab_size, word2idx)
        return target
    
# if __name__ == "__main__":
#     corpus = Data.Data('fra_clean.txt',2000)
#     corpus_en, vocab_size_en, word2idx_en, idx2word_en = corpus.encoder()