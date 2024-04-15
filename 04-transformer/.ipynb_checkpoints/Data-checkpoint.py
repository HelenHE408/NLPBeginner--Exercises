import numpy as np
import string
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
        return vocab, vocab_size, word2idx, idx2word
    
    def decoder(self):
        corpus_de = self.corpus_de
        vocab, vocab_size = get_vocab(corpus_de)
        word2idx, idx2word = make_dic(vocab)
        return vocab, vocab_size, word2idx, idx2word
    
# module
# if __name__ == '__main__':
#     data = Data('fra_clean.txt', 2000)
#     vocab_en, vocab_size_en, word2idx_en, idx2word_en = data.encoder()
#     print(vocab_size_en)