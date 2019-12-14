#! /usr/bin/env python3
# -*- coding:utf-8 -*-

############################################################################################################################################################
####### This part of data processing source code is based on [DevinZ1993](https://github.com/DevinZ1993/Chinese-Poetry-Generation)'s implementation. #######
############################################################################################################################################################


import os
import jieba
from paths import raw_dir, data_sxhy_path, check_uptodate



#global Constants 
NUM_OF_SENTENCES = 4
CHAR_VEC_DIM = 512


def is_cn_char(ch):
    """ Test if a char is a Chinese character. """
    return ch >= u'\u4e00' and ch <= u'\u9fa5'

def is_cn_sentence(sentence):
    """ Test if a sentence is made of Chinese characters. """
    for ch in sentence:
        if not is_cn_char(ch):
            return False
    return True

def split_sentences(text):
    """ Split a piece of text into a list of sentences. """
    sentences = []
    i = 0
    for j in range(len(text) + 1):
        if j == len(text) or \
                text[j] in [u'，', u'。', u'！', u'？', u'、', u'\n']:
            if i < j:
                sentence = u''.join(filter(is_cn_char, text[i:j]))
                sentences.append(sentence)
            i = j + 1
    return sentences



def generate_sxhy_word_set():
#def _gen_sxhy_dict():
    print("Parsing shixuehanying dictionary ...")
    words = set()
    raw_sxhy_path=os.path.join(raw_dir, 'shixuehanying.txt')
    with open(raw_sxhy_path, 'r') as f_raw_sxhy:
        for line in f_raw_sxhy.readlines():
            #skip section titles : ex--> "<begin>   1   人事类"
            if line[0] == '<':
                continue
            # for each word in a list of words from a single line, ignoring starting number and space
            for phrase in line.strip().split()[1:]:
                if not is_cn_sentence(phrase):
                    #ignore all non-Chinese characters
                    continue
                idx = 0
                while idx + 4 <= len(phrase):
                    # Cut 2 chars each time.
                    words.add(phrase[idx : idx + 2])
                    idx += 2
                # Use jieba to cut the last 3 chars.
                if idx < len(phrase):
                    for word in jieba.lcut(phrase[idx:]):
                        words.add(word)
    with open(data_sxhy_path, 'w') as fout:
        fout.write(' '.join(words))





