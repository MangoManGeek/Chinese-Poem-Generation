#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from paths import raw_dir, data_sxhy_path, check_uptodate
from singleton import Singleton
from utils import split_sentences, generate_sxhy_word_set
import jieba
import os

############################################################################################################################################################
####### This part of data processing source code is based on [DevinZ1993](https://github.com/DevinZ1993/Chinese-Poetry-Generation)'s implementation. #######
############################################################################################################################################################



class Segmenter(Singleton):

    def __init__(self):
        if not check_uptodate(data_sxhy_path):
            # check if previously processed shixuehanying already, if not process and write to data/ dir
            generate_sxhy_word_set()
        with open(data_sxhy_path, 'r') as fin:
            self.sxhy_dict = set(fin.read().split())

    def segment(self, sentence):
        # TODO: try CRF-based segmentation.
        """
        toks = []
        idx = 0
        while idx + 4 <= len(sentence):
            # Cut 2 chars each time.
            if sentence[idx : idx + 2] in self.sxhy_dict:
                # if this word previous exists in sxhy_dict, use directly
                toks.append(sentence[idx : idx + 2])
            else:
                # if this word is not in sxhy_dict, use jieba to tokenize the words
                for tok in jieba.lcut(sentence[idx : idx + 2]):
                    toks.append(tok)
            idx += 2
        # Cut last 3 chars.
        if idx < len(sentence):
            if sentence[idx : ] in self.sxhy_dict:
                toks.append(sentence[idx : ])
            else:
                for tok in jieba.lcut(sentence[idx : ]):
                    toks.append(tok)
        return toks
        """
        return jieba.lcut(sentence)


# For testing purpose.
if __name__ == '__main__':
    segmenter = Segmenter()
    with open(os.path.join(raw_dir, 'qts_tab.txt'), 'r') as fin:
        for line in fin.readlines()[1 : 6]:
            for sentence in split_sentences(line.strip().split()[3]):
                print(' '.join(segmenter.segment(sentence)))

