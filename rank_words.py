#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from paths import raw_dir, wordrank_path, check_uptodate
from poems import Poems
from segment import Segmenter
from singleton import Singleton
import json
import os
import sys


from tencent_embedding_utils.word2vec_utils import get_tencent_embedding_keyedVectors


_stopwords_path = os.path.join(raw_dir, 'stopwords.txt')

_tencent_embedding_path='../data/truncated_Tencent_AILab_ChineseEmbedding.txt'

NUM_Of_ITERATIONS=100
_damp = 0.85
#_damp = 1


"""
# what's the purpose of stop words?
"""
def _get_stopwords():
    stopwords = set()
    with open(_stopwords_path, 'r') as fin:
        for line in fin.readlines():
            stopwords.add(line.strip())
    return stopwords


# TODO: try other keyword-extraction algorithms. This doesn't work well.

class RankedWords(Singleton):

    def __init__(self):
        self.stopwords = _get_stopwords()
        if not check_uptodate(wordrank_path):
            self._do_text_rank()
        with open(wordrank_path, 'r') as fin:
            self.word_scores = json.load(fin)
        self.word2rank = dict((word_score[0], rank) 
                for rank, word_score in enumerate(self.word_scores))

    def _do_text_rank(self):
        print("Do text ranking ...")
        adjlists = self._get_adjlists()
        #adjlists = self._build_adjlists_from_tencent_embeddings()
        print("[TextRank] Total words: %d" % len(adjlists))

        # Value initialization.
        scores = dict()
        for word in adjlists:
            #score[0] is previous score, score[1] is new score
            scores[word] = [1.0, 1.0]

        # Synchronous value iterations.
        itr = 0
        #### train text rank here #####
        while True:
            sys.stdout.write("[TextRank] Iteration %d ..." % itr)
            sys.stdout.flush()
            for word, adjlist in adjlists.items():
                scores[word][1] = (1.0 - _damp) + _damp * \
                        sum(adjlists[other][word] * scores[other][0] 
                                for other in adjlist)
            

            #eps is the difference between new score and previous score, used to check for convergence
            eps = 0
            for word in scores:
                eps = max(eps, abs(scores[word][0] - scores[word][1]))
                scores[word][0] = scores[word][1]
            print(" eps = %f" % eps)
            # if eps <= 1e-6:
            #     break
            #if itr == 200:  # train for only 200 iteration ###########################
            if itr == NUM_Of_ITERATIONS:
                break
            itr += 1

        # Dictionary-based comparison with TextRank score as a tie-breaker.
        segmenter = Segmenter()
        def cmp_key(x):
            word, score = x
            return (0 if word in segmenter.sxhy_dict else 1, -score)
        words = sorted([(word, score[0]) for word, score in scores.items()], 
                key = cmp_key)

        # Store ranked words and scores.
        with open(wordrank_path, 'w') as fout:
            json.dump(words, fout)

    def _get_adjlists(self):
        print("[TextRank] Generating word graph ...")
        segmenter = Segmenter()
        poems = Poems()
        adjlists = dict()   # 2D dict, dict[word1][word2]=prob(going from word1 to word2)
        # Count number of co-occurrence.

        """
        ######################## count relationship per sentence ###################
        for poem in poems:
            for sentence in poem:
                words = []
                for word in segmenter.segment(sentence):
                    # for each word selected from the sentence
                    if word not in self.stopwords:
                        #keep only non-stopwords words
                        words.append(word)
                for word in words:
                    if word not in adjlists:
                        #initialize all words to a new dict()
                        adjlists[word] = dict()
                for i in range(len(words)):
                    for j in range(i + 1, len(words)):
                        #### if two words present in the same sentence, their score +=1 #####
                        if words[j] not in adjlists[words[i]]:
                            adjlists[words[i]][words[j]] = 1.0
                        else:
                            adjlists[words[i]][words[j]] += 1.0
                        if words[i] not in adjlists[words[j]]:
                            adjlists[words[j]][words[i]] = 1.0
                        else:
                            adjlists[words[j]][words[i]] += 1.0

        ######################## end count relationship per sentence ###################
        """


        ######################## count relationship per poem ###################
        for poem in poems:
            for sentence in poem:
                words = []
                for word in segmenter.segment(sentence):
                    # for each word selected from the sentence
                    if word not in self.stopwords:
                        #keep only non-stopwords words
                        words.append(word)
            for word in words:
                if word not in adjlists:
                    #initialize all words to a new dict()
                    adjlists[word] = dict()
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    #### if two words present in the same sentence, their score +=1 #####
                    if words[j] not in adjlists[words[i]]:
                        adjlists[words[i]][words[j]] = 1.0
                    else:
                        adjlists[words[i]][words[j]] += 1.0
                    if words[i] not in adjlists[words[j]]:
                        adjlists[words[j]][words[i]] = 1.0
                    else:
                        adjlists[words[j]][words[i]] += 1.0

        ######################## end count relationship per poem ###################

        # Normalize weights.
        for a in adjlists:
            sum_w = sum(w for _, w in adjlists[a].items())
            for b in adjlists[a]:
                adjlists[a][b] /= sum_w
        return adjlists

    def _build_adjlists_from_tencent_embeddings(self):
        print("[TextRank] Generating word graph ...")
        segmenter = Segmenter()
        poems = Poems()
        adjlists = dict()   # 2D dict, dict[word1][word2]=prob(going from word1 to word2)
        wv=get_tencent_embedding_keyedVectors(_tencent_embedding_path)



        # Count number of co-occurrence.


        ######################## get a 2D cos sim matrix for all words ###################
        words = set()
        for poem in poems:
            for sentence in poem:
                for word in segmenter.segment(sentence):
                    # for each word selected from the sentence
                    if word not in self.stopwords:
                        #keep only non-stopwords words
                        words.add(word)
        for word in words:
            if word not in adjlists:
                #initialize all words to a new dict()
                adjlists[word] = dict()

        for word in words:
            for other in words:

                if word==other:
                    continue

                if other in adjlists[word] or word in adjlists[other]:
                    continue

                sim=wv.similarity(word,other)
                adjlists[word][other]=sim
                adjlists[other][word]=sim


        # Normalize weights.
        for a in adjlists:
            sum_w = sum(w for _, w in adjlists[a].items())
            for b in adjlists[a]:
                adjlists[a][b] /= sum_w
        return adjlists

    def __getitem__(self, index):
        if index < 0 or index >= len(self.word_scores):
            return None
        return self.word_scores[index][0]

    def __len__(self):
        return len(self.word_scores)

    def __iter__(self):
        return map(lambda x: x[0], self.word_scores)

    def __contains__(self, word):
        return word in self.word2rank

    def get_rank(self, word):
        if word not in self.word2rank:
            return len(self.word2rank)
        return self.word2rank[word]


# For testing purpose.
if __name__ == '__main__':
    ranked_words = RankedWords()
    for i in range(1000):
        print(ranked_words[i])

