from char2vec import Char2Vec
from char_dict import CharDict, end_of_sentence, start_of_sentence
from data_utils import batch_train_data
from paths import save_dir
from pron_dict import PronDict
from random import random
from singleton import Singleton
from utils import CHAR_VEC_DIM, NUM_OF_SENTENCES
import numpy as np
import os
import sys
import tensorflow as tf

_BATCH_SIZE = 64
_NUM_UNITS = 512

_model_path = os.path.join(save_dir, 'model')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator(Singleton):
    def __init__(self):
        super(Generator, self).__init__()

        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self._build_graph()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.saver = tf.train.Saver(tf.global_variables())
        self.trained = False

        #initial weights and other trainable variables

    def _build_graph(self):
        self._build_keyword_encoder()
        self._build_context_encoder()
        self._build_decoder()
        self._build_projector()
        self._build_optimizer()


