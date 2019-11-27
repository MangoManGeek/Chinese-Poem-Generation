#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from generateModel import GenerateModel
from plan import Planner
from paths import save_dir
import tensorflow as tf
import sys

if __name__ == '__main__':
    planner = Planner()
    generator = GenerateModel()
    if not tf.train.get_checkpoint_state(save_dir):
        print("Please train the model first! (./train.py -g)")
        sys.exit(1)

    checkpoint = tf.train.Checkpoint(generator=generator)
    try:
        checkpoint.restore(tf.train.latest_checkpoint(save_dir)).assert_consumed()
        print("Checkpoint is loaded successfully !")
    except AssertionError:
        print("Fail to load checkpoint")

    while True:
        hints = input("Provide a title >> ")
        keywords = planner.plan(hints)
        print("Keywords: " + ' '.join(keywords))
        poem = generator.generate(keywords)
        print("Poem generated:")
        for sentence in poem:
            print(sentence)

