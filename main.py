#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from generateModel import GenerateModel
from plan import Planner

if __name__ == '__main__':
    planner = Planner()
    generator = GenerateModel(False)

    while True:
        hints = input("Provide a title >> ")
        keywords = planner.plan(hints)
        print("Keywords: " + ' '.join(keywords))
        poem = generator.generate(keywords)
        print("Poem generated:")
        for sentence in poem:
            print(sentence)

