#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################################################################################
####### This source code is based on [DevinZ1993](https://github.com/DevinZ1993/Chinese-Poetry-Generation)'s implementation.      #######
#########################################################################################################################################


from generateModel import GenerateModel
from transformerModel import GenerateTransformerModel
from plan import Planner

if __name__ == '__main__':
    planner = Planner()
    # generator = GenerateModel()
    generator = GenerateTransformerModel(False)


    while True:
        hints = input("Provide a title >> ")
        keywords = planner.plan(hints)
        print("Keywords: " + ' '.join(keywords))
        poem = generator.generate(keywords)
        print("Poem generated:")
        for sentence in poem:
            print(sentence)


