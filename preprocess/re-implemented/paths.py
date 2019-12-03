#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os

#starting directory
#root_dir = os.path.dirname(__file__)
root_dir = os.path.join(os.path.dirname(__file__),"../../../data/")
data_dir = os.path.join(root_dir, 'data')
raw_dir = os.path.join(root_dir, 'raw')
save_dir = os.path.join(root_dir, 'save')

data_sxhy_path = os.path.join(data_dir, 'sxhy_dict.txt')
data_char_dict_path = os.path.join(data_dir, 'char_dict.txt')
data_poems_path = os.path.join(data_dir, 'poem.txt')
data_char2vec_path = os.path.join(data_dir, 'char2vec.npy')
data_wordrank_path = os.path.join(data_dir, 'wordrank.txt')
data_plan_data_path = os.path.join(data_dir, 'plan_data.txt')
data_gen_data_path = os.path.join(data_dir, 'gen_data.txt')

sxhy_path=data_sxhy_path
char_dict_path=data_char_dict_path
poems_path=data_poems_path
char2vec_path=data_char2vec_path
wordrank_path=data_wordrank_path
plan_data_path=data_plan_data_path
gen_data_path=data_gen_data_path

# TODO: configure dependencies in another file.
_dependency_dict = {
    data_poems_path: [data_char_dict_path],
    data_char2vec_path: [data_char_dict_path, data_poems_path],
    data_wordrank_path: [data_sxhy_path, data_poems_path],
    data_gen_data_path: [data_char_dict_path, data_poems_path, data_sxhy_path, data_char2vec_path],
    data_plan_data_path: [data_char_dict_path, data_poems_path, data_sxhy_path, data_char2vec_path],
}


#check if given file is previously processed and saved in data/ dir
def check_uptodate(path):
    """ Return true iff the file exists and up-to-date with dependencies."""
    if not os.path.exists(path):
        # File not found.
        return False
    timestamp = os.path.getmtime(path)
    if path in _dependency_dict:
        for dependency in _dependency_dict[path]:
            if not os.path.exists(dependency) or \
                    os.path.getmtime(dependency) > timestamp:
                # File stale.
                return False
    return True
