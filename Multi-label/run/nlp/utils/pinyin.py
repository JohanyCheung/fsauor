# -*- coding:utf-8 -*-
import os


class PinYin(object):
    def __init__(self, dict_file="word.data"):
        self.word_dict = {}
        self.dict_file = dict_file

    def load_word(self):
        if not os.path.exists(self.dict_file):
            raise IOError("NotFoundFile")
        with open(self.dict_file) as f:
            for line in f.readlines():
                try:
                    items = line.rstrip().split('    ')
                    self.word_dict[items[0]] = items[1]
                except:
                    items = line.rstrip().split('   ')
                    self.word_dict[items[0]] = items[1]

    def word2pinyin(self, word="", sep=""):
        result = []
        for char in word:
            key = '%X' % ord(char)
            result.append(self.word_dict.get(key, char).split()[0][:-1].lower())
        if sep == "":
            return result
        else:
            return sep.join(result)

    def char2pinyin(self, char=""):
        key = '%X' % ord(char)
        result = self.word_dict.get(key, char).lower()
        return result