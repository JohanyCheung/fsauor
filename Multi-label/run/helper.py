# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd
from string import punctuation
from collections import Counter, OrderedDict


def logging(file):
    def write_log(s):
        print(s, end='')
        with open(file, 'a') as f:
            f.write(s)
    return write_log


def logging_csv(file):
    def write_csv(s):
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(s)
    return write_csv


def get_stopwords():
    with open("dict/stopwords.txt") as f:
        return [line.strip() for line in f.readlines()]


def get_chars(text):
    res = [c for c in text if c not in filter_characters]
    return res


def process_csv_dict(srcfile=None, desfile=None, func=None, fields=None):
    """按行以字典形式读取 csv 文件进行自定义处理，将结果保存到目标文件。
    直到遇到空行或者到达文件末尾为止。
    """
    n = 0
    with open(srcfile, 'r', encoding='utf-8') as src, open(desfile, 'w', encoding='utf-8') as des:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(des, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            writer.writerow(func(row, fields))
            n += 1
            if n % 100 == 0:
                print(n)
    print("process_csv_dict complete!")


def process_pred(srcfile=None, desfile=None, func=None, fields=None):
    n = 0
    with open(desfile, 'w', encoding='UTF-8') as des:
        if fields:
            des.write(','.join(fields) + '\n') # header
        with open(srcfile, 'r', encoding='UTF-8') as src:
            for line in iter(src.readline, ''):
                des.write(','.join(func(line)) + '\n')
                n += 1
                if n % 100 == 0:
                    print(n)
    print("process_pred complete!")


def generate(pfile=None, desfile=None, k=None, default=None):
    """用粒度 k 的预测结果更新测试文件中相应的列。
    """
    assert pfile is not None, "pfile can not be None"
    assert desfile is not None, "desfile can not be None"
    assert k is not None, "k can not be None"

    test = pd.read_csv(desfile)
    ddata = test.copy()
    pdata = pd.read_csv(pfile)
    ddata[k] = pdata[k].fillna(default).astype(int)
    with open(desfile, mode='w', newline='\n', encoding='UTF-8') as f:
        ddata.to_csv(f, index=False)


def f1(pred, label, vocab):
    """计算单一粒度的 f1
    usage:
        vocab = [1, 0, -1, -2]
        pred = [1, 0, 1, -2, -1, 0, -1, 1, -2]
        label = [1, -2, -1, -2, -1, 0, 1, 1, 1]
        print(f1_score(pred, label, vocab))
    """
    assert vocab is not None, "vocab can not be None!"
    assert len(pred) == len(label), "len(pred) must be equal to len(label)!"
    tmp = []
    for v in vocab:
        tp, fp, tn, fn = 0, 0, 0, 0
        precision, recall = 0, 0
        for p, l in zip(pred, label):
            if p == v:
                if p == l:
                    tp += 1
                else:
                    fp += 1
            else:
                if l == v:
                    fn += 1
                else:
                    tn += 1
        # print(v, tp, fp, tn, fn)
        precision = tp / (tp + fp + 0.00001)
        recall = tp / (tp + fn + 0.00001)
        f1_v = 2 * precision * recall / (precision + recall + 0.00001)
        tmp.append(f1_v)
    # print(tmp)
    return np.mean(tmp)


def f1_mltc(pfile=None, lfile=None, fields=None, vocab=None):
    """Computing f1_score of Multi Label Task Classification
    """
    assert pfile is not None, "pfile can not be None"
    assert lfile is not None, "lfile can not be None"
    assert fields is not None, "fields can not be None"
    assert vocab is not None, "vocab can not be None"

    pdata = pd.read_csv(pfile)
    ldata = pd.read_csv(lfile)
    tmp = []
    for k in fields:
        f1_k = f1(pdata[k], ldata[k], vocab)
        tmp.append(f1_k)
        print(k, f1_k)
    return np.mean(tmp)


def count_label(filepath=None, fields=None):
    """统计指定粒度的标签。
    """
    assert filepath is not None, "filepath can not be None"
    data = pd.read_csv(filepath)
    mat = '{:50}'
    res = {}
    for k in fields:
        tmp = Counter(data[k].astype(int))
        counter = OrderedDict(sorted(tmp.items(), key=lambda t: t[0]))
        print(mat.format(k), counter, '\n')
        res[k] = counter
    return res


def get_maxlen(srcfile=None):
    """获取训练集单个序列的最大长度
    """
    maxlen = 0
    assert srcfile is not None, "srcfile can not be None"
    with open(srcfile, 'r', encoding='UTF-8') as src:
        for line in iter(src.readline, ''):
            maxlen = max(maxlen, len(line.split()))
    return maxlen


def get_avglen(srcfile=None):
    """获取训练集单个序列的平均长度
    """
    assert srcfile is not None, "srcfile can not be None"
    with open(srcfile, 'r', encoding='UTF-8') as src:
        tmp = [len(line.split()) for line in iter(src.readline, '')]
        counter = Counter(tmp)
        res = OrderedDict(sorted(counter.items(), key=lambda t: t[0]))
        return np.mean(tmp), res
    return None