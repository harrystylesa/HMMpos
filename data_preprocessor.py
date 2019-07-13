import re
import pandas as pd
import sys
import os
import collections


def main(filename):
    '''

    :return:
    '''
    path = './data/'
    file = open(path+filename, 'r', encoding='utf-8')
    labels = ["Ag", "a", "ad", "an", "b", "c", "Dg", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "Ng", "n", "nr",
              "ns", "nt", "nx", "nz", "o", "p", "q", "r", "s", "Tg", "t", "u", "Vg", "v", "vd", "vn", "w", "x", "y",
              "z", "nrx"]
    punc = ['。', '！', '？', '，', '；', '：', '、', '【', '】', '[', '］', '”', '“', '（', '）']
    seg_punc = ['。', '！', '？', '，', '；', '：', '、']
    pi = collections.OrderedDict()
    for i in labels:
        pi[i] = 0
    words = []
    a = [[0 for i in range(len(labels))] for j in range(len(labels))]
    b = [dict() for i in range(len(labels))]
    # b = [[] for i in range(len(labels))]
    lines = []
    for line in file.readlines():
        lines.append(getHMMParams(line, labels, words, a, b, pi, seg_punc))
    raw_txt = filename + '_raw.txt'
    dfa_csv = filename + '_alpha.csv'
    dfb_csv = filename + '_b.csv'
    dfpi_csv = filename + '_pi.csv'
    files = [raw_txt, dfa_csv, dfb_csv, dfpi_csv]
    for i in files:
        if os.path.exists(i):
            os.remove(i)
    with open(raw_txt, mode='a', encoding='utf-8') as f:
        f.writelines(lines)
    print("created raw_txt to: "+raw_txt)
    dfa = pd.DataFrame(a, columns=labels)
    dfb = pd.DataFrame(b, columns=words)
    dfa.to_csv(dfa_csv, encoding='utf-8', index=False)
    print("created alpha matrix to: " + dfa_csv)
    dfb.to_csv(dfb_csv, encoding='utf-8', index=False)
    print("created b matrix to: " + dfb_csv)
    dfpi = pd.Series(pi)
    dfpi.to_csv(dfpi_csv, encoding='utf-8')
    print("created pi matrix to: " + dfpi_csv)
    return files


def getHMMParams(text, labels, words, a, b, pi, seg_punc):
    '''

    :param text:
    :param labels:
    :param words:
    :param a:
    :param b:
    :param pi:
    :param seg_punc:
    :return:
    '''
    wordlist = text.strip().split(" ")
    lines = ''
    beginning = True
    prelabel = None
    for i in wordlist:
        word = None
        label = None
        if i in ['///w', '//w']:
            word = '/'
            label = 'w'
        elif i == '</w':
            word = '<'
            label = 'w'
        elif len(i) < 3 or i.find('$$_') > -1 or '/' not in i:
            continue
        else:
            iterator = re.finditer('/', i)
            for j in iterator:
                if i[j.start() - 1:j.start()] == '<':
                    continue
                word = i[:j.start()]
                if i[j.end():].find('<') > -1:
                    end = i[j.end():].find('<')
                    label = i[j.end():j.end() + end]
                    break
                else:
                    label = i[j.end():]
                    break
            # if word is None and label is None:
            #     continue
        if label not in labels:
            continue
        lines = lines + word + " "
        if word not in words:
            words.append(word)
        if beginning:
            updatepi(pi, label)
            updateb(b, label, word, labels)
            beginning = False
            prelabel = label
        elif word in seg_punc:
            beginning = True
            prelabel = None
            continue
        else:
            updateb(b, label, word, labels)
            updatea(a, label, prelabel, labels)
            prelabel = label
    return lines.strip() + '\n'


def updateb(b, label, word, labels):
    '''

    :param b:
    :param label:
    :param word:
    :param labels:
    :param words:
    :return:
    '''
    if word in b[labels.index(label)]:
        b[labels.index(label)][word] += 1
    else:
        b[labels.index(label)][word] = 1


def updatea(a, label, prelabel, labels):
    '''

    :param a:
    :param label:
    :param prelabel:
    :param labels:
    :return:
    '''

    a[labels.index(prelabel)][labels.index(label)] += 1


def updatepi(pi, label):
    '''

    :param pi:
    :param label:
    :return:
    '''
    pi[label] = (pi[label] + 1) if pi.get(label) else 1


if __name__ == '__main__':
    filename = sys.argv[1]
    # filename = 'train_pos.txt'
    main(filename)
