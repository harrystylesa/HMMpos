import re
import os
import pandas as pd
import sys
import numpy as np


def main(filename):
    path = 'D:/WorkSpace/python/jupyter/nlp/homework2/HMM_pos/data/'
    dfa = pd.read_csv(path + 'train_pos.txt_alpha.csv', encoding='utf-8')
    dfb = pd.read_csv(path + 'train_pos.txt_b.csv', encoding='utf-8')
    dfpi = pd.Series.from_csv(path + 'train_pos.txt_pi.csv', encoding='utf-8', header=None)
    wordlist = list(dfb)
    labelist = list(dfa)
    dfa_normal = (dfa + 1).apply(get_prob, axis=1).fillna(0)
    dfb_normal = (dfb.fillna(0) + 1).apply(get_prob, axis=1)
    dfpi_normal = (dfpi + 1) / (dfpi + 1).sum()
    pi = np.array(np.log(dfpi_normal))
    alpha = np.array(np.log(dfa_normal))
    b = np.array(np.log(dfb_normal))
    rpunc = re.compile(r'[。！？，；：、【】\[\]“”（）]')
    result = []
    with open(path + filename, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            start = 0
            pos = []
            if re.findall(rpunc, line):
                for seg_sign in re.finditer(rpunc, line):
                    segment = line[start:seg_sign.start()]
                    seglist = segment.strip().split(' ')
                    if not seglist == ['']:
                        ob_seq = [wordlist.index(i) if i in wordlist else -1 for i in seglist]
                        pos.extend(get_pos(ob_seq, alpha, b, pi, labelist))
                    start = seg_sign.end()
                    pos.append('w')
            else:
                ob_seq = [wordlist.index(i) if i in wordlist else -1 for i in line.strip().split(' ')]
                pos.extend(get_pos(ob_seq, alpha, b, pi, labelist))
            result.append(get_out_put(line.strip().split(' '), pos))
    outputfile = path + filename + '_HMMpos.txt'
    if os.path.exists(outputfile):
        os.remove(outputfile)
    with open(outputfile, mode='a', encoding='utf-8') as out:
        out.writelines(result)
    print('created HMMpos file to: ' + outputfile)
    print(result)
    print(len(result))


def get_out_put(line, pos):
    result = []
    for i, j in zip(line, pos):
        result.append(i + '/' + j)
    return ' '.join(result) + '\n'


def get_pos(ob_seq, alpha, b, pi, labels):
    N = alpha.shape[0]
    T = len(ob_seq)
    delta = np.zeros([N, T], dtype=float)
    memo = [None for j in range(T)]
    delta[:, 0] += (pi + b[:, ob_seq][:, 0])
    for i in range(1, T):
        max_ = np.max(delta[:, i - 1] + alpha)
        _, y = np.where(delta[:, i - 1] + alpha == max_)
        memo[i] = [i for i in y]
        delta[:, i] = max_ + b[:, ob_seq][:, i]
    y = np.where(delta[:, T - 1] == np.max(delta[:, T - 1]))
    memo.append(list(y[0]))
    return [labels[i[0]] if i else 'end' for i in memo][1:]


def get_prob(df):
    return df.div(df.sum())


if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)
