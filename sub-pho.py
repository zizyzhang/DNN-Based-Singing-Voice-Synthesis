import re
from pypinyin import pinyin, lazy_pinyin, Style
import copy
import numpy as np


def pinyin_normalize(input):
    result = re.sub(r'([yjqx])u', r'\1v', input)
    result = re.sub(r'iu', 'iou', result)
    result = re.sub(r'ui', 'uei', result)
    result = re.sub(r'un', 'uen', result)
    result = re.sub(r'yi', r'i', result)
    result = re.sub(r'wu', r'u', result)
    result = re.sub(r'^y', 'i', result)
    result = re.sub(r'^w', 'u', result)
    return result


def pinyin_search(input):
    phos = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
            "i", "u", "v", "a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong", "ia", "ua", "uo",
            "ie", "ve", "uai", "uei", "iao", "iou", "ian", "uan", "van", "in", "uen", "vn", "iang", "uang", "ing",
            "ueng", "iong", "er"]
    for i in phos:
        if re.match(r'' + input, i):
            return True
    return False


def pinyin_split(input_pinyin):
    input_pinyin = pinyin_normalize(input_pinyin)
    result = []
    i = 0
    j = 1
    while True:

        if pinyin_search(input_pinyin[i:j]) > 0 and j <= len(input_pinyin):
            j += 1
        else:
            sub = input_pinyin[i:j - 1]
            result.append(sub)
            i = j - 1
            j = i + 1

        if i >= len(input_pinyin):
            break
    return result


def sentence_split(sentence):
    pinyins = pinyin(sentence, style=Style.NORMAL)
    splited = [pinyin_split(p[0]) for p in pinyins]
    return splited


def change_lab(lab, phos):
    idx = 0
    result = copy.copy(lab)
    for index, frame in enumerate(lab):
        if frame['cur_pho'] != 'pau':
            result[index]['cur_pho'] = phos[idx]
            idx += 1
    return lab



res = sentence_split("今天很开心")

print(np.ndarray.flatten(np.array(res)))

