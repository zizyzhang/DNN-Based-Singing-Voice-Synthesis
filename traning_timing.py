import re
import numpy as np
import pandas as pd
from keras.models import *
from keras.layers import *
from keras.layers import *
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.models import load_model
import math
import os.path

reg = "(.*) (.*) (.*)\@(.*)\^(.*)\-(.*)\+(.*)\=(.*)\_(.*)\%(.*)\^(.*)\_(.*)\~(.*)\-(.*)\!(.*)\[(.*)\$(.*)\](.*)/A:(.*)\-(.*)\-(.*)\@(.*)\~(.*)/B:(.*)\_(.*)\_(.*)\@(.*)\|(.*)/C:(.*)\+(.*)\+(.*)\@(.*)\&(.*)/D:(.*)\!(.*)\#(.*)\$(.*)\%(.*)\|(.*)\&(.*)\;(.*)\-(.*)/E:(.*)\](.*)\^(.*)\=(.*)\~(.*)\!(.*)\@(.*)\#(.*)\+(.*)\](.*)\$(.*)\|(.*)\[(.*)\&(.*)\](.*)\=(.*)\^(.*)\~(.*)\#(.*)\_(.*)\;(.*)\$(.*)\&(.*)\%(.*)\[(.*)\|(.*)\](.*)\-(.*)\^(.*)\+(.*)\~(.*)\=(.*)\@(.*)\$(.*)\!(.*)\%(.*)\#(.*)\|(.*)\|(.*)\-(.*)\&(.*)\&(.*)\+(.*)\[(.*)\;(.*)\](.*)\;(.*)\~(.*)\~(.*)\^(.*)\^(.*)\@(.*)\[(.*)\#(.*)\=(.*)\!(.*)\~(.*)\+(.*)\!(.*)\^(.*)/F:(.*)\#(.*)\#(.*)\-(.*)\$(.*)\$(.*)\+(.*)\%(.*)\;(.*)/G:(.*)\_(.*)/H:(.*)\_(.*)/I:(.*)\_(.*)/J:(.*)\~(.*)\@(.*)"
li = re.findall(reg,"0 12000000 p@xx^xx-pau+d=e_xx%xx^00_00~00-1!1[xx$xx]xx/A:xx-xx-xx@xx~xx/B:1_1_1@xx|xx/C:2+1+1@JPN&0/D:xx!xx#xx$xx%xx|xx&xx;xx-xx/E:xx]xx^2=2/4~100!1@120#48+xx]1$1|0[12&0]48=0^100~xx#xx_xx;xx$xx&xx%xx[xx|0]0-n^xx+xx~xx=xx@xx$xx!xx%xx#xx|xx|xx-xx&xx&xx+xx[xx;xx]xx;xx~xx~xx^xx^xx@xx[xx#xx=xx!xx~xx+xx!xx^xx/F:A4#7#2-2/4$100$1+45%18;xx/G:xx_xx/H:xx_xx/I:13_13/J:3~3@6")

phonemes = ['pau','xx'] +["ny","ty","py","ky","ry","f","br","sil","cl","a","i","u","e","o","k","g","s","z","sh","j","t","d","ch","q","ts","h","b","p","m","y","r","w","N","n","v" ]
pitches = ['xx']+[pitch + str(i) for i in range(1,8) for pitch in ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]]

local_path='/Users/Zizy/Programming/HKU/Dissertation/'
lbl_name = ['t0','t1']+['p' + str(i) for i in range(1, 17)] + \
           ['a' + str(i) for i in range(1, 6)] + \
           ['b' + str(i) for i in range(1, 6)] + \
           ['c' + str(i) for i in range(1, 6)] + \
           ['d' + str(i) for i in range(1, 10)] + \
           ['e' + str(i) for i in range(1, 61)] + \
           ['f' + str(i) for i in range(1, 10)] + \
           ['g' + str(i) for i in range(1, 3)] + \
           ['h' + str(i) for i in range(1, 3)] + \
           ['i' + str(i) for i in range(1, 3)] + \
           ['j' + str(i) for i in range(1, 4)]


def read_songs_labels():
    labs = None
    for i in range(0, 71):
        label_file_name = 'nitech_jp_song070_f001_0' + (('0' + str(i)) if i < 10 else str(i)) + '.lab'
        if not os.path.isfile(local_path + 'lab/' + label_file_name):
            continue
        with open(local_path + 'lab/' + label_file_name, 'r') as lab_file:
            lines = lab_file.readlines()
            params_list = []
            for line in lines:
                ps = re.findall(reg, line)
                params_list.append(ps[0])
            temp = []
            for params in params_list:
                params_temp = {}
                for i in range(0, 120):
                    params_temp[lbl_name[i]] = params[i]
                temp.append(params_temp)
            params_list = temp
            if labs is None:
                labs = np.array(params_list)
            else:
                labs = np.append(labs, np.array(params_list))
    return labs


def make_one_hot(data1, size):
    data1 = np.array(data1)
    return (np.arange(size) == data1[:, None]).astype(np.integer)


def make_class(data, classes):
    return [classes.index(p) for p in data]


def get_params_by_name(name, params_list):
    return [params[name] for params in params_list]


def convert_params_to_one_hot(name, classes, params_list):
    data = get_params_by_name(name, params_list)
    data = make_class(data, classes)
    data = make_one_hot(data, len(classes))
    return data


# with mono file updated
def read_songs_labels_with_mono():
    labs = []
    for i in range(0, 71):
        label_file_name = 'nitech_jp_song070_f001_0' + (('0' + str(i)) if i < 10 else str(i)) + '.lab'
        if not os.path.isfile(local_path + 'lab/' + label_file_name):
            continue
        if not os.path.isfile(local_path + 'mono/' + label_file_name):
            continue

        lab_file = open(local_path + 'lab/' + label_file_name, 'r')
        lab_file_mono = open(local_path + 'mono/' + label_file_name, 'r')
        lines = lab_file.readlines()
        lines_mono = lab_file_mono.readlines()
        params_list = []
        for index, line in enumerate(lines):
            ps = list(re.findall(reg, line)[0])
            ps2 = re.findall("(.*) (.*) (.*)", lines_mono[index])[0]

            ps[0] = ps2[0]
            ps[1] = ps2[1]
            params_list.append(ps)
        temp = []
        for params in params_list:
            params_temp = {}
            for i in range(0, 120):
                params_temp[lbl_name[i]] = params[i]
            temp.append(params_temp)
        params_list = temp

        lab_file.close()
        lab_file_mono.close()

        labs.append(params_list)
    return labs


def timing_extract(labs):
    temp = []
    stacked_timing = np.array([])
    for lab in labs:
        temp.append(len(lab))
    max_sample_size = max(temp)
    print("max sample size: %s" % max_sample_size)

    for lab in labs:
        t0 = np.array(get_params_by_name('t0', lab)).astype(float) / 10000
        t1 = np.array(get_params_by_name('t1', lab)).astype(float) / 10000
        timing = np.hstack([np.reshape(t0, (len(t0), 1)), np.reshape(t1, (len(t1), 1))])
        if len(stacked_timing) == 0:
            stacked_timing = timing
        else:
            stacked_timing = np.vstack([stacked_timing, timing])
    return stacked_timing


def feature_extract(lab):
    t0 = np.array(get_params_by_name('t0', lab)).astype(float) / 10000
    t1 = np.array(get_params_by_name('t1', lab)).astype(float) / 10000
    params_phoneme_duration = t1 - t0

    params_phoneme_duration_pre = np.insert(params_phoneme_duration[:-1], 0, np.average(params_phoneme_duration))
    params_phoneme_duration_next = np.append(params_phoneme_duration[1:], np.average(params_phoneme_duration))
    params_phonemes_one_hot = convert_params_to_one_hot('p4', phonemes, lab)
    params_phonemes_pre_one_hot = convert_params_to_one_hot('p3', phonemes, lab)
    params_phonemes_next_one_hot = convert_params_to_one_hot('p5', phonemes, lab)
    params_pitches_one_hot = convert_params_to_one_hot('e1', pitches, lab)
    params_pitches_pre_one_hot = convert_params_to_one_hot('d1', pitches, lab)
    params_pitches_next_one_hot = convert_params_to_one_hot('f1', pitches, lab)

    stacked_feature = np.hstack([params_phonemes_one_hot,
                                 params_phonemes_pre_one_hot,
                                 params_phonemes_next_one_hot,
                                 params_pitches_one_hot,
                                 params_pitches_pre_one_hot,
                                 params_pitches_next_one_hot,
                                 np.reshape(t0, (len(t0), 1)),
                                 np.reshape(t1, (len(t1), 1)),
                                 np.reshape(params_phoneme_duration, (len(params_phoneme_duration), 1)),
                                 np.reshape(params_phoneme_duration_pre, (len(params_phoneme_duration_pre), 1)),
                                 np.reshape(params_phoneme_duration_next, (len(params_phoneme_duration_next), 1)),
                                 ])
    return stacked_feature


# X
labs = np.array(read_songs_labels())
X = feature_extract(labs)
# X = timing_extract_x(labs)

# Y
labs_y = read_songs_labels_with_mono()
Y = timing_extract(labs_y)


from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint

# This returns a tensor
inputs = Input(shape=(371,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
predictions = Dense(2, activation='linear')(x)

# This creates a model that includes
# the Input layer and three Dense layers
# model = Model(inputs=inputs, outputs=predictions)
model = load_model(local_path+'timing.h5')
model.compile(optimizer=Adam(),
              loss='mse',
              )
model.summary()
model.fit(X, Y,nb_epoch=2000,batch_size=20,validation_split=0.1,
          callbacks=[ModelCheckpoint(local_path+"checkpoint/{epoch:02d}.h5",save_best_only=True, period=10),
                     TensorBoard(log_dir=local_path+'./tmp/log', histogram_freq=0, write_graph=True,
                                          write_images=True)])  # starts training
