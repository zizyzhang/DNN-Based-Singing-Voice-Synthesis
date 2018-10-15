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


print(len(li[0]))


def read_songs_labels():
    labs = []
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

            labs.append(params_list)
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


def feature_expend(labs):
    modified_labs = []
    for index, lab_file in enumerate(labs):
        modified_labs.append([])
        max_time_stamp = int(lab_file[-1]["t1"])
        f0_size = math.ceil((max_time_stamp / 10000 / 5)) + 1
        sample_index = 0
        for sample in range(0, f0_size):
            if int(lab_file[sample_index]["t1"]) / 10000 / 5 > sample:
                modified_labs[index].append(lab_file[sample_index])
            else:
                if len(lab_file) > sample_index + 1:
                    sample_index += 1
                modified_labs[index].append(lab_file[sample_index])

    return modified_labs




# labs  = read_songs_labels()
labs = read_songs_labels_with_mono()

labs = feature_expend(labs)

stacked_labs_with_feature = []
feature_num  = 0

temp = []
for lab in labs:
    temp.append(len(lab))
max_sample_size = max(temp)
print("max sample size: %s" % max_sample_size)

for lab in labs:
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
                                 params_pitches_one_hot,
                                 params_pitches_pre_one_hot,
                                 params_pitches_next_one_hot,
                                 np.reshape(params_phoneme_duration, (len(params_phoneme_duration), 1)),
                                 np.reshape(params_phoneme_duration_pre, (len(params_phoneme_duration_pre), 1)),
                                 np.reshape(params_phoneme_duration_next, (len(params_phoneme_duration_next), 1)),
                                 ])
    stacked_feature = np.vstack([stacked_feature,
                                 np.zeros((max_sample_size-stacked_feature.shape[0],stacked_feature.shape[1]))])

    stacked_labs_with_feature.append(stacked_feature)

stacked_labs_with_feature = np.array(stacked_labs_with_feature)

# np.reshape(stacked_labs_with_feature ,(31, max_sample_size, 314))
# print(len(stacked_labs_with_feature))
# print(stacked_labs_with_feature[:-1])

def read_f0s():
    max_f0_size = 0
    f0s = []
    for i in range(0, 71):
        f0_file_name = 'nitech_jp_song070_f001_0' + (('0' + str(i)) if i < 10 else str(i)) + '_f0.txt'
        if not os.path.isfile(local_path + 'gen/' + f0_file_name):
            continue
        with open(local_path + 'gen/' + f0_file_name, 'r') as f0_file:
            lines = f0_file.readlines()
            lines = np.array(lines).astype(np.float)

            if len(lines) > max_f0_size:
                max_f0_size = len(lines)
            f0s.append(lines)

    # 即要生成的最多个f0，
    print(max_f0_size)

    for index, f0 in enumerate(f0s):
        f0s[index] = np.append(f0, np.zeros(max_f0_size - len(f0)))

    f0s = np.array(f0s)
    f0s = np.reshape(f0s, (f0s.shape[0], f0s.shape[1], 1))
    return f0s, max_f0_size


f0s, max_f0_size = read_f0s()

print(f0s.shape)
print(stacked_labs_with_feature.shape)
max_f0_size


def wavenetBlock(n_atrous_filters, atrous_filter_size, dilation_rate):
    def f(input_):
        residual = input_
        tanh_out = Conv1D(n_atrous_filters, atrous_filter_size,
                          dilation_rate=dilation_rate,
                          padding='causal',

                          activation='tanh')(input_)
        sigmoid_out = Conv1D(n_atrous_filters, atrous_filter_size,
                             dilation_rate=dilation_rate,
                             padding='causal',
                             activation='sigmoid')(input_)
        merged = Multiply()([tanh_out, sigmoid_out])
        skip_out = Conv1D(1, 1, activation='relu', padding='same')(merged)
        out = Add()([skip_out, residual])
        return out, skip_out

    return f


def get_basic_generative_model(input_dim):
    input_ = Input(shape=(input_dim, 1))
    A, B = wavenetBlock(64, 2, 2)(input_)
    skip_connections = [B]
    for i in range(20):
        A, B = wavenetBlock(64, 2, 2 ** ((i + 2) % 9))(A)
        skip_connections.append(B)
    net = Add()(skip_connections)
    net = Activation('relu')(net)
    net = Convolution1D(1, 1, activation='relu')(net)
    net = Convolution1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(1000, activation='softmax')(net)
    model = Model(input=input_, output=net)
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    return model


def frame_generator(total_frame, input_data_size, frame_shift, control_input, minibatch_size=20):
    X = []
    y = []
    while 1:

        for lab_file_index in range(0, len(total_frame)):
            total_frame_len = len(total_frame[lab_file_index])

            for cold_start_index in range(0, input_data_size):
                frame = np.zeros((input_data_size - cold_start_index, 1))
                frame = np.append(frame, total_frame[lab_file_index][:cold_start_index])
                frame = np.append(frame, control_input[lab_file_index][cold_start_index])
                X.append(frame.reshape((input_data_size + control_input.shape[2], 1)))

                temp = total_frame[lab_file_index][cold_start_index]
                target_val = int(temp)
                y.append((np.eye(1000)[target_val]))

            for i in range(0, total_frame_len - input_data_size - 1, frame_shift):

                # 获取一帧
                frame = total_frame[lab_file_index][i:i + input_data_size]

                if len(frame) < input_data_size:
                    break

                if i + input_data_size >= total_frame_len:
                    break

                # 获取该帧 frame 后面的那个sample
                temp = total_frame[lab_file_index][i + input_data_size]
                target_val = int(temp)

                # frame.shape = (64,1)

                # Control Input
                frame = np.append(frame, control_input[lab_file_index][i + input_data_size])
                # print(frame.shape)

                # X值即前一帧
                X.append(frame.reshape((input_data_size + control_input.shape[2], 1)))

                # 相当于一个one hot vec. 只激活第target_val个元素
                y.append((np.eye(1000)[target_val]))

                # 获取一个mini-batch的数据返回
                if len(X) == minibatch_size:
                    yield np.array(X), np.array(y)
                    X = []
                    y = []


def f0_generate(model, f0_window_size, input_size, control_input):
    print('Generating audio...')
    new_f0s = np.zeros(max_f0_size)
    seed_audio = np.zeros(f0_window_size)
    curr_sample_idx = 0
    while curr_sample_idx < new_f0s.shape[0]:
        seed = np.append(seed_audio, control_input[curr_sample_idx])

        distribution = np.array(model.predict(seed.reshape(1, input_size, 1)
                                              ), dtype=float).reshape(1000)

        # 让output变成一个零和的输出
        distribution /= distribution.sum().astype(float)

        # 按照几率输出预测值（很厉害），即把one hot转换成了单一数值
        predicted_val = np.random.choice(range(1000), p=distribution)

        new_f0s[curr_sample_idx] = predicted_val
        seed_audio[:-1] = seed_audio[1:]
        seed_audio[-1] = predicted_val

        # percent
        # pc_str = str(round(100 * curr_sample_idx / float(new_audio.shape[0]), 2))
        # sys.stdout.write('Percent complete: ' + pc_str + '\r')
        # sys.stdout.flush()
        curr_sample_idx += 1
        if curr_sample_idx % int(new_f0s.shape[0] / 10) == 0:
            print(curr_sample_idx / int(new_f0s.shape[0] / 10))
    print('Audio generated.')
    return new_f0s.astype(np.int16)

n_epochs = 2000
input_data_size = 64
frame_shift = 1

# model = get_basic_generative_model(input_data_size+stacked_labs_with_feature.shape[2])
model = load_model(local_path+"model.h5")
training_data_gen = frame_generator(f0s[0:29], input_data_size, frame_shift, stacked_labs_with_feature[0:29])
validation_data_gen = frame_generator(f0s[29:32], input_data_size, frame_shift, stacked_labs_with_feature[29:32])

model.fit_generator(training_data_gen, samples_per_epoch=3000, nb_epoch=n_epochs,
                    validation_data=validation_data_gen, nb_val_samples=500, verbose=1,
                    callbacks=[TensorBoard(log_dir='./tmp/log', histogram_freq=0, write_graph=True,
                                          write_images=True)])