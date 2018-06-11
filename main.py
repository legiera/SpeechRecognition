import scipy.io.wavfile as wavfile 
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import math
from scipy.io import wavfile
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from python_speech_features import mfcc
from python_speech_features import delta
from pydub import AudioSegment
import logging
import random
import pywt
from dtw import dtw

######################################################################################################################
# Constants
######################################################################################################################
WIN_LENGTH = 0.025
WIN_STEP = 0.01
NUM_CEP = 13


######################################################################################################################
# Methods for prepare data to futher processing and classification
######################################################################################################################
def match_signal_file_with_label_track_file():
    signal_and_label_track_pairs = list()
    for _file in os.listdir("data/"):
        if (_file.endswith(".wav")):
            signal_and_label_track_pairs.append({"signal_path": _file, "label": os.path.splitext(_file)[0] + ".txt"})
    return signal_and_label_track_pairs


def get_signal_from_file(signal_file_name):
    fs, signal = wavfile.read(signal_file_name)
    if (signal.ndim > 1):
        signal = signal[:, 0]
    return signal, fs


def show_signal(signal, label):
    plt.plot(signal)
    plt.xlabel(label)
    plt.show()


def parse_track_file(label_track_file_name):
    label_track_file = open(label_track_file_name)
    label_track_file_line = label_track_file.readlines()
    label_track_list = list();
    for line in label_track_file_line:
        label_item = re.split(r'\s+', line);
        label_track_list.append({"timestamp_start": float(label_item[0].replace(",", ".")) * 1000,
                                 "timestamp_end": float(label_item[1].replace(",", ".")) * 1000,
                                 "label": label_item[2]})
    return label_track_list  # ["timestamp_start":0, "timestamp_end": 1234, "label": "GARAZ", "label_id"=}]


def audio_segment(signal_file_name, label_track_file):
    dataset = list();
    file_number = (os.path.splitext(signal_file_name))[0][-1:]
    index_number = (os.path.basename(signal_file_name)).split("_")[0]
    gender = (os.path.basename(signal_file_name)).split("_")[2]
    label_track_list = parse_track_file(label_track_file)
    for label_track in label_track_list:
        try:
            newAudio = AudioSegment.from_wav(signal_file_name);
            newAudio = newAudio[label_track["timestamp_start"]:label_track["timestamp_end"]]
            signal = "data_output/" + label_track["label"] + "_" + index_number + "_" + file_number + ".wav"
            newAudio.export(signal, format="wav")
            dataset.append({"signal_path": signal, "label": label_track["label"], "gender": gender})
        except (FileNotFoundError):
            print("Not able to segment %s file. File not found error" % signalFileName)
    return dataset  # [{"signal_path": data_output/signal.wav, "label":"GARAZ", "gender":"K"}]


######################################################################################################################
# (1) Slice signals to smaller part which represents one single word.
# (2) Extract raw data signal from particular file and assign it to its item


# Dataset dictionary structure:
# dataset =
#     {'signal_path': 'data_output/OTWORZ_258118_3.wav',
#     'label': 'OTWORZ',
#     'gender': 'K',
#     'fs': 44100,
#     'signal': array([48, 24, 38, ...,  0, 14,  6]}
######################################################################################################################

# (1)
data = list()
for pair in match_signal_file_with_label_track_file():
    data += audio_segment("data/" + pair['signal_path'], "data/" + pair['label'])

# (2)
for d in data:
    signal, fs = get_signal_from_file(d['signal_path'])
    d["fs"] = fs
    d["signal"] = signal

print("It was created %s new .wav files, where each signal represents one word." % len(data))


def get_MFCC(signal, samplerate):
    # Mute warnings
    logging.getLogger().setLevel(logging.ERROR)
    mfcc_features = mfcc(signal, samplerate, winlen=WIN_LENGTH, winstep=WIN_STEP, numcep=NUM_CEP, winfunc=np.hamming)
    # Unmute warnings
    logging.getLogger().setLevel(logging.NOTSET)
    return mfcc_features


# Prepare dataset for train and test.
dataset = list()
for d in data:
    # For classification was used only initial 13x20 coefficients
    dataset_features = np.reshape(((get_MFCC(d["signal"], d["fs"]))[:20]), (-1))
    dataset.append([dataset_features, d['label']])

# Example of MFFC spectrum image
print("Example of MFFC result image")
mffc_example = get_MFCC(data[0]["signal"], data[0]["fs"])
plt.xlabel("frames")
plt.ylabel("frequency")
plt.title(data[0]["label"])
plt.imshow(mffc_example.T)
plt.show()


# Prepare datasets for classification
def get_datasets(dataset, N):
    random.shuffle(dataset)

    x, y = zip(*dataset)
    x = list(x)
    y = list(y)

    train_x = x[0:N]
    train_y = y[0:N]
    test_x = x[N:len(dataset)]
    test_y = y[N:len(dataset)]

    #   Standardize features
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = [s.reshape(-1, 20) for s in np.array(scaler.transform(train_x))]
    test_x = [s.reshape(-1, 20) for s in np.array(scaler.transform(test_x))]

    return train_x, train_y, test_x, test_y


def cross_validation(train_x, train_y, test_x, test_y):
    predict = ['' for x in range(len(test_y))]

    for test_iter, test in enumerate(test_x):
        dist_min = math.inf
        for train_iter, train in enumerate(train_x):
            dist, _, _, _ = dtw(test, train, dist=lambda test, train: np.linalg.norm(test - train, ord=1))
            if dist_min > dist:
                dist_min = dist
                predict[test_iter] = train_y[train_iter]
    return predict


train_x, train_y, test_x, test_y = get_datasets(dataset_train, 500)
predicted = cross_validation(train_x, train_y, test_x, test_y)
expected = test_y

print("Classification report: \n\n%s\n"
      % (metrics.classification_report(expected, predicted)))
mat = metrics.confusion_matrix(expected, predicted)
label_names = list(set(expected))
plt.figure()
plt.imshow(mat, interpolation='nearest', cmap='Blues')
plt.title('normalized confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(label_names))
plt.xticks(tick_marks, label_names, rotation=90)
plt.yticks(tick_marks, label_names)
plt.tight_layout()

accuracy = metrics.accuracy_score(expected, predicted)
print('Accuracy classification score: {0:.2f}%'.format(100*accuracy))
precision = metrics.precision_score(expected, predicted, average='weighted')
print('Precision classification score: {0:.2f}%'.format(100*precision))