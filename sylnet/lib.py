"""
Created on Tue May 28 12:54:43 2019 (SS)
Modified on Wed June 5 12:30:00 2019 (OR)

@author: Shreyas Seshadri, Okko Rasanen
"""

from __future__ import print_function
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import tensorflow as tf
import time
from . import SylNet_model
import os


########### Global Static Constants ###########

mainFile = os.path.dirname(os.path.realpath(__file__))
modelFile = mainFile + '/trained_models/'
means_path = modelFile + 'means.npy'
std_path = modelFile + 'stds.npy'
MODEL_PATH = modelFile + 'model_trained.ckpt'

REQUIRED_SAMPLING_RATE = 16000
MAX_DETECTED_SYLLABLES = 91   # HARD CODED AS THIS IS WHAT THE MAIN MODEL IS TRAINED ON
MEAN = np.load(means_path)  # Z norm data based on training data mean and std
STD = np.load(std_path)
frame_size = round(0.025*REQUIRED_SAMPLING_RATE)
frame_step = round(0.01*REQUIRED_SAMPLING_RATE)
MELS_DIMENSION = 24

########### Mandatory Operations for TF 2.0. #############

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()


########### Common Datatypes #############

class SylNet:

    def __init__(self) -> None:
        self._session = None
        self._x = None
        self._logits = None
        self._predictions = None

    def init(self):
        session, x, logits, predictions, ids, ids_len, is_train = load_model(MODEL_PATH)

        self._session = session
        self._x = x
        self._logits = logits
        self._predictions = predictions
        self._ids = ids
        self._ids_len = ids_len
        self._is_train = is_train

    
    def run(self, audio_data, sampling_rate):
        start_ts = time.time()

        y = librosa.core.resample(y=audio_data, orig_sr=sampling_rate, target_sr=REQUIRED_SAMPLING_RATE)
        normalized = librosa.util.normalize(y)
        X = np.transpose(20*np.log10(librosa.feature.melspectrogram(y=normalized, sr=REQUIRED_SAMPLING_RATE, n_mels=MELS_DIMENSION, n_fft=frame_size, hop_length=frame_step)+0.00000000001))    
        X = (X - MEAN)/STD
        
        X_mini = X
        X_mini = X_mini[np.newaxis,:,:]
        l_mini = np.asarray([X_mini.shape[1]],dtype=np.int32)
        E_list = np.asarray([[0,X_mini.shape[1]-1]])
        PRED = self._session.run([self._predictions], feed_dict={self._x: X_mini, self._ids:E_list, self._ids_len:l_mini, self._is_train:False})

        Y = sum(sum(np.asarray(PRED[0]>=0.5, dtype=np.float32)))

        print(f"computed syllables count: {Y}")

        execution_time = time.time() - start_ts
        print(f"execution time: {execution_time}")

        return Y


    def shutfdown(self):
        self.session.close()


########### Pipeline #############

def create_preloaded_session(model_path) -> tf.compat.v1.Session:
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
    tfconfig = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    
    session = tf.compat.v1.Session(config=tfconfig)
    saver = tf.compat.v1.train.Saver()

    session.run(init_op)
    saver.restore(session, model_path)

    return session

########### Transform Audio Chunk #############

def load_model(model_path):
    residual_channels = 128
    filter_width = 5
    dilations = [1]*10
    input_channels = MELS_DIMENSION
    output_channels = MAX_DETECTED_SYLLABLES
    postnet_channels= 128
    droupout_rate = 0.5

    ids = tf.compat.v1.placeholder(shape=(None, 2), dtype=tf.int32)
    ids_len = tf.compat.v1.placeholder(shape=(None), dtype=tf.int32)
    is_train = tf.compat.v1.placeholder(dtype=tf.bool)
    S = SylNet_model.CNET(name='S',
                    input_channels=input_channels,
                    output_channels=output_channels,
                    residual_channels=residual_channels,
                    filter_width=filter_width,
                    dilations=dilations,
                    postnet_channels=postnet_channels,
                    cond_dim=None,
                    do_postproc=True,
                    do_GLU=True,
                    endList=ids,
                    seqLen=ids_len,
                    isTrain=is_train,
                    DRrate=droupout_rate)


    # data placeholders of shape (batch_size, timesteps, feature_dim)
    x = tf.compat.v1.placeholder(shape=(None, None, input_channels), dtype=tf.float32)
    logits = S.forward_pass(x)
    predictions = tf.nn.sigmoid(logits)


    init=tf.compat.v1.global_variables_initializer()
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
    tfconfig = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    saver = tf.compat.v1.train.Saver()
    
    session = tf.compat.v1.Session(config=tfconfig)

    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    session.run(init_op)
    saver.restore(session, model_path)

    return session, x, logits, predictions, ids, ids_len, is_train


if __name__ == "__main__":
    audio_file = "/Users/a.boltava/Desktop/run.wav"
    signal, sr = librosa.load(audio_file)

    sylnet_impl = SylNet()
    sylnet_impl.init()
    
    syllables = sylnet_impl.run(signal, sr)
    print(syllables)

    syllables = sylnet_impl.run(signal, sr)
    syllables = sylnet_impl.run(signal, sr)
