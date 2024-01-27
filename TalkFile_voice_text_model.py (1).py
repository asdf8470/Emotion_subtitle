import pandas as pd
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sentence_transformers import SentenceTransformer
import librosa
import librosa.display




txt_data = pd.read_csv(r"C:\Users\s_sanghkim\kjk\voice_emotion_classification\dataset\data_final.csv", encoding = 'cp949')
print(f"txt_data 길이: {len(txt_data)}")
print(f"txt_data shape: {txt_data.shape}")


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

#def stretch(data, rate=0.8):
#    return librosa.effects.time_stretch(data, rate)

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

#def pitch(data, sampling_rate, pitch_factor=0.7):
#    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def pitch(data, sampling_rate=0.8, n_step=4):  # y, sr=sr, n_steps=4  / , pitch_factor=0.7
    return librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps = n_step)




def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.0)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.concatenate((result, res2), axis=0)


    # data with stretching and pitching
    new_data = stretch(data)  # error
    data_stretch_pitch = pitch(new_data)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.concatenate((result, res3), axis=0)

    return result

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result
#####################################################

txt_data['1번 감정'] = txt_data['1번 감정'].apply(str.lower)
txt_data['2번 감정'] = txt_data['2번 감정'].apply(str.lower)
txt_data['3번 감정'] = txt_data['3번 감정'].apply(str.lower)
txt_data['4번 감정'] = txt_data['4번 감정'].apply(str.lower)
txt_data['5번 감정'] = txt_data['5번 감정'].apply(str.lower)


def get_keys(dic):  # returns a key for max values in dic
    key_list = list(dic.keys())
    val_list = list(dic.values())
    pos = val_list.index(max(val_list))
    return key_list[pos]


final_label = []
for i in range(len(txt_data)):
    sentiments = {'angry': 0, 'sadness': 0, 'happiness': 0, 'fear': 0, 'disgust': 0, 'surprise': 0, 'neutral': 0}
    sentiments[txt_data.iloc[i]['1번 감정']] += txt_data.iloc[i]['1번 감정세기']
    sentiments[txt_data.iloc[i]['2번 감정']] += txt_data.iloc[i]['2번 감정세기']
    sentiments[txt_data.iloc[i]['3번 감정']] += txt_data.iloc[i]['3번 감정세기']
    sentiments[txt_data.iloc[i]['4번 감정']] += txt_data.iloc[i]['4번감정세기']
    sentiments[txt_data.iloc[i]['5번 감정']] += txt_data.iloc[i]['5번 감정세기']

    final_label.append(get_keys(sentiments))

final_label_df = pd.DataFrame(final_label, columns=['final_label'])
new_txt_data = pd.concat([txt_data[['wav_id', '발화문']], final_label_df], axis=1)






#####################################################3


new_txt_data = txt_data[['wav_id','발화문', '상황']]
new_txt_data = new_txt_data.rename(columns={'상황':'final_label'})
new_txt_data['final_label'].apply(str.lower)
new_txt_data.head(5)

fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x = new_txt_data['final_label'], palette = "husl",  ax = ax)
plt.show()

####################  음성데이터 로드 ####################

audio_path = 'C:/Users/s_sanghkim/PycharmProjects/pythonProject/voice_emotion_classification/dataset/5차_wav'
wav_list = os.listdir(audio_path)
wav_list_tmp = random.sample(wav_list, 10000) #colab 용량 한계로 1000개만

wav_list_tmp_id = []  # wav 파일명에서 .wav 제외하고 순수한 wav_id를 추출
for i in range(10000):
    wav_list_tmp_id.append(wav_list_tmp[i][:-4])

wav_list_tmp_label, wav_list_tmp_sentence = [], []
for x in wav_list_tmp_id:
    wav_list_tmp_label.append(new_txt_data[new_txt_data['wav_id'] == x]['final_label'].values[0])
    wav_list_tmp_sentence.append(new_txt_data[new_txt_data['wav_id'] == x]['발화문'].values[0])

wav_df = pd.DataFrame(
    {'wav_id': wav_list_tmp_id,
     'final_label': wav_list_tmp_label,  # 'final_label': wav_list_tmp_label,
     'sentence': wav_list_tmp_sentence
     })

wav_df


###################### feature extraction ###########################

print(wav_df['wav_id'])
print(wav_df['final_label'])
print(len(wav_df))
zip(wav_df['wav_id'][1], wav_df['final_label'][1])


path_sample = 'C:/Users/s_sanghkim/PycharmProjects/pythonProject/voice_emotion_classification/dataset/5차_wav/5f007a73704f492ee1256114.wav'
data, sample_rate = librosa.load(path_sample, duration=2.5, offset=0.0)


X_audio, Y = [], []
for path, label in zip(wav_df['wav_id'], wav_df['final_label']):
    audio_features = get_features(audio_path + '/' + path + '.wav')
    X_audio.append(audio_features)
    Y.append(label)

audio_features = pd.DataFrame(X_audio)
final_df = pd.concat([audio_features, wav_df[['wav_id', 'final_label', 'sentence']]], axis=1)
final_df.head(3)



######################## text embedding ############################

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
import keras

from tensorflow.python.keras.utils import np_utils
#from keras.utils import np_utils, to_categorical

class text_embedding():
    def __init__(self, model_name):
        self.model_name = model_name
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        embedding_model = SentenceTransformer(self.model_name)
        embedding_vec = embedding_model.encode(X['sentence'])
        X_val = np.concatenate((X.drop(['final_label', 'wav_id', 'sentence'], axis=1), embedding_vec), axis=1)
        return X_val

def custom_model(x_train):
  model= keras.models.Sequential()
  model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
  model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

  model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
  model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

  model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
  model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
  model.add(Dropout(0.2))

  model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
  model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

  model.add(Flatten())
  model.add(Dense(units=32, activation='relu'))
  model.add(Dropout(0.3))

  model.add(Dense(units=6, activation='softmax'))
  model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

  #model.summary()
  return model

scaler = StandardScaler()
encoder = OneHotEncoder()
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001) #learning rate 조절
pre_trained_models = ['sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens',
'sentence-transformers/multi-qa-distilbert-cos-v1',
'jhgan/ko-sroberta-multitask',
'all-distilroberta-v1',
'jhgan/ko-sbert-multitask',
'all-MiniLM-L12-v2', 'jhgan/ko-sroberta-sts']


Y = final_df['final_label'].values
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

for i in pre_trained_models:
  txt_embed = text_embedding(model_name = i)
  X = txt_embed.transform(final_df)

  x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)

  x_train = np.expand_dims(x_train, axis=2)
  x_test = np.expand_dims(x_test, axis=2)
  #x_train.shape, y_train.shape, x_test.shape, y_test.shape

  model = custom_model(x_train)
  history=model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
  print("Pre-trained Model: ", i)
  print("Test Accuracy: ",test_acc)
