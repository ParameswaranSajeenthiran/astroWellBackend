# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import pandas as pd
import io

from  flask import Flask
# from sklearn.preprocessing import LabelEncoder
from flask import  request
import  statistics
import numpy as np
import soundfile

app = Flask(__name__)
import librosa
import os, glob, pickle

from flask_cors import CORS
CORS(app)


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result


def predict():
    filename = 'modelForPrediction1.sav'
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

    feature = extract_feature("/content/03-01-08-02-02-01-17.wav", mfcc=True, chroma=True, mel=True)

    feature = feature.reshape(1, -1)

    prediction = loaded_model.predict(feature)
    prediction

@app.get("/get")
def hello_world():
    return "test"


@app.route('/result', methods=['GET', 'POST'])
def result():
    print('test ')
    file = request.files.get('audio_data')
    if file:
        tmp = io.BytesIO(file.read())
        data, samplerate = soundfile.read(tmp)
        soundfile.write("data.wav",data,samplerate)

        filename = 'modelForPrediction1.sav'
        loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

        feature = extract_feature("data.wav", mfcc=True, chroma=True, mel=True)

        feature = feature.reshape(1, -1)

        prediction = loaded_model.predict(feature)
        print(prediction)

        print(data)

    # ...

    return "succes"


