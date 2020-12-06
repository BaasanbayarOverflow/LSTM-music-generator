import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import tensorflow as tf
import numpy as np
import glob as gl
import pickle
from music21 import note, converter, instrument, chord

from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint



def noteExtracter():
    note_arr = []

    for file in gl.glob("raw_data/*.mid"):

        midi_file = converter.parse(file)
        print(f"Parsing {file}")

        notes = None
        try:
            var = instrument.partitionByInstrument(midi_file)
            notes = var.parts[0].recurse()
        except:
            notes = midi_file.flat.notes 

        for nt in notes:
            if isinstance(nt, note.Note):
                note_arr.append(str(nt.pitch))
            elif isinstance(nt, chord.Chord):
                note_arr.append('.'.join(str(n) for n in nt.normalOrder))

        with open('data/notes', 'wb') as path:
            pickle.dump(note_arr, path)

    return note_arr


def seqPreparetion(notes, vocab):
    seqLength = 100
    pitches = sorted(set(it for it in notes))
    noteEnum = dict((note, num) for num, note in enumerate(pitches))
    
    networkInput = []
    networkOutput = []
    
    for i in range(0, len(notes) - seqLength, 1):
        seqIN = notes[i : i+seqLength]  
        seqOut = notes[i + seqLength]
        networkInput.append([noteEnum[ch] for ch in seqIN])
        networkOutput.append(noteEnum[seqOut])
    
    patterns = len(networkInput)
    
    networkInput = np.reshape(networkInput, (patterns, seqLength, 1))
    networkOutput = np_utils.to_categorical(networkOutput)
    return(networkInput, networkOutput)


def neuralNetwork(networkIN, vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(networkIN.shape[1], networkIN.shape[2]), recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.2,))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return model


def train(model, netIn, netOut):
    fileName = "weights.hdf5"
    checkpoint = ModelCheckpoint(fileName, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callback = [checkpoint]
    
    model.fit(netIn, netOut, epochs=500, batch_size=128, callbacks=callback)


tf.debugging.set_log_device_placement(True)
notes = noteExtracter()
vocab = len(set(notes))
networkIN, networkOUT = seqPreparetion(notes, len(set(notes)))
model = neuralNetwork(networkIN, vocab)
train(model, networkIN, networkOUT)


