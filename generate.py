import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pickle as pc
import tensorflow as tf

from music21 import instrument, stream, note, chord

from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import LSTM


def seqGen(notes, pitchNames, vocab):
    noteEnum1 = dict((note, num) for num, note in enumerate(pitchNames))
    
    seqLen = 200
    networkIN = []
    networkOUT = []
    
    for i in range(0, len(notes) - seqLen, 1):
        seqIn = notes[i : i+seqLen]
        seqOut = notes[i + seqLen]
        networkIN.append([noteEnum1[ch] for ch in seqIn])
        networkOUT.append(noteEnum1[seqOut])
    patterns = len(networkIN)
    
    normalizedIN = np.reshape(networkIN, (patterns, seqLen, 1))
    normalizedIN = normalizedIN / float(vocab)
    
    return(networkIN, normalizedIN)



def neuralNet(networkIN, vocab):
    
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
    
    model.load_weights('weights.hdf5')
    return model


def genNotes(model, netIN, pitchName, vocab):
    begin = np.random.randint(0, len(netIN)-1)
    noteEnum2 = dict((num, note) for num, note in enumerate(pitchName))
    print(noteEnum2)
    pattern = netIN[begin]
    predictOut= []
    
    for note in range(400):
        prediction = np.reshape(pattern, (1, len(pattern), 1))
        prediction = prediction / float(vocab)
        predictionResult = model.predict(prediction, verbose = 1)
        ind = np.argmax(predictionResult)
        result = noteEnum2[ind]
        predictOut.append(result)
        pattern.append(ind)
        pattern = pattern[1:len(pattern)]
    return predictOut



def generateMusicFile(notes):
    
    res = []
    offset = 0
    
    for singleNote in notes:
        if ('.' in singleNote) or singleNote.isdigit():
            chordAll = singleNote.split('.')
            music = []
            for nt in chordAll:
                n = note.Note(int(nt))
                n.storedInstrument = instrument.Piano()
                music.append(n)
            newChord = chord.Chord(music)
            newChord.offset = offset
            res.append(newChord)
        else:
            newNote = note.Note(singleNote)
            newNote.offset = offset
            newNote.storedInstrument = instrument.Piano()
            res.append(newNote)
        offset += 0.5
    fileStream = stream.Stream(res)
    fileStream.write('midi', fp='music.mid')



with open('data/notes', 'rb') as filePath:
    notes = pc.load(filePath)

pitchNames = sorted(set(note for note in notes))
vocab = len(set(notes))

tf.debugging.set_log_device_placement(True)
netIn, normalIn = seqGen(notes, pitchNames, vocab) 
model = neuralNet(normalIn, vocab)
predictionRes = genNotes(model, netIn, pitchNames, vocab)
generateMusicFile(predictionRes)
