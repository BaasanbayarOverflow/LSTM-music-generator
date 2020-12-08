import tensorflow as tf
import numpy as np
import glob as gl
import pickle as pc
import os
import sys
from music21 import note, converter, instrument, chord, stream

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def noteExtract():
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
            pc.dump(note_arr, path)
    return note_arr

def noteGenerate(model, networkInput, pitches, vocabulary):
    begin = np.random.randint(0, len(networkInput) - 1)
    noteEncoding = dict((num, nt) for num, nt in enumerate(pitches))
    pattern = networkInput[begin]
    predictOut = []
    for nt in tqdm(range(300)):
        prediction = np.reshape(pattern, (1, len(pattern), 1))
        prediction = prediction / float(vocabulary)
        predictionResult = model.predict(prediction, verbose = 1)
        index = np.argmax(predictionResult)
        result = noteEncoding[index]
        predictOut.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return predictOut

def netSequence(notes, vocabulary, pitches, sequenceLength, flag):
    networkInput = []
    networkOutput = []
    noteEncoding = dict((nt, num) for num, nt in enumerate(pitches))
    for i in tqdm(range(0, len(notes) - sequenceLength)):
        sequenceInput = notes[i : i+sequenceLength]
        sequenceOutput = notes[i + sequenceLength]
        networkInput.append([noteEncoding[ch] for ch in sequenceInput])
        networkOutput.append(noteEncoding[sequenceOutput])
    pattern = len(networkInput)
    if (flag == True): #train
        networkInput = np.reshape(networkInput, (pattern, sequenceLength))
        networkOutput = np_utils.to_categorical(networkOutput)
        return(networkInput, networkOuput)
    else:   #predict
        normalizedInput = np.reshape(networkInput, (pattern, sequenceLength))
        normalizedInput = normalizedInput / float(vocabulary)
        return(networkInput, normalizedInput)

def neuralNetwork(networkInput, vocabulary, flag):
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
    if (flag == True):
        model.load_weights('weights.hdf5')
    return model

def trainLSTM(model, networkInput, networkOuput, n):
    weightFile = "weight.hdf5"
    checkpoint = ModelCheckpoint(weightFile, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback = [checkpoint]
    model.fit(networkInput, networkOuput, epochs = n, batch_size = 128, callbacks = callback)  #default batch size = 32

def musicGenerate(notes):
    results = []
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
            results.append(newChord)
        else:
            newNote = note.Note(singleNote)
            newNote.offset = offset
            newNote.storedInstrument = instrument.Piano()
            results.append(newNote)
        offset += 0.5
    fileStream = stream.Stream(results)
    fileStream.write('midi', fp='music.mid')
    print('Music successfully generated')

def train(sequenceLength, n):
    print('Feed data extraction from midi files phase is initialized')
    notes = noteExtract()
    vocabulary = len(set(notes))
    pitches = sorted(set(it for it in notes))
    print('Data sequence generation phase is started')
    networkInput, networkOuput = netSequence(notes, vocabulary, pitches, sequenceLength, True)
    model = neuralNetwork(networkInput, vocabulary, False)
    print('Model training phase is initialized')
    trainLSTM(model, networkInput, networkOuput, n)

def predict(sequenceLength):
    print('Prediction phase is initialized')
    with open('data/notes', 'rb') as dataPath:
        notes = pc.load(dataPath)
    pitches = sorted(set(nt for nt in notes))
    vocabulary = len(set(notes))
    networkInput, normalizedInput = netSequence(notes, vocabulary, pitches, sequenceLength, False)
    model = neuralNetwork(networkInput, vocabulary, True)
    prediction = noteGenerate(model, networkInput, pitches, vocabulary)
    musicGenerate(prediction)

def help():
    print("Hi. I'm Sonar :)))")
    print('-T/-t    --- train LSTM model')
    print('-g/-G    --- generate music file from TRAINED model')

def main(i):
    tf.debugging.set_log_device_placement(True)
    sequenceLength = int(input('Please enter sequence length - '))
    n = int(input('Please input epochs number - '))
    if (i == '-t' or i == '-T'):
        train(sequenceLength, n)
    elif (i == '-g' or i == '-G'):
        predict(sequenceLength)
    else:
        help()

main(sys.argv[1])
