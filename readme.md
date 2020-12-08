# SONAR

<h1 align="center">
  <br>
  <img src="https://imgprd19.hobbylobby.com/d/f4/8d/df48d440fc8512673aa57e7bf7ee6665133fd4db/350Wx350H-946806-0519-px.jpg" alt="note" width='250'>
</h1>

<h3 align='center'>Piano music generator based on LSTM neural network</h3>

Sonar is a project resulted from Machine learning case study :computer:. Sonar project's idea is motivated by an article
<a href='https://medium.com/@ageitgey/machine-learning-is-fun-part-2-a26a10b68df3'>Machine learning is fun part2</a>
by <a href='https://medium.com/@ageitgey'>Adam Geitgey</a> :raised_hands:. When i was reading this article, i found out that Super Mario game levels
can be generated from text file using RNN. So i thought that if i extract notes from music file and save those into text file all the other processes will be same as written in the article. That is where i started my own little case study. While doing my research i discovered LSTM RNN architecture is specifically designed to work with sequences of data and used it in my project.

## Used Libraries
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Music21](http://web.mit.edu/music21/)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [Tqdm](https://tqdm.github.io/)

## Requirements /cause of compatibility issues used older versions/
- [Musescore](https://musescore.org/en)
- [CUDA toolkit version 10.1](https://developer.nvidia.com/cuda-downloads)
- [Nvidia GPU driver version 486.31](https://www.nvidia.com/Download/driverResults.aspx/147971/tr-tr)/Studio driver/

## Code examples

```sh
$ python sonar.py -T
```
<img src="https://github.com/BaasanbayarOverflow/LSTM-music-generator/blob/master/images/example2.png" alt="example">

```sh
$ python sonar.py -G
```
<img src="https://github.com/BaasanbayarOverflow/LSTM-music-generator/blob/master/images/example1.png" alt="example">

## Results

1. Result of 1 epoch
<img src="https://github.com/BaasanbayarOverflow/LSTM-music-generator/blob/master/images/one_epoch.png" alt="note">

2. Result of 250 epoch
<img src="https://github.com/BaasanbayarOverflow/LSTM-music-generator/blob/master/images/250_epoch.png" alt="note">

3. Result of 500 epoch
<img src="https://github.com/BaasanbayarOverflow/LSTM-music-generator/blob/master/images/500_epoch.png" alt="note">

4. Result of 1 epoch after added 10 new midi file
<img src="https://github.com/BaasanbayarOverflow/LSTM-music-generator/blob/master/images/one_epoch_data.png" alt="note">

5. Result of 250 epoch after added 10 new midi file
<img src="https://github.com/BaasanbayarOverflow/LSTM-music-generator/blob/master/images/250_epoch_data.png" alt="note">

6. One of music files note sheet used to feed NN
<img src="https://github.com/BaasanbayarOverflow/LSTM-music-generator/blob/master/images/yogore.png" alt="note">

### Features

- [ ] Capability of working with mp3 files
- [ ] Add more music instruments
- [ ] Create more beautiful musics

## Materials used in this project

- [Machine_Learning_is_fun](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471)
- [Write_decent_readme](https://github.com/akashnimare/foco/blob/master/readme.md)
- [Mastering_Machine_Learning_with_Python_in_Six_Steps](https://www.amazon.com/Mastering-Machine-Learning-Python-Steps/dp/1484228650#:~:text=Master%20machine%20learning%20with%20Python,maximum%20of%20six%20steps%20away.)
- [How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
- [LSTM](https://medium.com/@divyanshu132/lstm-and-its-equations-5ee9246d04af)
- [Generate_music_using_Deep_Learning](https://github.com/gauravtheP/Music-Generation-Using-Deep-Learning)
