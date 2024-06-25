The whole code block consists of the following: Data Acquisition, Data Prepoccessing, Feature Extraction, Training block of 2DCNN/LSTM, Visualisation of the Results. 
The preprocessing part of the data is the augmentation where we add noise to our data and later is connected to module of features extraction which then gets sent to training
module. 
1) Data Acquisition: We acquired a data set named
dont cry corpus from [17]. The data is widely used and has
457 clean audio files, which we will be using for our purpose.
The data is already categorised under this dataset into 5 labels
which are ’bellypain’, ’burping’, ’discomfort’, ’hungry’, and
’tired.
2) Data Prepossessing: This method augments the clean
data we have gotten from data acquisition. The augmentation is
done by adding Gaussian Noise ,Pitch Shifting and Frequency
attenuation. The augmentation rate is defined for all and the
total dataset size is expanded to 1939 files.
3) Feature Extraction: Here we generate spectogram
graphs for all the augmented audio files and also extract
features for all of them. We calculate the short time fourier
transformation of audio files and convert the audio files
amplitudes into decibels and normalize it. We reshape this
array to fit how we further process our data and store the
values in a tuple corresponding to the labels , which is further
down broken into two lists one with normalised stft matrix
is stored in one list and corresponding label of that data into
another
4) Training of Data: Trains the machine learning model
using the stft matrix data and then. We train the data with two
models one is 2D CNN and another is LSTM.The dimensions
of input that does into these layers are differnet as 2D CNN
provides more depth which LSTM does not.
5) Visualisation of Data: We generate plots of the training
process plot loss curves, accuracies, and confusion matrices.
These provide insights into the model’s behavior and
performance.
6) Evaluation Module: We evaluate the trained model’s
performance using methods such as accuracy, precision, recall,
and F1-score. then we conduct testing on separate datasets to
assess the model’s generalization capability.

