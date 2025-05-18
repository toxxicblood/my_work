#here were building a Recurrent neural network for music generating using pyorch.
# we traing a model to learn patterns in raw sheet music in ABC notation then use it to generate new music.
import comet_ml
#this is the comet api key for tracking my model development and training runs
COMET_API_KEY = "fux8OQ14wngoM7Cmu6HRqiGQa"

#importing pytorch and other libraries
import torch
import torch.nn as nn
import torch.optim as optim

# import mit intro to deep learing
import mitdeeplearning as mdl

# import all remaining packages
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write

#assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#ensuring the api is correctly set
assert COMET_API_KEY != "", "Please set your COMET_API_KEY in the environment variables."
#downloading the dataset
songs = mdl.lab1.load_training_data()
#check one of the songs
#example_song = songs[0]
#print("\nExample song: \n", example_song)

#we can convert a song in abc to an audio and play it back
#mdl.lab1.play_song(example_song)

#the abc notatin doesnt contain only the notes but also there is metadata.
# this includes the song title, key and tempo
# also realise we need to generate numerical representation of the textual data.

#join all song strings inot one  containing all songs
songs_joined = "\n\n".join(songs)
#find all unique chars in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset.")

