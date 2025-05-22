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


#vectorising text: creating a numerical representation of our txt baed dataset
# for this we generate lookup tables, one maps chars to numbers and the other maps nums back to chars.
char2idx = {u: i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)
#lets see how the conversion has gone
"""print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')"""

### now we vectorise the songs string###
#print (char2idx)
def vectorize_string(string):
    """
    Vectorizes a string using the char2idx mapping.
    """
    return np.array([char2idx[c] for c in string], dtype=np.int32)
    array = np.array([])
    for char in string:
        array = np.append(array,[char2idx[char]])

    return array

vectorized_songs = vectorize_string(songs_joined)
#print ('{} -------chars mapped to int----->{}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
#check that vectorisedsongs is a numpy array
assert isinstance(vectorized_songs, np.ndarray), "returned reshult should be a numpy array"


###create training examples and targets ###
###Batch definition to create training examples ###
def get_batch(vectorized_songs, seq_length, batch_size):
    #the length of the vectorized songs string
    n = vectorized_songs.shape[0]-1
    #randomly choose a starting position in the training batch
    idx = np.random.choice(n-seq_length, batch_size)

    #list of input sequences

    """ # how i did it
    input_batch = vectorized_songs[idx:min(n,(idx+seq_length))]
    #listo of output sequences shifted one place
    output_batch = vectorized_songs[idx+1:min(n+1, (idx+seq_length+1))]"""

    #how its done
    #list of input sequences
    input_batch = np.array([vectorized_songs[i:i+seq_length]for i in idx])
    #listo of output sequences shifted one place
    output_batch = np.array([vectorized_songs[i+1:i+seq_length+1]for i in idx])

    #convert the input and output batches to tensors
    x_batch = torch.tensor(input_batch,dtype=torch.long)
    y_batch = torch.tensor(output_batch,dtype=torch.long)

    return x_batch, y_batch

#perform some simple tests to make sure the batch function works properly
test_args = (vectorized_songs, 10, 2)
x_batch, y_batch = get_batch(*test_args)
assert x_batch.shape == (2,10), "x_batch shape is incorrect"
assert y_batch.shape == (2, 10), "y_batch shape is incorrect"
print("batch function works correctly")

#for each off the vectors , each index is processed invividually at a single timestep.
"""
So, for the input at time step 0, the model receives the index for the first
character in the sequence, and tries to predict the index of the next character.
At the next timestep, it does the same thing, but the RNN considers the
information from the previous step, i.e., its updated state, in addition
to the current input."""

#eg:
x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

for i, (input_idx, target_idx) in enumerate(zip(x_batch[0],y_batch[0])):
    print("Step{:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx.item()])))
    print(" expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx.item()])))


### Defining the model ###
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        #defining the layers
        #layer 1 is the embedding layer to transform indices to dense vectors of a
        #fixed size
        self.embedding = nn.Embedding(vocab_size, embedding_dim )

        #layer2: lstm with hidden size
        self.lstm = nn.LSTM(embedding_dim, hidden_size, dropout=0.5, bidirectional=False, batch_first=True)

        #layer3: linear (dense) layer that transforms the lstm output into vocab size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        #initializing a hidden state and cell state with zeros
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
               torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)

        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)

        out = self.fc(out)
        return out if not return_state else(out, state)


###model instantiation: build simple model with default hyperparameters.
vocab_size = len(vocab)
embedding_dim = 256
hidden_size = 1024
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(vocab_size, embedding_dim, hidden_size).to(device)

#print out a summary of model
print(model)

### testing out the rnn
#check the layers in the model, the output shape fo each layer , the batch size, and dimensionality  of the output
#test model with some simple data
x,y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
x= x.to(device)
y = y.to(device)

pred = model(x)
print("Input shape:      ",x.shape, "#(batch_size, sequence_length)")
print("prediction shape: ", pred.shape,"#(batch_size, sequence_length, vocab_size)")

#untrained model predictions are random
#lets check the predictions
sampled_indices = torch.multinomial(torch.softmax(pred[0],dim=-1), num_samples=1)
sampled_indices = sampled_indices.squeeze(-1).cpu().numpy()
sampled_indices

print("Input: \n", repr("".join(idx2char[x[0].cpu()])))
print()
print("Next char predictions: \n", repr("".join(idx2char[sampled_indices])))

###model training : loss and training operations

"""At this point, we can think of our next character prediction
 problem as a standard classification problem. Given the previo
 us state of the RNN, as well as the input at a given time step
 , we want to predict the class of the next character -- that i
 s, to actually predict the next character."""
##Defining losss function"

cross_entropy = nn.CrossEntropyLoss()