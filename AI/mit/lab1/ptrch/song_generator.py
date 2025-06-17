import os
#loading the api key
#COMET_API_KEY = os.getenv("COMET_API_KEY")
#assert COMET_API_KEY, "Please set your COMET_API_KEY in the environment variables."
COMET_API_KEY = "fux8OQ14wngoM7Cmu6HRqiGQa"
import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mitdeeplearning as mdl
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
assert COMET_API_KEY != "", "please load your api key"

#downloading the dataset
songs = mdl.lab1.load_training_data()
#see if it contains any songs
#join songs
songs_joined = "\n\n".join(songs)

#find all unique chars in the joined string
vocab = sorted(set(songs_joined))

##text nuetical representation
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

##vecorising the songs string
def vectorize_string(string):
    """
    Converts a string to a vector of integers.
    """
    return np.array([char2idx[c] for c in string])

vectorized_songs = vectorize_string(songs_joined)
#print(songs_joined[:100])
#print (vectorised_songs[:100]) # Print the first 1000 characters of the joined songs

###batch defined to create training examples
def get_batch(vectorized_songs, seq_length, batch_size):
    #len o vectorized songs str
    n = vectorized_songs.shape[0] -1
    idx = np.random.choice(n - seq_length, batch_size)

    #input seq list for training bathc
    input_batch = np.array([vectorized_songs[i:i + seq_length] for i in idx])
    #target seq list for training batch
    output_batch = np.array([vectorized_songs[i + 1:i + seq_length + 1] for i in idx])

    #convert hese batches to tensors
    x_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
    y_batch = torch.tensor(output_batch, dtype=torch.long, device=device)

    return x_batch, y_batch

x_batch, y_batch = get_batch(vectorized_songs, seq_length=20, batch_size=5)

"""for i, (input_idx, target_idx) in enumerate(zip(x_batch[0], y_batch[0])):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx.item()])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx.item()])))"""


###defining the neural network
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #add a dropout layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size,num_layers=2, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        #initialize hidden state and cell sstate to zeros
        #return (torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
              #  torch.zeros(1, batch_size, self.lstm.hidden_size, device=device))
        return [torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device)]

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)

        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)

        out = self.fc(out)
        return out if not return_state else (out, state)

#instantiating  the model with default parameters
vocab_size = len(vocab)
embedding_dim = 256
hidden_size = 1024
batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = LSTMModel(vocab_size, embedding_dim, hidden_size).to(device)

#print out the model summary
#print(model)

#its always good ot run a few simple check on the model to see that it runs as expected.
# Test the model with some sample data
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
x = x.to(device)
y = y.to(device)

#pred = model(x)
#print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
#print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

#untrained model predictions

#sampled_indices = torch.multinomial(torch.softmax(pred[0],dim=-1), num_samples=1)
#sampled_indices = sampled_indices.squeeze(-1).cpu().numpy()
#sampled_indices

#decode to txt:
#print("Input: \n", repr("".join(idx2char[x[0].cpu()])))
#print()
#print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
#undtrained model predictioins are noisy and nonsensical

#model training
#loss and training ops
cross_entropy = nn.CrossEntropyLoss()#instantiate function
def compute_loss(labels, logits):
    """
    Inputs:
      labels: (batch_size, sequence_length)
      logits: (batch_size, sequence_length, vocab_size)

    Output:
      loss: scalar cross entropy loss over the batch and sequence length
    """
    #batch labels so theat their shape is (b*L)
    batched_labels = labels.view(-1)

    #batch logits so that their shape is (B*, V)
    batched_logits = logits.view(-1, logits.size(-1))

    #compute cross entropy loss using batched nex chars and predictions
    loss = cross_entropy(batched_logits, batched_labels)
    return loss

#compute los on predictions from untrained model
y.shape #(batch_size, sequence_length)
#pred.shape #(batch_size, sequence_length, vocab_size)
#compute loss using true next chars and predictions several cells above
#example_loss = compute_loss(y, pred)
#print(f"Prediction shape: {pred.shape} # (batch_size, sequence_length, vocab_size)")
#print(f"scalar_loss:      {example_loss.mean().item()}")


###hyperparameter setting and optimisation
vocab_size = len(vocab)

#model parametrs
params = dict(
    num_training_iterations = 300, # increase to traing longer
    batch_size = 16, #experiment btw 1 and 64
    seq_length = 200, # experiment between 50 and 500
    learning_rate = 5e-4, # experiment between 1e-5 and Ie-I
    embedding_dim = 256,
    hidden_size = 1024, #experiment betweeen 1 and 2048
    vocab_size = vocab_size
)

#checkpoiint location
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

#create comet experiment to track our training
#note when hyperparams change i can run te """create_experiment()""" function to initiate
# a new experiment
#all new experiments with the same projjo name live under that projjo
def create_experiment():
    #end any prior experiments
    if 'experiment' in locals():
        experiment.end()
        #locals()['experiment'].end()

    #continue an existing experiment or create a new one
    experiment = comet_ml.ExistingExperiment(
        api_key=COMET_API_KEY,
        previous_experiment  ="",#this is my experiment id
        project_name = "pytorch_music_generator_ RNN"
    )
    # log our hyperparams
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()

    return experiment


### define optimizer and trainingoperation###
#instaniitate a new lstm model for training using hyperparams
model = LSTMModel(
    vocab_size=params['vocab_size'],
    embedding_dim=params['embedding_dim'],
    hidden_size=params['hidden_size']
)
model.to(device)
#load trained model weights
"""model.load_state_dict(torch.load("music_model.pth", map_location=device))"""
#load the best trained model weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

#create a new experiment

experiment = create_experiment()

###precdiction of a generated song
def generate_text(model, start_string, generation_length= 1000):
    #eval step
    input_idx = vectorize_string(start_string)
    input_idx = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(device)

    #initialize hidden state
    state = model.init_hidden(batch_size= 1, device =device)

    #empty string to store our results
    text_generated = []
    tqdm._instances.clear()

    with torch.no_grad():
        for i in tqdm(range(generation_length)):
            #evaluate inputs and generate the next char predictions
            predictions, state = model(input_idx, state, return_state=True)

            #remove batch dimension
            predictions = predictions[:, -1, :]

            #probs = torch.softmax(predictions, dim=-1)
            #input_idx = torch.multinomial(probs, num_samples=1)
            #temperature controls randomness
            temperature = 0.5
            #use a moltinominal dist to sample over probabilities
            input_idx = torch.multinomial(torch.softmax(predictions[-1] / temperature, dim=-1), num_samples=1)
            input_idx = input_idx.unsqueeze(0)  # Ensure shape is [1, 1]
            text_generated.append(idx2char[input_idx.item()])
    return start_string + "".join(text_generated)

#use model and function to generate as ong of len 1000
#abc files start with x and this might be a goot start string
start_string = "X:1\nT:Generated Tune\nM:4/4\nK:C\n"
generated_text = generate_text(model, start_string=start_string, generation_length=1000)

#play back generated music
generated_songs = mdl.lab1.extract_song_snippet(generated_text)
for i, song in enumerate(generated_songs):
    #synthesize waveform
    waveform = mdl.lab1.play_song(song)

    #if its a valid song play it
    if waveform:
        print("Generated song",i )
        ipythondisplay.display(waveform)

        numeric_data = np.frombuffer(waveform.data, dtype = np.int16)
        wav_file_path = f"output_{i}.wav"
        #scaled_data = np.int16(waveform.data / np.max(np.abs(waveform.data)) * 32767)
        #write(wav_file_path, waveform.sample_rate, scaled_data)

        write(wav_file_path, 88200, numeric_data)

        #save song to comet interface
        if os.path.exists(wav_file_path):
            print(f"File {wav_file_path} exists:", os.path.exists(wav_file_path))
            print(f"File size: {os.path.getsize(wav_file_path)} bytes")

            experiment.log_audio(wav_file_path, file_name=f"generated_song_{i}.wav", metadata = {"source":"generated"})
            time.sleep(120)
            experiment.flush()
            #if the above doesnt work try this
            ##experiment.log_audio(wav_file_path, file_name=wav_file_path, step=i, metadata={"type": "generated_song"})

            print(f"Saved generated song {i} to {wav_file_path}")
        else:
            print(f"Failed to save generated song {i} to {wav_file_path}")

experiment.end()
print("Waiting for Comet uploads to finish...")
time.sleep(60)

"""# (Optional) Log inference parameters
experiment.log_parameter("generation_length", 1000)
experiment.log_parameter("temperature", 0.5)

# Generate and log songs as before
start_string = "X:1\nT:Generated Tune\nM:4/4\nK:C\n"
generated_text = generate_text(model, start_string=start_string, generation_length=1000)
generated_songs = mdl.lab1.extract_song_snippet(generated_text)
for i, song in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(song)
    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)
        numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
        wav_file_path = f"output_{i}.wav"
        write(wav_file_path, 88200, numeric_data)
        if os.path.exists(wav_file_path):
            print(f"File {wav_file_path} exists:", os.path.exists(wav_file_path))
            print(f"File size: {os.path.getsize(wav_file_path)} bytes")
            experiment.log_audio(wav_file_path, file_name=f"generated_song_{i}.wav", metadata={"source": "generated"})
            experiment.flush()
            print(f"Saved generated song {i} to {wav_file_path}")
        else:
            print(f"Failed to save generated song {i} to {wav_file_path}")

experiment.end()
print("Waiting for Comet uploads to finish...")
time.sleep(60)"""