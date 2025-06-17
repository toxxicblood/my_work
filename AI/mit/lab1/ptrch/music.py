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
from google.colab import files
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
        self.lstm = nn.LSTM(embedding_dim, hidden_size,num_layers=2, batch_first=True, dropout=0.2)
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
#y.shape #(batch_size, sequence_length)
#pred.shape #(batch_size, sequence_length, vocab_size)
#compute loss using true next chars and predictions several cells above
#example_loss = compute_loss(y, pred)
#print(f"Prediction shape: {pred.shape} # (batch_size, sequence_length, vocab_size)")
#print(f"scalar_loss:      {example_loss.mean().item()}")


###hyperparameter setting and optimisation
vocab_size = len(vocab)

#model parametrs
params = dict(
    num_training_iterations = 800, # increase to traing longer
    batch_size = 64, #experiment btw 1 and 64
    seq_length = 450, # experiment between 50 and 500
    learning_rate = 5e-3, # experiment between 1e-5 and Ie-I
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

    #initiate the comet experient
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
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

### intantiate teh optimizer with learing rate
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

def train_step(x,y):
    #set model mode to train
    model.train()
    #zero gradients for every step
    optimizer.zero_grad()

    #forward pass
    y_hat = model(x)

    #compute loss
    loss = compute_loss(y, y_hat)

    #backward pass
    #complete gradient computation and update step.
    #Step 1: backpropageate loss
    #step 2: update model params using optimizer
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # gradient clipping
    optimizer.step()
    return loss

####Training
history = []
plotter = mdl.util.PeriodicPlotter(sec = 2, xlabel='Itarations', ylabel='loss')
experiment = create_experiment()

best_loss = float('inf')
best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")

if hasattr(tqdm, '_instances'): tqdm._instances.clear() #clear if it exists
for iter in tqdm(range(params["num_training_iterations"])):
    #grab a batch and propagate it through the network.
    x_batch, y_batch = get_batch(vectorized_songs, params['seq_length'], params['batch_size'])

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    #take a train step
    loss = train_step(x_batch, y_batch)

    #log the loss to the comet interface
    if iter % 10 == 0:
        experiment.log_metric("loss", loss.item(), step=iter)

    #update progress bar and visualise within notebook
    history.append(loss.item())
    plotter.plot(history)

    #save regular model checkpoint
    if iter % 100 ==0:
        torch.save(model.state_dict(), checkpoint_prefix)

    #save best checkpoint if loss improves
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"New best model saved at iteration {iter} with loss {best_loss:.4f}")

#save the final trained model
torch.save(model.state_dict(), checkpoint_prefix)
#save model state dict:
files.download(best_ckpt_path)

experiment.flush()

