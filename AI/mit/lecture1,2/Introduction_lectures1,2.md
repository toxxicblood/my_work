# Mit Introduction to Deep Learning

This is the voice clone chat that the intro lecture uses:
![voice clone](image.png)

Intelligence - The ability to process information that can influence some furture decisios.
__AI__ -> Any technique that allows a computer to mimick human behaviour
__ML__ -> The ability of a computer to learn without explicitly programming it but by providing annotated data
__Deep Learning__ -> This is the ability of an algo to extract patterns from data and use the patterns learned to vet furture data.

- Deep learning is basically teaching computers how to do tasks directly from observation/data.

## The perceptron

![The perceptron](image-2.png)
We first multiply our inputs with our weights.
We add in the bias term that is used to shift left and right along our activation function.
The equation can be vectorised as follows: ![vectors](image-3.png)
The non linearity is a function used to qualify the outputs.
Here are some :
![Non linearities](image-4.png)
Activation functions are used to introduce nonlinearities that can separate complex data .
For example:
![EG](image-5.png)
Which can be plotted as follows:
![Gra ph](image-7.png)
The separation happens between 0 and 1

In short here is a simplification:
![Perceptron](image-8.png)

To create a neural network we can add the output nodes.
![Neural net](image-9.png)
The outputs are different because each output node, though with the same input and activation function, has different weights.

## Neural network

### Creating a dense layer from scratch

__Tensorflow__:![Dense](image-10.png)   __Pytorch__:![Dense](image-11.png)

Normally though, all we need to do is call it.
![Call](image-12.png)

### Multi layer neural network(Deep neural networks)

This is a neural network with any number of  __hidden__ layers.
THe layers are hidden because we dont control the values in the layers.
![Hidden layers](image-13.png)
Each neuron in the hidden layer is a perceptron

To build a neural network:
![alt text](image-14.png)

In deep neural networks  the output is formed by a deep combination of linear followed by nonlinear functions.![deep neual net](image-15.png)

THe more complex the task, the deeper the neural network should be.

## Applying neural networks

### Loss

- This is the measure of the cost incurred from incorrect predictions
  ![loss](image-16.png)
  THe smaller the loss the better the neural network.
  __Empirical loss__: measure of total loss over our entire dataset
  ![Empirical loss](image-17.png)
  We want neural networks that minimise loss
  __Binary(Softmax) crossentropy loss__: can be used with models that output a probability between 0 and 1:
  Ite measures the distance between two probability distributions
  ![Crossentroppy](image-18.png)
  __Mean squared error loss__: used with regression models that output continuous real numbers.
  it measures the distance between the predicted number and true number.
  ![mean squared err](image-19.png)

### Training neural netwrks

Ultimately we want to find the network whose weights _achieve the lowest loss_
![nn](image-20.png)
Our aim is loss minimisation.

#### Loss optimisation

The loss function is just a function of our weights
If we only had two weights we can plot them and see which configuration gives the lowest loss:
![loss](image-21.png)
In the above landscape we want to find the lowest point(which weight configuration give the lowest loss)
Process:

    Randomly pick an initial starting position$(W_0, W_1)$
        Compute the gradient(which way is up locally)![gradient](image-22.png)
        Take a small step int the opposite direction of gradient.
The above algorithm is known as __Gradient descent__
![gradient](image-24.png)

The process of computing the gradient is called __Back propagation__

#### Back Propagation

How does a small change in one weight affect the final loss?
![backprop](image-25.png)
We can write out the derivative and use the chain rule to decompose it:
![chain rule](image-26.png)
If we want to do the same thing for W1
![alt text](image-27.png)

We start from te output and keep computing the iterative chain rules till we get to  the beginning.
![alt text](image-28.png)
We repeat this for every weight in our netwrk using gradients from later layers.

- Back propagation is basically an application of the _chain rule_ from _differential calculus_

### Neural netwiorks in practice

#### 1. Optimisation

 in reality training and optimising neural netwirks is difficult and takes alot of compute.
 Here is a plot of some real world data.
 ![loss landcape](image-29.png)

Optimisation through gradient descent is done as follows:
![alt text](image-30.png)

Now we focus on the learning rate:
![rate](image-31.png)

- learning rate dictates how quickly we take the steps and how quickly we learn from our gradients.
__Small learning rates__ = slow convergence and gets stuck in false local minima
__Large learning rates__ = overshoot, unstable learning and divergence
__Stable learning rates__ = smooth convergence and avoids local minima

To best optimize learning rates is by building _adaptive algorithms_ that adapt the learning rate to the landscape
This means learning rate isn't fixed but can be changed according to the following:
    - how large the gradient is
    - how fast learning is happening
    - weights etc

Here are a few adaptive learning rate algorithms:
![rate](image-32.png)

In summary:
![alt text](image-33.png)

### 2. Mini batches

#### Stochastic gradient descent algo

Gradient descent is computed as a function/average of all data points in the data set which can be very computationally intensive. (we compute the gradient of all data points not just one.)
![alt text](image-34.png)
Instead we can use __stochastic gd__:
Here instead of computing the gradient over all data points we compute it from one data point.
Though this is way noisier we save on time and compute but we sacrifice accuracy in the process.
![alt text](image-35.png)
The middle ground here is mini-batch gradient descent.
Here we use mini batches to compute our gradient thus eliminating the inefficiency of vanilla gradient descent and the noise/inaccuracy of the stochastic gradient descent.

### Mini batches while training

![alt text](image-36.png)

- More accurate gradient estimation
- Smoother convergence
- Larger learning rates

The relationships between gradients and learning rates is extremely connected.
_With more stable gradients, we can begin to take bigger steps.
With minibatches we can also parallelise traning achieving higher speeds by utilising GPUs_

### Overfitting

In model training we dont want our models to work extremely well only on our training sets but also to generalise to our testing set.
![alt text](image-37.png)
Overfitting is the problem where an algorithm learns the data too well which may lead to it just saving the data to memory rather than learning the patterns.
How well the model perorms on a trainig set is used as a proxy for how well it will work on real time data.

- As a "by the way" your model's complexity should match the complexity of the patterns you want to extract from your data and shouldn't be too complex or too simple.

### Regularisation

This is a technique used to improve our model's generalisation to real world and unseen data.

#### Dropout

![alt text](image-38.png)
This is where during training we randomly set some node's activations to 0
Tipically we drop 50% of each layer
This forces the network not to rely on any 1 or more nodes.

#### Early stopping

![alt text](image-39.png)
This is where we monitor the deviation between our training loss and testing loss and we stop training before we have a chance to overfit.
When the deviation increases we stop training.

## Deep sequence modelling

In sequence modelling we give an agent a past history of an object/ subject and tell it to predict where the thing is going next.
This is mostly done with data that occurs sequentially like speech, text, images and music etc

Sequential data like text can be split up into chunks called tokens.
These can vary in size from single character, words or even phrases being used as tokens to tranin different models.

### Sequence modelling applications

Here instead of binary classificatin which is one to one
We may have :
sentiment analsis -> many to one classification
Image captioning -> one to many
machine translation -> many to many
![alt text](image-40.png)

### Neural networks with recurrence

We can simplify the perceptron into the following:
![alt text](image-41.png)
This forms a neural network with multiple inputs and outputs and hidden layers:
![alt text](image-42.png)
This can be simplified into the following:
![alt text](image-43.png)
Which can be translated vertically:
Thus timesteps can be added for sequential data due to the inherent dependence between each sequence of the data.
We want the networks internal computations at eah individual timestep to prior computations of the previous timesteps.
We want to feed forward this information to each subsequent timestep
This is done as follows:
![alt text](image-44.png)
THe state h is passed onwards from timestep to timestep.

- The output produced is a function not only of the current inputs but also the previous states in the sequence.
- Recurrence can be simplified into the following image:
![alt text](image-45.png)

### Recurrent neural networks(RNNs)

The key idea here is that the internal state $h_t$ is updated at each time step as the sequence is processed
In short, when each token is inputted, the internal state (_weights and activation functions_) is passed forward to the next input in the sequence.
The internal state can be a single neuron or an entire deep neural network passed forward.

__Recurrence relation__: this is a function to process a sequence timestep by timestep.

- we want te internal cell state $h_t$ to be a function of not only the current input $X_t$ but also of the previous cell state $h_t -1$
![alt text](image-46.png)

We can apply the above function recurrently at each timestep

__Pseudo code__:

![alt text](image-47.png)

### RNN state update and output

![alt text](image-48.png)
THe above is simply rolled out is:
![alt text](image-49.png)

- Realise that we reuse the same weight matrices at every timestep

To actually train this network we have to find a way to define a _loss_
Fo RNNs tho we can compute a loss at each timestep and when the sequence is over compute a total loss for the entire process by summing up the individual losses.
![alt text](image-50.png)

![alt text](image-51.png)

But because this has already been implemented in the libraries, we can just call a RNN as a function:
![alt text](image-52.png)

## Sequence modellling

### Sequence modelling:Design criteria

1. Handle _variable length sequences_
2. Track _long term dependencies_
3. _Maintain order_ information
4. _Shape parameters_ across the sequence

### Sequence modelling problem: Next word prediction

This is a was a fundamental problem in learning because neural networks represent data inputted as numbers
![alt text](image-53.png)

### Encoding language

__Embedding__:This is where we take an input and transform it to an index that can nap to a vector of a fixed size.

__Vocabulary__: corpus of possible word encounters of a fixed size
__Indexing__: mapping words to a number
__Embedding__: mappnig an index to a fixed size vector.

__One hot embedding__:this is where the index of a word is used as its vector.
We can also use a neural network layer to _learn an embedding_ of a fixed size that can better represent the word.
![alt text](image-54.png)

With learned embedding, similar words end up in a similar vector space thus capturing the semantic meaning of words.
We understand the meanings of words by the company the word keeps.

### Backpropagation through time(BPTT)

We need to be able to do a backpropagation over all timesteps of our  RNN.

__Back propagation in a neural network(feed forward model)__:
![alt text](image-55.png)

But in RNNs we move in timesteps
THe loss is computed timestep by timestep.
When training then we not only consider the individual loss but also the aggregate loss across all the time steps.
Therefore we backpropagate errors across the individual timesteps.
![alt text](image-57.png)

Here then we can avoid errors that happen late in the network by propagating them to the beginning

__Exploding gradients__:
![alt text](image-58.png)

__Vanishing gradient__:
![alt text](image-59.png)

The above two pose a big threat because it becomes difficult to track long term dependencies
![alt text](image-60.png)

#### Gatimg mechanisms in neurons

The solution crated was using gates to selectively add or remove information within each recurrent unit
![alt text](image-61.png)
Thus this gate system controls hidden state updates.

__Long short term memory(LSTMs) networks__: these rely on gated cells to track info throughout many timesteps

## Example of RNN use: music generating

We input each note in the music and receive an output of the next note:
![alt text](image-62.png)

## limitations of RNNs

1. _THe state_ $h_t$: the state of an RNN is a vector of fixed length thus having a information bottleneck
2. _Timestep/sequential processing_ : this makes them difficult to parallelise and thus improve computation speed.
3. The encoding bottleneck can hinder the long term memory capacity of these networks.

__Sequence modelling goal__:
![alt text](image-63.png)
![alt text](image-64.png)

_Can we eliminate the need for recurrence_:
![alt text](image-65.png)

## Attention models

In attention we focus on the most important parts of an input

1. Identify the important parts to attend to
2. Extract features with high attention.

__Building a self attention model__:

1. Encode positional information
  ![alt text](image-66.png)

2. Extract _query,key value for search_
   ![alt text](image-67.png)

3. COmpute _attention weighting_
   ![alt text](image-68.png)

4. Extract _features with high attention_
   More similarity causes higher attention weights and inverse
   ![alt text](image-69.png)
   We use the attention weighting to extract features.
   ![alt text](image-70.png)

The full algorithm would then look like this:
![alt text](image-72.png)

Attention is the foundation of transformers .
And also we can stack attention heads to form a transformer network