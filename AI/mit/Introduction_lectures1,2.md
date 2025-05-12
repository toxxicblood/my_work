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