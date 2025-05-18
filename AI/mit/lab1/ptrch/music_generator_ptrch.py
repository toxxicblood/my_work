import torch
import torch.nn as nn
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt

integer = torch.tensor(1234)
decimal = torch.tensor(3.14159265358979323846)
#the above is a 0-d tensor
print (f"`integer` is a {integer.ndim}-d Tensor: {integer}")
print (f"`decimal` is a {decimal.ndim}-d Tensor:{decimal}")

fibonacci = torch.tensor([1, 1, 2, 3, 5, 8])
count_to_100 = torch.tensor(range(100))
#vectors and lists can be used to create 1-d tensors
print(f"`fibonacci` is a {fibonacci.ndim}-d Tensor with shape: {fibonacci.shape}")
print(f"`count_to_100` is a {count_to_100.ndim}-d Tensor with shape: {count_to_100.shape}")

#2-d /higher rank tensors can be created using matrices.
#in image processing/computer vision, we use 4-d tensors
#This is corresponding to the dimensions which are batch size, color channels, image height and width

### Defining higher-order Tensors ###

'''TODO: Define a 2-d Tensor'''
matrix = torch.tensor([[1,2,3,4,4],[3,3,445,56,67]])

assert isinstance(matrix, torch.Tensor), "matrix must be a torch Tensor object"
assert matrix.ndim == 2
print(f"`matrix` is a {matrix.ndim}-d Tensor with shape: {matrix.shape}")
'''TODO: Define a 4-d Tensor.'''
# Use torch.zeros to initialize a 4-d Tensor of zeros with size 10 x 3 x 256 x 256.
#   You can think of this as 10 images where each image is RGB 256 x 256.
images = torch.zeros(size=(10, 3, 256, 256))

assert isinstance(images, torch.Tensor), "images must be a torch Tensor object"
assert images.ndim == 4, "images must have 4 dimensions"
assert images.shape == (10, 3, 256, 256), "images is incorrect shape"
print(f"images is a {images.ndim}-d Tensor with shape: {images.shape}")

# We can also use slicing to access subtensors within a higher rank tensor
row_vector = matrix[1]
col_vector = matrix[:,3]
scalar = matrix[0,4]

print(f"`row_vector`: {row_vector}")
print(f"`col_vecotr`: {col_vector}")
print(f"`scalar`: {scalar}")

#Math operations on tensors

#the best way is to visualixe computations as a graph.
#the graph is made up of tensors that hold the data.
#the edges that connect them contain the math operations

# create nodes in graph and initialise values
a = torch.tensor(15)
b = torch.tensor(61)

#add them
c1 = torch.add(a,b)
c2 = a+b #pytorch overrides the "+" operation so it can act on the tensors
print(f"c1: {c1}")
print (f"c2:{c2}")

#basically the computatin cerates a new tensor
### Defining tensor operations ###
def func1(a,b):
    c = torch.add(a,b)
    d = torch.subtract(b,1)
    e = torch.multiply(c,d)
    return e

def func2(a,b):
    c = a+b
    d = b-1
    e = c*d
    return e

def func3(a,b):
    return (a+b)*(b-1)
#a,b = 1.5,2.5
e_out1 = func1(a,b)
e_out2 = func2(a,b)

print(f"e_out1:{e_out1}")
print(f"e_out2:{e_out2}")
print(f"e_out3:{func3(a,b)}")
print(f"e_out4:{(a+b)*(b-1)}")


## Neural networks in pytorch##
### Defining a dense layer###
# num_ inputs : number of input nodes
#num_outputs: number of output nodes
#x: input to the layer

class OurDenseLayer(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(OurDenseLayer, self).__init__()
        #def ans initialise params: a weight matrix W and bias B
        # note parameter iniialisation is random
        self.W = torch.nn.Parameter(torch.randn(num_inputs, num_outputs))
        self.bias = torch.nn.Parameter(torch.randn(num_outputs))

    def forward(self, x):
        z = torch.matmul(x,self.W)+self.bias
        y = torch.sigmoid(z)
        return y

    def forward1(self,x):
        z = x*self.W + self.bias
        y = torch.sigmoid(z)
        return y

#num_inputs = 2
#num_outputs = 3
layer = OurDenseLayer(num_inputs=2, num_outputs=3)
x_input = torch.tensor([[1,2.]])
y = layer(x_input)

print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")

###defining a neural network using the pytorch sequential API###
#with this we can stack layers like building blocks
#define number of inputs and outputs
N_INPUT_NODES = 2
N_OUTPUT_NODES = 3

### Defining a neural network using the PyTorch Sequential API ###

# define the number of inputs and outputs


# Define the model
'''TODO: Use the Sequential API to define a neural network with a
    single linear (dense!) layer, followed by non-linearity to compute z'''
model = nn.Sequential(
    nn.Linear(N_INPUT_NODES, N_OUTPUT_NODES),  # Linear layer
    nn.Sigmoid()  # Non-linearity
)


# Test model with example input
x_input = torch.tensor([[1., 2.]])
z = model(x_input)
print(f"input shape: {x_input.shape}")
print(f"output shape: {z.shape}")
print(f"output result: {z}")

### Defining a custom model using subclassing ###
class LinearWithSigmioidActivation(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearWithSigmioidActivation, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        linear_output = self.linear(inputs)
        outpu = self.activation(linear_output)
        return outpu

model = LinearWithSigmioidActivation(N_INPUT_NODES, N_OUTPUT_NODES)
x_input = torch.tensor([[1.,2.]])
y = model(x_input)
print(f"input shape: {x_input.shape}")
print(f"output shape: {y.shape}")
print(f"output result: {y}")

### custom behaviour with subclassing nn.module ###
class LinearButSometimesIdentity(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearButSometimesIdentity, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, inputs, isidentity = False):
        if isidentity:
            return inputs
        else:
            return self.linear(inputs)

# Test the IdentityModel
model = LinearButSometimesIdentity(num_inputs=2, num_outputs=3)
x_input = torch.tensor([[1, 2.]])

'''TODO: pass the input into the model and call with and without the input identity option.'''
out_with_linear = (model(x_input, isidentity=False))

out_with_identity = (model(x_input, isidentity=True))

print(f"input: {x_input}")
print("Network linear output: {}; network identity output: {}".format(out_with_linear, out_with_identity))

### Gradient computation ###
#y = x^2
#Example c = 3.0
x = torch.tensor(3.0, requires_grad=True)
y = x**2
y.backward() # compute gradient

dy_dx = x.grad
print("dy_dx of y=x^2 at x=3.0 is:",dy_dx)
assert dy_dx == 6.0, "dy_dx is not correct"


### Function minimisation with autograd and gradint descent ###

# initialise a random value for our initial x
x = torch.randn(1)
print(f"initialising x = {x.item()}")

learning_rate = 1e-2 #learnging rate
history = [] #to store the history of x values
x_f = 4 #target value

# We will run gradient descent for a number of iterations. At each iteration, we compute the loss,
#   compute the derivative of the loss with respect to x, and perform the update.
for i in range(500):
    x = torch.tensor([x], requires_grad=True)
    # Compute the loss
    loss = (x - x_f)**2

    #backpropagate through loss to compute gradient
    loss.backward()

    #update x with gradient descent
    x = x.item() - learning_rate*x.grad

    #record the update
    history.append(x.item())

#plot the evolution of x as we optimise toward x_f!
plt.plot(history)
plt.plot([0,500], [x_f, x_f])
plt.legend(('predicted', 'true'))
plt.xlabel('iteration')
plt.ylabel('x value')
plt.show()