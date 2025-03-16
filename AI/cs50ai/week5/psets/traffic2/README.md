# Traffic.py

This is a convolutional neural network that classifies traffic signs into their specific categories.

## Reading the data(read data function)

I am reading the image data as arrays.
The categories are nums with each category being a directory in the data directory.
When data is inputted, i first initiale lists to store the data and labels
then i loop througbh every category in the range provided
ensure each directory exists
Read images in category folders with the cv2 library, after ensuring that the image is readable  i resize it to a fixed size and store the image and corresponding label.
after this, i convert the list of images into a numpy.nd array( a multidimensional array.) then give thi as my return value along with the labels.

I include a label for each image, therefore i label each image in each category in the labels list.

## THe neural network(get_model function)

This function returns a compiled neural network.
I assuem the input shape will be(IMG_WIDTH, IMG_HEIGHT, 3)
The output layer has ```NUM_CATEGORIES``` units.
I can experiment with:
    - Different nos of convolutional and pooling layers
    - Filter sizes and convolutional layers
    - Pool sizes for pooling layers
    - Numers and sizes of hidden layers
    - Dropout

### The beginning

### Keras documentation

This is an api for the tensorflow platforms
THe core components are _layers_ and _models_
__Layer__: simple input/output transformation
__Model__: directed acyclic graph

#### layers

```tf.keras.layers.Layer``` class, fundamental keras abstraction
A ```layer``` encapsulates a state(weights) and some computation defined in ```tf.keras.layers.Layer.call```

- Weights can be trainable or untrainable.
- Layers are recursively composable: this means if i assingn a layer as the attribute of another layer, the outer layer track the weights created by the inner layer
- I can also use layers for data preprocessing which i may have directly inside the model during or after training.

#### models

An object that groups layers together and can be trained on data.

- __Sequential model__: the simplest type of model which is a linear stack of layers.
- The ```tf.keras.Model``` features built in training and evaluation methods:
  - ```tf.keras.Model.fit```: trains model for a fixed number of epochs
  - ```tf.keras.Model.predict```:generate output prediction for inputs
  - ```tf.keras.Model.evaluate```: return loss and metrics values for the model. Configured via ```tf.keras.Model.compile```

#### the sequential model

This model is ideal for a _plain stack of layers_ with each layer having one input tensor and one output tensor.

the model is less appropriate when:
    - the model has multiple inputs or outputs
    - my layers have multiple inputs or outputs
    - i need to  do layer sharing
    - i want non linear topology

##### creating my sequential model

- I want to experiment with different numbers of layers so instead of creating the model with layers outright, i will use the ```model.add``` method to add layers as needed, ensuring the highest efficiency on compute and also the best accuracy.
- To remove layers i will use the ```model.pop()``` method.
- __a seuential model behaves like a list of layers.__

----
__Note__: when a model is instantiated, without an input shape, it isn't built and therefore has no weights. The weights are created when the model first sees smoe input data and can be seen via the ```model.weights``` method or ```model.summary```

----

- To build my model, i will incrementally stack my layers with ```add()``` and print model summaries so i can see which performs best.

- Firstly im gonna go with the layout thet handwriting.py used.

- I satrt by adding my model, a sequential one
- I then add the input layer indicating my expected input with the image height and width. also that im expecting an rgb image.
- After this i add a convolution layer which learns 32 filters with the filter size being a 5*5 kernel and a stride of 2(_moves 2 pixels at a time_)
- I add a maxpool layer with a 2*2 dimenions. I will play with this value to optimum
- I add a hidden layer with dropout. Here ill experiment with differentd dropout and activation functions
- I then implement an output layer with categories and any activation function.

I have implemented the entire thing.
I have varying results. With my first model, i get 100% accuracy but with the class model, i barely etch 50%

I am doing my comparisons with the two programs to see which one performs better. THey are currently at 82% and 71% accuracy respectively. I want to experiment with different layers and optimisations mentioned above.

__hidden layer activation__:

- For the hidden layers, im using __relu__ as my activation function. It avoids vanishing gradients and learns complex patterns while also speeding up trainin by maintaining non-linearity and being computationally simple.
  - Experimenting with ```LeakyReLU```
  - Experimenting with ```elu``` for better weight updates.

After experimenting, imma go with elu, it performs better and faster. results after the change are 90% and 86% accuracy respectively

__output layer activation__:

- For my output layer im using a `softmax` activation because of the multiple classes and works best with ```categorical crossentropy```

__model optimiser__:

- Im currently using adam. I dunno why.
  - imma experiment with ```stochastic Gradient Descent```
    - Takes longer when training(doubles the time taken)
    - I see this as worse, takes more compute and doesnt work. Increased loss
  - Also experiment with ``RMSprop``
    - Immediately takes longer - 18s from 4s
    - On the second go it takes a shorter time.
    - It works kinda. accuracy is 91% and 81%. so imma still stikc it out with adam with an accuracy of
- For this category imma go with adam but imma consider rms

__loss function__:

- Currently using categorical crossentropy which works best with softmax thus remains the same.

__metrics__:

- At the moment im using accuracy, but i can experiment with:
  - ```precision,recall```
  - F1-score `tf.keras.metrics.AUC`
    - IMMA JUST skip to f1 juu its the mean btw precision and recall.
    - It performs worse, ill stick with accuracy.

Now with the experimentation out of the way, ill try adding the layers.
The adding of layers didnt work. I experimented with all variations, didn't do shit, so i regularised the images input while appending them to the image array: `images.append(img/255.0)`
This increased my accuracy from 91 and 87% with the previous settings to 98% and 97%. So far this is the best performance i have seen on both models. I will leave them as they are.
