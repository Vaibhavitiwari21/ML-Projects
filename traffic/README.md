# CS50AI - Traffic

## Task:

Write an AI to identify which traffic sign appears in a photograph, using a tensorflow convolutional neural network.

## Background:

As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, you’ll use TensorFlow to build a neural network to classify road signs based on an image of those signs. To do so, you’ll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, we’ll use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

## Specification:

Complete the implementation of load_data and get_model in traffic.py.

* The load_data function should accept as an argument data_dir, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
  * You may assume that data_dir will contain one directory named after each category, numbered 0 through NUM_CATEGORIES - 1. Inside each category directory will be some number of image files.
  * Use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width IMG_WIDTH and height IMG_HEIGHT.
  * The function should return a tuple (images, labels). images should be a list of all of the images in the data set, where each image is represented as a numpy.ndarray of the appropriate size. labels should be a list of integers, representing the category number for each of the corresponding images in the images list.
  * Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows. Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.
* The get_model function should return a compiled neural network model.
  * You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).
  * The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.
  * The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
    * different numbers of convolutional and pooling layers
    * different numbers and sizes of filters for convolutional layers
    * different pool sizes for pooling layers
    * different numbers and sizes of hidden layers
    * dropout

Ultimately, much of this project is about exploring documentation and investigating different options in cv2 and tensorflow and seeing what results you get when you try them!

## Model Experimentation Process:

To build this model, I started with very simple models (models with small 'capacity' i.e. small number of learnable parameters), and then gradually added in more layers, increasing the complexity/capacity of the model. Each model was trained against the training data, and then evaluated using the testing data, each set of data randomly selected using Scikit-learn train_test_split (test size = 40%). I could then compare the accuracy of each model on the training set and the testing set. An ideal model would have high and similar accuracy on both the training and testing data sets.

Where a model has a higher loss on the testing data than the training data, this may suggest that the model is overfitting the training data, and so not generalising well onto the test data. When overfitting is severe, a model may be highly accurate (low loss) on the training data but have very poor accuracy (high loss) on the test data. Strategies to reduce overfitting of a model include reducing the capacity (complexity) of the model, adding 'dropout' to layers of the model, or adding weight regularization (penalizing large weights) [1](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting). However, while a simple model may reduce the risk of overfitting the training data, a model with insufficient capacity may suffer from higher loss for both the training and testing data. The capacity of the model must be tweaked to get the best results without overfitting.

There are many different model parameters that can be specified and tuned, e.g.:
* Different numbers of convolutional and pooling layers (learn features, and reduce image size/complexity)
* Different numbers and sizes of filters for convolution layers (the number of kernel matrices to train on and the size of the matrices)
* Different pool sizes for pooling layers (bigger pool size will reduce image size more)
* Different numbers and sizes of hidden layers (model complexity/capacity)
* Additional parameters for the model layers such as dropout, weight regularization, activation functions.
* Other model settings such as the optimizer algorithm, loss function and metric used to monitor the training and testing steps
* etc....!


### Overall Result

One of the best models found during the testing was:

```python
model = tf.keras.models.Sequential([

    # Add 2 sequential 64 filter, 3x3 Convolutional Layers Followed by 2x2 Pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten layers
    tf.keras.layers.Flatten(),

    # Add A Dense Hidden layer with 512 units and 50% dropout
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add Dense Output layer with 43 output units
    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
```

This model had a training and testing accuracy of 99% , and the training and testing loss were similar during trainig runs. The model appears to fit the training data well without overfitting, and generalises well to the testing data.


## Usage:

Requires Python(3) and the python package installer pip(3) to run.

First install requirements:

$pip(3) install -r requirements.txt

Download the GTSRB dataset from https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip

Run the training and testing script:

$python3 traffic.py data_directory [model_name_to_save_model.h5]

## Acknowledgements:

Data provided by [J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=dataset#Acknowledgements)















