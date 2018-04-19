---
layout: post
title: My First Try on NN-network 
date: 2016-05-28 15:46
comments: true
external-url:
categories: Integrated-Circuits-Design
---

# Install Tensorflow on Windows

## Step 1 Installation of CUDA ##

##### What is CUDA, and why do we use it?
　　CUDA is short for Compute Unified Device, and it is a production of NVIDIA corporation that aims to solve the complicated computing problems with GPU within a parallel computing architecture. Developers can process programming with C, C++ or FORTRAN under a standard, mature environment (CUDA environment) to control GPU to solve problems.

##### Installation Procedures
  
1. If your computer is equipped with a NVIDIA graphics card that is not too *old*, it is almost sure for running CUDA. To double check whether your GPU satisfies the CUDA running condition, visit this site [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus "https://developer.nvidia.com/cuda-gpus").
2. Download CUDA Toolkit from NVIDIA official website (see:[https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit "https://developer.nvidia.com/cuda-toolkit")). A reference choice is as follows: 
3. Install the CUDA as instructions.

## Step 2 Installation of CUDNN ##

##### What is CUDNN?
　　CUDNN is a computing package provided by NVIDIA CUDA Toolkit to speed up the computation of convolutional neural network by converting common computation to GPU-friendly one.

##### Installation Procedures
 

1. You can visit the NVIDIA official website to freely download the latest edition of th cuDNN computing package after filling some basic information required, or you can directly search package through search engine and download it to local computer.
2. Install it as instructions.

## Step 3 Installation of Anaconda and TensorFlow package ##

##### Q: What is Anaconda, and why do we use it?

　　Anaconda is an integrated Python environment equipped with Python main programme, IDE, IPython and other third-party packages. And conda is used as a attached tool to manage packages as well as programming environments. You can directly run the conda command in command lines for conda have been defaultly added to system environment varibies during the Anaconda installation process.

##### Installation Procedures

1. Add *CDUA bin* and *NVIDIA Computing Toolkit* to system path.
2. Download the installation package of Anaconda from [https://www.continuum.io/downloads](https://www.continuum.io/downloads "https://www.continuum.io/downloads"). If the downloading process is too slow, you can also download the mirror file from domestic mirror ware, for example: [http://mirrors.ustc.edu.cn/](http://mirrors.ustc.edu.cn/ "http://mirrors.ustc.edu.cn/") .
3. After the Anaconda installation, open the Anaconda Navigator to add a new environment in local computer, note that to choose a python 3.5 version (because some new features are not supported in python 3.6 ).
4. Use the Anaconda to install TensorFlow package. Open Anaconda Prompt and type in *anaconda search -t conda tensorflow* command to check the tensorflow avaliable for current system. Then use command *anaconda show dhirschfeld/tensorflow+'version'* to download and install the package.

##### Well down! Enjoy the convenience from TensorFlow by open Jupyter Notebook.


# Implementation of Neural Network Based on MNIST (Basic level)

## Fundamental Principles

### Logistic Regression

　　Logistic regression is a method helping you implement binary classification by outputing a specific probility of being "1" after mathematical operations. Two steps are needed to finish a single logistic regression: 

- The first step can be thought to combine all input factors together, you need two   parameters - vector W and b, both of which have same dimension as inputs and after doing the operation "W^T*X+b" you get a new variable denoted as "z" that reflects  compositive influence of all inputs.  

- The second step is to apply an activation function to new earned variable "z". The most common activation function is Sigmoid function whose expression is "1/(1+e^(-z))".Two main reasons of applying activation are by doing so you can get a probility within the range from 0 to 1, and making deeper neural network have more complexity. 

### Loss Function and Cost Function

　　It is essential to make judagement of how well your predictation goes by defining Loss Function to a single example and Cost Function that can be thought to be a combined Loss Function to a dataset with more than more examples.

　　In logistic regression, we have a stereotype defined Loss Function as "L(y,y_hat)=-[y*log(y_hat)+(1-y)*log(1-y_hat)]". The smaller L is, the more presice your prediction is. Similarly, we define cost function as "J(W,b)=(1/m)*Sigma(1,m)(y(i)*log(y_hat(i))+(1-y(i))*log(1-y_hat(i)))". W and b here are two vectors with same dimension as dataset example.

### Logistic Regression Gradient Descent in Common Programming Mindset

　　Logisic regression gradient descent is a method to find the minimum target function(Cost Function) value. The simplest representation of regression gradient decent pseudocode is as follows:

    Repeat{
	 w := w - a*(d(J(w,b))/dw)
	 b := b - b*(d(J(w,b))/db)
	}

　　	From the above pseudocode we know that the essence of regression gradient descent method is to constantly refresh the parameters so that the target function (J(w,b)) can have steepest drop till the minimus value is found(or the gradient of target function remains so small that can be seen as 0).

　　To a dataset with many examples, the logistic regression gradient descent pseudocode is as follows(let number of examples be 2):

    J = 0, dw1 = 0, dw2 = 0, db = 0

	For i = 1 to m

		z(i) = w^T*x(i) + b
		a(i) = sigmoid(z(i))
		J += -[y(i)log(a(i)) + (1-y(i))log(1-a(i))]
		
		dz(i) = a(i) - y(i)
		dw1(i) += x1(i)dz(i)  % dw1 refers to d(J(w1,w2,b))/dw1
		dw2(i) += x2(i)dz(i)  % dw2 refers to d(J(w1,w2,b))/dw2
		db += dz(i)

	J/=m, dw1/=m, dw2/=m, db/=m
	w1 := w1 - a*(d(J(w1,w2,b))/dw1)
	w2 := w2 - a*(d(J(w1,w2,b))/dw2)
	b := b - b*(d(J(w1,w2,b))/db)

### Vectorization for speeding up

　　We can see there are two explict "for" loop in above code, however the "for" loop runs so slowly in computer that it makes impossible to implement deep neural network. Therefore, we use vectorization method to avoid explict "for" loop in our code to speed up the neural network training.

　　By using packages provided with python such as numpy we can process vectorized calculation easily, for example, the "np.dot(A,B)" function in numpy calculate the two vectors' multiplcation without using explict "for loop" that makes our code run more efficiently.

　　The vectorized version code of the logistic regression gradient descent is shown as follows:

    for iter in range(number_of_examples):
		Z = np.dot(W,T,X) + b
		A = sigmoid(Z)
		dZ = A - Y
		dW = (1/m)*X*dZ^T
		db = (1/m)*np.sum(dZ)
		W := W - a*dW
	 	b := b - a*db

　　There are some notes to do vectorized programming in python, you'd better to do the vector definition by clearly pointing the dimension of the vector. For example, "a = np.random.randn(5,1)" instead of difining like "a = np.random.rand(5)" because there is a special wired data structure called "rank 1 array" in python that may causes really confusing bugs if you don't clearly clarify the dimension of the vector.

### Fundamentals of Neural Network

　　After figuring out the basic knowledge of logistic regression and its gradient descent method it will be easy to know the bassic principle of neural network because we can see neural work as iteration of logistic regression for many times through different layers of the network. Take vectorized logistic regression as reference, we get four baisc step to implement a neural network as follows(let the layer of the network be 2):

    z[1] = W[1]*x + b[1]
	a[1] = activation(z[i])
	z[2] = W[2]*a[i] + b[2]
	a[2] = activation(z[2])

　　Note that the symbol we use here is slightly different from that in logistic regression. For example, in "z[1]", "1" refer to the 1st layer of the neural network, commonly it refers to the hidden layer but not the input layer as we usually think, and the "activation" means activation function, its impact is similar to "Sigmoid" function as we mentioned before, but we have to utilize more efficient activation function in neural network, the most frequently used activation function is ReLU function. There are many other forms of activation functions in different application of neural network.

　　The reason we use activation function is to make neural network have more complexities so that the network can get ideal result. Note that unless being used in output layer in some "rare" occasions such as binary classification, we do not use "Sigmoid" Function, and we usually set ReLU function as default.

## Implementation of Neural Network based on MNSIT

### What is MNSIT

　　The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

　　To import MNSIT database in python with this command:

    import tensorflow.examples.tutorials.mnist.input_data as input_data
	mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

### Define the Training Cost Function and Implement Gradient Descent Algorithm

　　We use cross-entropy to be cost function( but not the Cost Function we implement in logistic regression ). You can think cross-entropy as a degree of confusion, the less confusion a system is, the better the prediction we get. Then we have to use gradient descent descent algorithm to minimize the target function so that we get the optimized parameters. TensorFlow can do the optimiztation very essily. The relevant codes are as follows:

    y_ = tf.placeholder("float",[None,10])
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

### Iteration and Result Judgement

　　To my notebook, it;s impossible to do the training with all range of the dataset, therefore we use "batch" function to randomly choose 100 data to do the training. To test how precise our model is we use "tf.argmax" function to compare the predictation and the true value and then get the mean number of the boolen array we get to represent the accuracy our neural network. There are some other notices, for example, we have to initialize all parameters first. The code is as follows:

    init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for i in range(1000):
		batch_xs, batch_ys =mnist.train.next_batch(100)
    	sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

### Result of Our First Neural Network

　　The full version of the project is as follows:

    import tensorflow as tf
	import tensorflow.examples.tutorials.mnist.input_data as input_data
	mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

	x = tf.placeholder("float", [None, 784])
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x,W) + b)

	y_ = tf.placeholder("float",[None,10])
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	for i in range(1000):
    	batch_xs, batch_ys =mnist.train.next_batch(100)
    	sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

	print (sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

　　Run the model many times we find that the accuracy of the neural network is around 91%, it is not a satisfying result because our neural network only have 2 layers - single hidden layer and a output layer. We will improve and perfect our model in next tutorials.



----------
Copyright clarification: The passage is created by Zhikai Huang, any copying or propagation behaviour without author's permission is forbidden.  
