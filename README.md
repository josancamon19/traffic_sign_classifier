## Project: Build a Traffic Sign Recognition Program
Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

# Results Report

## **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/3.sign9.png "Traffic Sign 3"
[image5]: ./test_images/12.sign10.png "Traffic Sign 12"
[image6]: ./test_images/14.sign1.png "Traffic Sign 14"
[image7]: ./test_images/17.sign2.png "Traffic Sign 17"
[image8]: ./test_images/18.sign11.png "Traffic Sign 18"
[image9]: ./examples/results.png "Images and top 5 predictions"

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. refer to the [5th cell]() of the notebook  ...

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

As a first step, I decided to convert the images to grayscale because that requires a model with a lower complexity and in order to 
get a first idea of the model based on the results...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it improve the results and also solve us so many problems and even speeding up convergence ...

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Flatten    |  output shape: batch_size x 800     									|
| Fully connected		| 800 x 512 shape        									|
| RELU					|												|
| Fully connected		| 512 x 128 shape        									|
| RELU					|												|
| Fully connected		| 128 x 43 shape        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy.

My final model results were:
* validation set accuracy of 0.950
* test set accuracy of 0.932

Architecture approach:

The first architecture I tried was LeNet, just changing the input and output size and processing the images to grayscale, I chose it just for test but the results was no great, then I tried adding more conv layers, tried reducing fc layers and even adding batchnorm layers (was so weird don't see any improvement with those) but the results was worse than the first one, finally I decided to tune the initial LeNet, adding, reducing the number of nodes in each layer in order to be able to catch more in each image, changing the depth and in this way was how I finally found a decent model. 

Finally then of the training with the train set, the model with completely new images (test_set) achieved an accuracy of 93.2% and with some images extracted independently from internet achieved an accuracy of 82%, with the good result with new images we verified that our model is a good modedl and performs well.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Some images might was difficult to classify due to the quality of the photos but essentially due to the missing of image augmentation, like scale resize rotate and so on, techniques very very useful. 

Obviously, in a real approach image augmentation is essential.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Priority Road     			| Priority Road 									|
| No Entry					| No Entry											|
| Traffic Signals      		| Traffic Signals					 				|
| General Caution			| General Caution      							|

The model was able to correctly guess 9 of the 11 traffic signs, which gives an accuracy of 82%. This seems not favorably relating to the accuracy on the test set of 93.2%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

For the first image, the model is pretty sure that this is a stop sign (probability of 0.99961), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99961         			| Stop sign   									| 
| .0002     				| Yield										|
| .0000x					| Speed limit (30 km/h)											|
| .0000x	      			| Speed limit (60 km/h)				 				|
| .0000x				    | Bycicles crossing     							|

![alt text][image9]
