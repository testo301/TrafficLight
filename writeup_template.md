# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/histogram1.jpg "Histogram"
[image2]: ./images/signs1.jpg "Sample Images under Respective Labels"
[image3]: ./images/histogram2.jpg "Histogram"
[image4]: ./images/1OriginalImage.jpg "Image prior to Augmentation"
[image5]: ./images/1Augmentation.jpg "Image after Augmentation"


[image6]: ./images/2OriginalImage.jpg "Image prior to Normalization"
[image7]: ./images/2Normalization.jpg "Image after Normalization"


[image8]: ./images/3trainvsvalid.jpg "Train versus Validation Accuracy through Epochs"

[image9]: ./images/Architecture.JPG "Architecture of the network"

[image10]: ./new_signs/0.png "New German Sign"
[image11]: ./new_signs/1.png "New German Sign"
[image12]: ./new_signs/11.png "New German Sign"
[image13]: ./new_signs/12.png "New German Sign"
[image14]: ./new_signs/2.png "New German Sign"
[image15]: ./new_signs/3.png "New German Sign"
[image16]: ./new_signs/4.png "New German Sign"
[image17]: ./new_signs/8.png "New German Sign"

[image18]: ./images/4newsigns_pred.jpg "New German Sign Predictions"
[image19]: ./images/5newsigns_labels.jpg "New German Sign Prediction Distribution"
[image20]: ./images/6newsigns_top5.jpg "New German Sign Prediction Top 5"

[image21]: ./images/FeatureMap1.JPG "Feature Map for Convolutional Layer 1"
[image22]: ./images/FeatureMap2.JPG "Feature Map for Convolutional Layer 2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Link to the github version [project code](https://github.com/testo301/TrafficLight/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows the distribution of the train/validation/test labels.
The distribution is not uniform, with some classes being underrepresented and some dominating the sample.

![alt text][image1]

Sample images for each of the respective classes are presented below, with the corresponding labels

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The first step involved preprocessing the data.

Since some of the classes were underepresented, an augmentation function was created in order to enrich the training data for those labels.

The following example represents a sample image prior and after transformation:
- rotation by a random angle (set as 20 by default)
- offset by a random number of pixels (set as 6 by default)

Image prior to the transformation:

![alt text][image4]

Image after transformation

![alt text][image5]

The augmentation was applied on the entire training dataset, for those labels whose count was under the average count in the sample.

The histogram prior/after augmentation for the training dataset is presented below.

![alt text][image3]

In the next step the images in training/testing/validation datasets are normalized by subtracting and dividing by 128. Prior to that transformation they are converted to the grayscale, which means that the dimensionality changes from (32x32x3) to (32x32x1). This will be reflected in the input to the neural network.

Sample image prior to normalization:

![alt text][image6]

Sample image post normalization:

![alt text][image7]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Input image is a grayscale converted 32x32x1.

Layer 1: Convolutional. The output shape is 28x28x6.

Activation. Rectified linear unit (relu).

Pooling. The output shape is 14x14x6.

Layer 2: Convolutional. The output shape is 10x10x16.

Activation. Rectified linear unit (relu).

Pooling. The output shape is 5x5x16.

Flatten. Flattening the output shape of the final pooling layer such that it's 1D instead of 3D.

Layer 3: Fully Connected. 120 outputs.

Activation. Rectified linear unit (relu).

Dropout. Dropout with probability 0.5.

Layer 4: Fully Connected. This should have 84 outputs.

Activation. Rectified linear unit (relu).

Dropout. Dropout with probability 0.5.

Layer 5: Fully Connected (Logits). 43 outputs.

Output
Returning the output of the second fully connected layer, and each of the layers (conv1,conv2, fc1, fc2).

The architecture of the network is presented below:

![alt text][image9]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For model training the following set of parameters were used
    EPOCHS = 25
    BATCH_SIZE = 128
    dropout_prob = 0.5
    rate = 0.001

For initializing weights with truncated normal distribution:
    mu = 0
    sigma = 0.1
    
Cross entropy was used as the loss function.

Adam optimizer was used for weights tuning.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.954 
* test set accuracy of 0.928

The following plot illustrates the raise of training and validation accuracies through the epochs. There are signs of overfitting at the upper epochs.

![alt text][image8]

The following questions illustrate the choices made:
* What was the first architecture that was tried and why was it chosen?

The first choice was the LeNet framework. Key modifications included the dropouts between the fully connected layers.

* What were some problems with the initial architecture?

The bare LeNet had a tendency to overfitting. Dropout layer aimed at preventing overfit, along with observation of the behaviour of the training versus validation accuracy.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Rectifier linear unit are quite robust as the activation function. In addition, dropout aimed at prevention of overfitting. 

* Which parameters were tuned? How were they adjusted and why?

The number of epochs was selected by benchmarking the execution time and observing the progression of training vs validation accuracies plot across the epochs.

Learning rate was kept at 0.001 as a good choice.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Dropout layer was created between the fully connected layers, as suggested by the following article:
https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16

If a well known architecture was chosen:
* What architecture was chosen?

LeNet is an old architecture, but it seemed to be a proper fit for the training of the traffic sign dataset. 

* Why did you believe it would be relevant to the traffic sign application?

This is justified by the low complexity, low dimensionality and the expectations regarding the output of the network.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
Although some evidence of overfitting is apparent, by examining the plot below, the validation and test accuracies are acceptable.

![alt text][image8]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The additional 8 German traffic signs I found are provided below:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15]
![alt text][image16] ![alt text][image17]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to predict the new signs correctly, yielding 100% accuracy. The predicted labels are presented below.

![alt text][image18]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The images along with corresponding logit label predictions are presented below.

![alt text][image19]

After selecting softmax probabilities for top 5, the model has proved to be very confident in its choices of labels.

![alt text][image20]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Feature map for the first convolutional layer:

![alt text][image21]

Feature map for the second convolutional layer:

![alt text][image22]

The first feature map suggests capturing the parts of the sign outline. The second layer capture specific area but it's not human-readable.