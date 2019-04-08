# **Writeup: Traffic Sign Recognition**

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

[exploratoryStat]: ./writeup_data/exploratory_stat.png "Exploratory visualization"
[randomImages]: ./writeup_data/25_random_image.png "25 Random Images"
[cvtGray]: ./writeup_data/cvt_to_grayscale.png "Convert to grayscale"
[Training]: ./writeup_data/Training.png "Training"
[web]: ./writeup_data/web.png "Web images"
[own]: ./writeup_data/own.png "Own images"
[webpredictions]: ./writeup_data/web_predictions.png "Web predictions"
[ownpredictions]: ./writeup_data/own_predictions.png "Own predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

I put the writeup into the repository's REAMDE.md folder you are currently reading. You can find my sourcecode as Jupyter notebook [here](./Traffic_Sign_Classifier.ipynb) or as HTML document [here](./Traffic_Sign_Classifier.html).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

To get an idea of the training set as base for setting up the neural network, I evaluated some statistics about it:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630

The original images of the German traffic signs vary, but Udacity provided a ready to use preprocessed set with the shape:

* (32, 32, 3) --> i.e. a 32x32px RGB image with a depth of 3 channels

All the images are pre-classified. There are **43** unique classes/labels in total.

#### 2. Include an exploratory visualization of the dataset.

Udacity provides a csv file (signs/[signnames.csv](./signs/signnames.csv)) with a full text representation of the labels:

* 00 - 'Speed limit (20km/h)'
* 01 - 'Speed limit (30km/h)'
* 02 - 'Speed limit (50km/h)'
* 03 - 'Speed limit (60km/h)'
* 04 - 'Speed limit (70km/h)'
* 05 - 'Speed limit (80km/h)'
* 06 - 'End of speed limit (80km/h)'
* 07 - 'Speed limit (100km/h)'
* 08 - 'Speed limit (120km/h)'
* 09 - 'No passing'
* 10 - 'No passing for vehicles over 3.5 metric tons'
* 11 - 'Right-of-way at the next intersection'
* 12 - 'Priority road'
* 13 - 'Yield'
* 14 - 'Stop'
* 15 - 'No vehicles'
* 16 - 'Vehicles over 3.5 metric tons prohibited'
* 17 - 'No entry'
* 18 - 'General caution'
* 19 - 'Dangerous curve to the left'
* 20 - 'Dangerous curve to the right'
* 21 - 'Double curve'
* 22 - 'Bumpy road'
* 23 - 'Slippery road'
* 24 - 'Road narrows on the right'
* 25 - 'Road work'
* 26 - 'Traffic signals'
* 27 - 'Pedestrians'
* 28 - 'Children crossing'
* 29 - 'Bicycles crossing'
* 30 - 'Beware of ice/snow'
* 31 - 'Wild animals crossing'
* 32 - 'End of all speed and passing limits'
* 33 - 'Turn right ahead'
* 34 - 'Turn left ahead'
* 35 - 'Ahead only'
* 36 - 'Go straight or right'
* 37 - 'Go straight or left'
* 38 - 'Keep right'
* 39 - 'Keep left'
* 40 - 'Roundabout mandatory'
* 41 - 'End of no passing'
* 42 - 'End of no passing by vehicles over 3.5 metric tons'

To have a good chance to train a neural network, it's not only important to have a sufficient amount of training data, but also to have all the relevant classes/labels quite equally distributen throughout the data. So for the training set, I visualized the total images per class in the following bar chart:

![Exploratory Visualization][exploratoryStat]

It can be seen that the classes are not fully equally distributed in the training set, but each class has at least around 200 images. Later in the training I will address this using a batch approach for training and shuffling the images randomly.

To see how the images in the training set look, I also selected 25 images randomly from the set an plotted them together with their class and class name:

![25 random images from the training set][randomImages]

Looking at the images it's obvious that they differ very much in the image quality. Lighting conditions are different, as well as the image angle, the colors, sometimes images are not fully visible, they are in bad condition or there are stickers on them. With a traditional traffic sign classification approach, it may be quite difficult to find an algorithm that covers all the cases. Let's see how machine learning is doing on the task.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried to find useful steps for the preprocessing phase while doing an interative training approach on the neural net. Im summary, I used only a conversion to grayscale and a normalization of the images.

So as the first step, I convert the images to grayscale:

```python
  # grayscale image
  print(image.shape)
  prep_image = np.sum(image/3, axis = 3, keepdims=True)
```

![Color conversion to grayscale][cvtGray]

I did this because reducing the number of channels improved the learning of the net and the accuracy for predicting the images in the validation set from 85.9% to 88%.

To further improve the prediction, I normalize the images so the values are around the mean zero. As a result, all the pixels in the image have a value from -1 to 1:

```python
  # normalize image
  prep_image = (prep_image - 128)/128

  if viz == True:
    print(prep_image)

```
[[[[-0.56770833]
   [-0.58333333]
   [-0.5703125 ]
   ...
   [-0.515625  ]
   [-0.54427083]
   [-0.578125  ]]

  [[-0.54947917]
   [-0.59375   ]
   [-0.57291667]
   ...
   [-0.5703125 ]
   [-0.52083333]
   [-0.515625  ]]]]

Using normalization not only speed up the training process, but also improved accuray to about 90% at max.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As a starting point for working on this project I used the LeNet architecture like suggested in the lessons. While in mentioned paper for traffic sign recognition by Sermanet and LeCun there was a different architecture that lead to much better classification, I sticked with the LeNet, by adapting the layers to the following:

First - feeding an RGB image directly into the net:

| Layer         		|     Description	        				          	|
|:-----------------:|:-------------------------------------------:|
| Input         		| 32x32x3 RGB image   							          |
| Convolution 5x5  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU activation		|												                      |
| Max pooling 2x2  	| 2x2 stride, outputs 14x14x6 				        |
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 5x5x16   |
| RELU activation		|        									                    |
| Max pooling	2x2		| 2x2 stride, outputs 5x5x16       						|
| Fully connected   | outputs 400 flattened                       |
| Fully connected   | outputs 120                                 |
| RELU activation		|        									                    |
| Fully connected   | outputs 84                                  |
| RELU activation		|        									                    |
| Fully connected   | outputs 43 - final logits, one hot          |

After training and iteratively improving the net, the architecture looks the following:


| Layer         		|     Description	        				          	|
|:-----------------:|:-------------------------------------------:|
| Input         		| 32x32x1 normalized grayscale image          |
| Convolution 5x5  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU activation		|												                      |
| Max pooling 2x2  	| 2x2 stride, outputs 14x14x6 				        |
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 5x5x16   |
| RELU activation		|        									                    |
| Max pooling	2x2		| 2x2 stride, outputs 5x5x16       						|
| Fully connected   | outputs 400 flattened                       |
| Dropout           | Keep Rate 50% (keep_prob = 0.5)             |
| Fully connected   | outputs 120                                 |
| RELU activation		|        									                    |
| Dropout           | Keep Rate 50% (keep_prob = 0.5)             |
| Fully connected   | outputs 84                                  |
| RELU activation		|        									                    |
| Dropout           | Keep Rate 50% (keep_prob = 0.5)             |
| Fully connected   | outputs 43 - final logits, one hot          |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model in an iterative approach, which steps are described in the next chapter. In general, I did run the net for a defined number of epochs while using batches each of 128 images size. The batches have been taken from a in each epoch shuffled training set. The batches are then preprocessed and feed into the LeNet. This is the training operation:

```python
  # learning rate as hyperparameter
  rate = 0.0005

  logits, conv1, conv2 = LeNet(x)

  # calculate the cross entropy via softmax
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

  # calculate loss and optimize (minimize loss)
  loss_operation = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate=rate)
  training_operation = optimizer.minimize(loss_operation)
```

A learning rate of 0.0005 turned out to be a good value for accuracy and training time. For the training itself I used the softmax cross entropy with logiths method, calculated the loss with the reduce_mean function, and used the AdamOptimizer on it with the defined learning rate. Goal was to minimize the loss.

So the final hyperparameters are:

* Number of epochs: 150
* Size of batches: 128
* Learning rate: 0.0005
* Mean mu for the layers: 0
* Standard deviation sigma for the layers: 0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

After training, my final model reached the following results:

* training set accuracy of 99.5%
* validation set accuracy of 97.6%
* test set accuracy of 95.0%

For the training itself I used an iterative approach:

Training_01.p:
I started with the LeNet architecture from the lessions as suggested, changed the input size to 3 channels and the output to 43 logits as there are 43 classes. I used 10 Epochs, a batch size of 128 and a learning rate of 0.001 - like in the lesson. This in the first run produced an accuracy on the validation set of 85.9%

Training_02.p:
Next, I converted the images to grayscale to reduce the channel count and therefore complexity. The LeNet input convolution layer was changed back to 1 channel. While there are no color information that may improve the detection are now missing, the net performed better with an accuracy of 88%.

Training_03.p:
Next, I normalized the images in the preprocessing as described above. This further improved the accuracy to about 90%.

From step two on, the LeNet startet to overfit, so I was going further into trying the following methods:

Training_04.p:
I increaded the epochs with different steps, and finally landed at 150 epochs to be a good value, improving the accuracy to 92.3% - but still overfitting.

Training_05.p:
To prevent from overfitting, I added a dropout after the first fully connected layer, with a keep probability of 0.5, meaning a dropout of 50%. This improved further to an accuracy of 95.6%

Training_06.p:
Added another dropout after the second fully connected layer, same keep_prob as above: Accuracy: 96.9%, overfitting got better.

Training_07.p:
Added another dropout after the third fully connected layer, same keep_prob as above: Accuracy: 96.7%, not really overfitting, but also not so much improvement.

Training_08.p:
Finally, I decreased the learning rate a few steps, where 0.0005 turned out to be a good final value. Accuracy: 95%.

The image below shows the validation accuracy during the training for the above steps:

![Training image][Training]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose five images for German traffic signs from Google image search:

![Web images][web]

The images are clearly visible, while there may be the following dificulties:

* Image one has a blue/gray background
* Image two as a part of an additional sign at the bottom and - from the image source - some watermark artifacts above it
* Image three is not aligned towards the camera, has an additional sign and is rotated
* Image four is one traffic sign in front of another, with an additional part of a sign at the top
* Image five is clearly visible, but the sign is not facing towars the camera.

And...as I live in Germany and did records some videos with my dashcam for other car projects, I took five additional images from there:

![Own images][own]

The images are not that clearly visible and partly of poor quality - as it's reality with cheaper cameras in cars:

* Image one is very dark and blurred
* Image two is very bright but with low contrast, something covering the image (windshield reflection) and another sign at the top
* Image three is very bringt, low contrast, very blurred.
* Image four is very dark with low contrast and an additional - in the set not known - sign telling the times when the sign needs to be followed. The speed sign is badly visible.
* Image five is quite clear to see, but blurred and with a troubled background.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction for the web images:

| Image			        |     Prediction	        					|
|:-----------------:|:---------------------------------:|
| Right of way   		| Right of way     									|
| No Vehicles       | No vehicles                       |
| Speed limit 30km/h| Speed limit 30 km/h               |
| Priority Road     | Priority road                     |
| Yield             | Yield                             |

The model was able to correctly predict 5 of 5 traffic signes, which gives an accuracy fo 100%
This is better than Training, Validation and Test set. Reasons for that are the very clear visible images and the low count of images leading to 20% accuracy per correct predicted image. In general one can assume that it corresponds to the measured accuracy of the bigger sets.

This are the results of the prediction for the images of my video footage:

| Image			        |     Prediction	        					|
|:-----------------:|:---------------------------------:|
| Priority Road  		| Priority Road    									|
| No Passing        | No Passing                        |
| Speed limit 80km/h| Speed limit 80 km/h               |
| Speed limit 50km/h| Dangerous curve to the right      |
| Pedestrians       | Pedestrians                       |

The model was able to correctly predict 4 of 5 traffic signes, which gives an accuracy fo 80%
I'm really impressed that the model does that good on the really bad quality images :-)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following image shows the probabilities for the 5 web images:

![Probability Web Images][webpredictions]

As you can see, the model is quite sure on 4 of 5 images. Nearly 100% for the correct ones. With the 30 km/h speed limit sign, it's 60% sure, followed by Speed Limit 50 km/h (20%), Speed limit 80 km/h (10%), End of Speed limit and Speed limit 60 km/h (each 5%).


The following image shows the probabilities for the 5 own video footage images:

![Probability Own Images][ownpredictions]

As the images are of poor quality, it can be noticed that the model is not so sure like in the clear web images. Two images are very confidently predicted. The speed limit 80 km/h is still quite confident, but not that much like the web images. The wrong classified Speed Limit 50 km/h is not even under the top 5 softmax probabilites - I need to analyse futher why this isn't the case - I think it's the additional traffic sign under the main sign. The Pedestrians traffic sign has just about 10% more confidence than the Right of way, so it is close to missclassified.

#### 4. Further improvements
There are some points I want to try next, but not for the submission of the project:

* work further on preprocessing (adaptive thresholds, maybe morphing images to make the signs "look" at the camera, increasing contrast, blurring to remove noise, maybe other colorspaces like HLS, HSV, YUV)
* show the steps while running the neural net to find out why the the image from the own video footage was missclassified
* play with other net architecture like the ones in the Traffic Sign Recognition with Multi-Scale CNNs paper from Sermanet and LeCun
