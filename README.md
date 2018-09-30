# Real Time Multi Digit Number using SVHN
This project has a deep convolute neural network (**CNN**) predicting the multi digit number in the images from the real world. I used keras framework using tensroflow-gpu as backend.


## Overview
##### Google Street View House Number Dataset
These dataset is taken from the google street view. So the images of house number is not that focus and dataset is created with the bounding box for each digit.
+ The dataset is around 600,000 digit images from the natural world
+ The mat file is given along with the data consists of all the bounding boxes of the digits from the data.
+ Each digit is divided into ten classes from 1 to 10, where the 1 to 9 represent the digits and 10 represent zero
+ original dataset consists of two type, one with the original image and bounding box for each digit, another is the 32x32 image with the full number as the output.

The link to the original dataset is [here](http://ufldl.stanford.edu/housenumbers/)
##### Preprocessing
I have worked on the orignal image with the bounding box. The data is preprocessed in the **` preprocessing.ipyn `** file. where
+ The images are cropped into smaller size and resized into 32 x 32 grayscale image.
+ The images with the 6 digits (*less than 5 images*) are  removed to avoid data disproportionate. And the output is change into array of five for each image.
+ Each digit is represented in class of 0 to 10 where 0 to 9 represents their value and 10 represent empty value.
 
##### Model

This CNN model can recognize numbers from the real world images.

The architecture is as follows:

1. INPUT [32x32x1]
2. CONV2_1 [32X32X32]
3. CONV2_2 [32X32X32]
4. MAX_POOL_3 [16X16X32] added dropout after pooling
5. CONV2_4 [16X16X64]
6. CONV2_5 [16X16X64]
7. MAX_POOL_6 [8x8x64] added dropout after pooling
8. CONV2_6 [8x8x128]
9. CONV2_7 [8x8x128]
10. CONV2_8 [8x8x128]
11. FLATTEN [8192,1]
12. DENSE_9 [256,1] added dropout after Flatten
13. DENSE_10 [253,1]
14. DENSE_11 [55,1] output layer for size [55,1]
15. LAMBDA_12 [55,1] loss layer added into the model.

The accuracy of the classifier is as follows

Training Accuracy : 98.8%

Test Accuracy : 98.2% 

## Results
This classifer can detect the full number correctly with the accuracy of 88.05% 
## Requirements
1. python 3.5.1
2. keras 2.2
3. Tensorflow 1.9(*better with gpu*)
4. h5py

### Reference
+ [Street_View_House_Numbers_(SVHN)_Dataset](http://ufldl.stanford.edu/housenumbers/)
+ [svhn-multi-digit_by_thomalm](https://github.com/thomalm/svhn-multi-digit)
+ [Keras_custom_loss_layer](https://github.com/keras-team/keras/issues/4781)
