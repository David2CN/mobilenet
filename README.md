# MobileNet

An implementation of the MobileNet architecture, as described in the [paper](https://arxiv.org/pdf/1704.04861v1.pdf). <br>
The mobilenet architecture is based on depthwise separable convolutions. It involves a two step process:
  1. depthwise convolution
  2. pointwise convolution
  
Depthwise convolution uses a single filter for each input channel and the output is combined with a 1x1 convolution filter (pointwise convolution). It produces the same output shape as a standard convolution layer. <br>
Compared to the standard convolution, depthwise separable convolution has the effect of reducing computation and model size significantly.

<img src="assets/standard_DS_convolutions.jpg">

The depthwise and pointwise convolution layers are each followed by a batchnormalization layer and a relu activation layer.
The full mobilenet architecture is shown in the image below. <br>

<img src="assets/mobilenet.jpg">

The implementation was tested on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
After training for 20 epochs, using the RMSProp optimizer with a learning rate of 1e-3, momentum of 0.9 and weight decay of 1e-6.
20% of the training data was used as a validation set. At the end of 20 epochs, the training and validation accuracies were about 93% and 83% respectively. 
<img src="assets/accuracy.jpg">
<img src="assets/loss.jpg">
The model was reloaded with the weights that had the lowest validation loss [epoch 17] and an overall accuracy of 84.62% was achieved on the test set.
The training and validation accuracies at epoch 17 was about 92% and 85% respectively.
