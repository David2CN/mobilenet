# MobileNet

An implementation of the MobileNet architecture, as described in the [paper](https://arxiv.org/pdf/1704.04861v1.pdf). <br>
The implementation is tested on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

The mobilenet architecture is based on depthwise separable convolutions. It involves a two step process:
  1. depthwise convolution
  2. pointwise convolution
  
Depthwise convolution uses a single filter for each input channel and the output is combined with a 1x1 convolution filter (pointwise convolution). It produces the same output shape as a standard convolution layer. <br>
Compared to the standard convolution, depthwise separable convolution has the effect of reducing computation and model size significantly.

<img src="assets/standard_DS_convolutions.jpg">

The depthwise and pointwise convolution layers are each followed by a batchnormalization layer and a relu activation layer.
The full mobilenet architecture is shown in the image below. <br>

<img src="assets/mobilenet.jpg">

