# StreamingSGD
This repository is an example implementation of Streaming SGD as published here: 

- ArXiv 2019 (full paper): ...
- MIDL 2018 (abstract, proof of concept): https://openreview.net/forum?id=HJ7lIcjoM

**Abstract**
To train deep convolutional neural networks, the input data and the intermediate
activations need to be kept in memory to calculate the gradient descent step. Given
the limited memory available in the current generation accelerator cards, this limits
the maximum dimensions of the input data. We demonstrate a method to train
convolutional neural networks holding only parts of the image in memory while
giving equivalent results.

### See [notebook](https://github.com/DIAGNijmegen/StreamingSGD/blob/master/SSGD%20example.ipynb) for example usage

# Requirements
  - PyTorch 1.0+
  - tqdm
  - numpy
  
# Model compatibility
Should work with all layers that keep local properties of a CNN intact. As such, batch / instance normalization are not supported in the streaming part of the network.
