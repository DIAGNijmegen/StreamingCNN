# StreamingCNN
This repository is an example implementation of StreamingCNN as published here: 

- Full paper (in review; arXiv preprint): http://arxiv.org/abs/1911.04432
- MIDL 2018 (abstract, proof of concept): https://openreview.net/forum?id=HJ7lIcjoM, _this work was based on [previous version](https://github.com/DIAGNijmegen/StreamingCNN/tree/befcb63e86d44730b9180a1db81427941e95b653) of the algorithm._

**Abstract**
To train deep convolutional neural networks, the input data and the intermediate
activations need to be kept in memory to calculate the gradient descent step. Given
the limited memory available in the current generation accelerator cards, this limits
the maximum dimensions of the input data. We demonstrate a method to train
convolutional neural networks holding only parts of the image in memory while
giving equivalent results.

#### See [this notebook](https://github.com/DIAGNijmegen/StreamingCNN/blob/master/sCNN%20numerical%20comparison.ipynb) for a numerical comparison between streaming and conventional backpropagation.
#### See [Imagenette example](https://github.com/DIAGNijmegen/StreamingCNN/blob/master/sCNN%20numerical%20comparison.ipynb) for an example comparing losses between streaming and conventional training.

# Requirements
  - PyTorch 1.0+
  - tqdm
  - numpy
  
# Model compatibility
Should work with all layers that keep local properties of a CNN intact. As such, batch / instance normalization are not supported in the streaming part of the network.
