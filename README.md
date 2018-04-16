# StreamingSGD
To train deep convolutional neural networks, the input data and the intermediate activations need to be kept in memory to calculate a gradient descent step. Given the limited memory available in current generation accelerator cards, this limits the maximum dimensions of the input data. Here we demonstrate a method to train convolutional neural networks holding only parts of the image in memory while giving equivalent results.

### See [notebook](https://github.com/DIAGNijmegen/StreamingSGD/blob/master/SSGD%20example.ipynb) for example usage

### See OpenReview.net for paper: https://openreview.net/forum?id=HJ7lIcjoM
