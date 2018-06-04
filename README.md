# StreamingSGD
This repository is an example implementation of Streaming SGD as published here: https://openreview.net/forum?id=HJ7lIcjoM

**Abstract**
To train deep convolutional neural networks, the input data and the intermediate
activations need to be kept in memory to calculate the gradient descent step. Given
the limited memory available in the current generation accelerator cards, this limits
the maximum dimensions of the input data. We demonstrate a method to train
convolutional neural networks holding only parts of the image in memory while
giving equivalent results.

### See [notebook](https://github.com/DIAGNijmegen/StreamingSGD/blob/master/SSGD%20example.ipynb) for example usage

# Requirements
  - PyTorch 0.4
  - tqdm
  
# Model compatibility
  - Layers supported (for now):
    - Conv2d layers (SAME padding now works, [example](https://github.com/DIAGNijmegen/StreamingSGD/blob/master/SSGD%20same%20padding.ipynb))
    - MaxPool2d layer
    - (all layer types are supported in the non-streaming part of the model)
  - Should work with all:
    - Operations performed on tensors before / after convolutional operations, e.g. non-linear activation functions
    - Loss functions (when using mini-batches: if they are averaged over the instances in the batch)
    - Optimizers
  - Currently under development:
    - Batch normalization alternative

# Model requirements
  - `model.forward(x, stop_at_layer, start_at_layer)`: the class will call the forward function of the model with two extra arguments  
  - See the [notebook](https://github.com/DIAGNijmegen/StreamingSGD/blob/master/SSGD%20example.ipynb) for example implementation

# Mini-batch support
  - Start a mini-batch by calling .start_batch() en end by calling .end_batch(), all images processed in between those calls are part of the mini-batch.
  - See [example notebook](https://github.com/DIAGNijmegen/StreamingSGD/blob/develop/SSGD%20example%20(with%20mini-batch).ipynb)
