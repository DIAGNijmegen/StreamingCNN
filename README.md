### Develop branch, refactored and easier to understand code. The implementation here is almost ready. 

# StreamingSGD
This repository is an example implementation of Streaming SGD as published here: https://openreview.net/forum?id=HJ7lIcjoM

**Abstract**
To train deep convolutional neural networks, the input data and the intermediate
activations need to be kept in memory to calculate the gradient descent step. Given
the limited memory available in the current generation accelerator cards, this limits
the maximum dimensions of the input data. We demonstrate a method to train
convolutional neural networks holding only parts of the image in memory while
giving equivalent results.

### See [notebook](https://github.com/DIAGNijmegen/StreamingSGD/blob/develop/SSGD%20example.ipynb) for example usage

# Requirements
  - PyTorch 0.4
  - tqdm
  
# Model compatibility
  - Layers supported:
    - Conv2d layers (without padding)
    - MaxPool2d layer
    - (all layer types are supported in the non-streaming part of the model)
  - Should work with all:
    - Operations performed on tensors before / after convolutional operations, e.g. non-linear activation functions
    - Loss functions (when using mini-batches: if they are averaged over the instances in the batch)
    - Optimizers
  - Currently under development:
    - Conv2d with padding
    
# Model requirements
  - `model.forward(x, stop_at_layer, start_at_layer)`: add start/stop module names to forward function.
  - (See the notebook for example implementation)

# Mini-batch support
- Start a mini-batch by calling `.start_batch()` and to end by calling `.end_batch()`, all images processed in between those calls are part of the mini-batch.
