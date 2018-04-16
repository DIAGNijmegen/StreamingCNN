# StreamingSGD
This repository is an example implementation of Streaming SGD as published here: https://openreview.net/forum?id=HJ7lIcjoM

### See [notebook](https://github.com/DIAGNijmegen/StreamingSGD/blob/master/SSGD%20example.ipynb) for example usage

# Requirements
  - PyTorch 0.3.1
  - tqdm
  
# Model compatibility
  - Layers supported:
    - Conv2d layers (without padding)
    - MaxPool2d layer
    - (all layer types are supported in the non-streaming part of the model)
  - Should work with all:
    - Operations performed on tensors before / after convolutional operations, e.g. non-linear activation functions
    - Loss functions
    - Optimizers
    
# Model requirements
  - `model.layers`: list of layers in order of execution
  - `model.gradients`: list of gradients of the inputs to convolutional operations
  - `model.output`: list of outputs of the convolutional operations
  - `model.forward(x, stop_index=-1, start_index=0, detach=False)`: add start/stop-index to forward function and ability to detach leaves from the graph. 
  - (See the notebook for example implementation)

