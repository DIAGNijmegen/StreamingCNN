# StreamingCNN
This repository is an example implementation of StreamingCNN as published here: 

- Full paper (in review; arXiv preprint): http://arxiv.org/abs/1911.04432
- MIDL 2018 (abstract, proof of concept): https://openreview.net/forum?id=HJ7lIcjoM, _this work was based on [previous version](https://github.com/DIAGNijmegen/StreamingCNN/tree/befcb63e86d44730b9180a1db81427941e95b653) of the algorithm._

**Abstract**

Due to memory constraints on current hardware, most convolution neural networks (CNN) are trained on sub-megapixel images. For example, most popular datasets in computer vision contain images much less than a megapixel in size (0.09MP for ImageNet and 0.001MP for CIFAR-10). In some domains such as medical imaging, multi-megapixel images are needed to identify the presence of disease accurately. We propose a novel method to directly train convolutional neural networks using any input image size end-to-end. This method exploits the locality of most operations in modern convolutional neural networks by performing the forward and backward pass on smaller tiles of the image. In this work, we show a proof of concept using images of up to 66-megapixels (8192x8192), saving approximately 50GB of memory per image. Using two public challenge datasets, we demonstrate that CNNs can learn to extract relevant information from these large images and benefit from increasing resolution. We improved the area under the receiver-operating characteristic curve from 0.580 (4MP) to 0.706 (66MP) for metastasis detection in breast cancer (CAMELYON17). We also obtained a Spearman correlation metric approaching state-of-the-art performance on the TUPAC16 dataset, from 0.485 (1MP) to 0.570 (16MP).

#### See [this notebook](https://github.com/DIAGNijmegen/StreamingCNN/blob/master/sCNN%20numerical%20comparison.ipynb) for a numerical comparison between streaming and conventional backpropagation.
#### See [Imagenette example](https://github.com/DIAGNijmegen/StreamingCNN/blob/master/Imagenette%20example.ipynb) for an example comparing losses between streaming and conventional training.

# Requirements
  - PyTorch 1.0+
  - tqdm
  - numpy
  
# Model compatibility
Should work with all layers that keep local properties of a CNN intact. As such, batch / instance normalization are not supported in the streaming part of the network.
