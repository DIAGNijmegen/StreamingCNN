"""
Author: Hans Pinckaers
MIT License
"""
import copy
import math
import os
from dataclasses import dataclass
from itertools import repeat
from typing import NamedTuple, Union, List

import numpy as np
import torch
import torch.autograd
import torch.backends
import torch.nn.functional

from torch._six import container_abcs
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.utils.cpp_extension import load

from tqdm import tqdm


# from torch.nn.grad import _grad_input_padding

if '1.6' in torch.__version__: # type:ignore
    def forward_amp_decorator(func): 
        return torch.cuda.amp.custom_fwd(func)  # type:ignore
    def backward_amp_decorator(func): 
        return torch.cuda.amp.custom_bwd(func)  # type:ignore
    from torch.cuda.amp import autocast
else:
    def forward_amp_decorator(func): 
        return func
    def backward_amp_decorator(func):
        return func

# Load and compile cpp code to call cudnn conv2d backward function
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "cpp_functions.cpp")
cpp_functions = load(name="cpp_functions", sources=[filename], verbose=False)

# inspired by torch/nn/modules/utils.py
def _ntuple(n):
    def parse(x, default=0):
        if isinstance(x, container_abcs.Iterable):
            if len(x) == n: 
                return x
            elif len(x) == n-1: 
                return tuple([default, *x])
            else: 
                return tuple(repeat(x[0], n))
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)

# Utility named tuples, makes code more readable
class Sides(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int

@dataclass
class Box:
    y: int
    height: int
    x: int
    width: int
    sides: Union[Sides, None]

class IOShape(NamedTuple):
    batch: int
    channels: int
    height: int
    width: int

@dataclass
class Lost:
    top: int
    left: int
    bottom: int
    right: int

    def __str__(self):
        return 'Lost(top:%2.1f, left:%2.1f, bottom:%2.1f, right:%2.1f)' \
            % (self.top, self.left, self.bottom, self.right)

class StreamingConv2dF(torch.autograd.Function):
    @staticmethod
    @forward_amp_decorator
    def forward(ctx, inpt, weight, bias, stride, padding, dilation, groups, grad_lost, seen_indices, output_stride, input_loc):
        ctx.save_for_backward(inpt, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.grad_lost = grad_lost
        ctx.seen_indices = seen_indices
        ctx.output_stride = output_stride
        ctx.input_loc = input_loc
        return torch.nn.functional.conv2d(inpt, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    @backward_amp_decorator
    def backward(ctx, grad_output):
        inpt, weight, bias = ctx.saved_variables
        grad = grad_weight = grad_bias = None

        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation 
        groups = ctx.groups 
        sides = ctx.input_loc.sides  # Type: Sides
        seen_indices = ctx.seen_indices
        grad_lost = ctx.grad_lost  # Type: Lost
        output_stride = ctx.output_stride
        grad_bias = None
        kernel_size = weight.shape[-1]

        if ctx.needs_input_grad[0]:
            # TODO: performance improvements possible by only backpropping valid input
            # grad_input_padding = _grad_input_padding(grad_output, inpt.shape, stride, padding, (weight.shape[2], weight.shape[3]))  
            # TODO: use this!?
            grad_in = cpp_functions.backward_input(inpt.shape, grad_output, weight.to(inpt.dtype), padding, 
                                                   stride, dilation, groups, 
                                                   torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        else:
            grad_in = None

        grad = grad_output

        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0

        valid_grad = grad[:, :, lost_top:grad.shape[H_DIM] - lost_bottom,
                          lost_left:grad.shape[W_DIM] - lost_right]

        stride, kernel_size, padding = _triple(stride), _triple(kernel_size), _triple(padding)

        output_stride = output_stride * torch.tensor(stride)
        input_loc = ctx.input_loc

        # Move the location according to how many pixels have been trimmed
        # this will be the location of the valid gradient of this layer in relation
        # to the actual gradient in a normal backpass
        data_loc_y = int(input_loc.y // output_stride[1]) + lost_top
        data_loc_x = int(input_loc.x // output_stride[2]) + lost_left

        data_loc = Box(data_loc_y, 0,
                       data_loc_x, 0,
                       input_loc.sides)

        # Calculate which part of the gradient is 'new'
        old_value_indices = seen_indices
        new_output_box, updated_total_indices = StreamingCNN._new_value_indices(valid_grad.shape,
                                                                                data_loc,
                                                                                old_value_indices)

        # Update inplace
        seen_indices.y = updated_total_indices.y
        seen_indices.height = updated_total_indices.height
        seen_indices.x = updated_total_indices.x
        seen_indices.width = updated_total_indices.width
        seen_indices.sides = updated_total_indices.sides

        if new_output_box.height > 0 and new_output_box.width > 0:
            relevant_grad = valid_grad[:, :,
                                       new_output_box.y:new_output_box.y + new_output_box.height,
                                       new_output_box.x:new_output_box.x + new_output_box.width]

            input_y = (new_output_box.y + lost_top) * stride[1]
            input_x = (new_output_box.x + lost_left) * stride[2]

            # Accounting for padding:
            # the kernel locations are relative to the padded input, inpt[0] is not padded
            # this means that the corresponding input of the grad_loc is module.padding shifted to the left
            # we account for this:
            input_y -= padding[1]
            input_x -= padding[2]
            input_x = max(0, input_x)
            input_y = max(0, input_y)

            relevant_input_height = relevant_grad.shape[H_DIM] * stride[1] + (kernel_size[1] - 1)
            relevant_input_width = relevant_grad.shape[W_DIM] * stride[2] + (kernel_size[2] - 1)

            relevant_input = inpt[:, :,
                                  input_y:input_y + relevant_input_height,
                                  input_x:input_x + relevant_input_width]

            # If layer has padding we need to pad based on if the current tile
            # is at the sides of the input.
            if (padding[0] > 0 or padding[1] > 0 or padding[2] > 0) and \
                    (sides.top or sides.left or sides.right or sides.bottom):
                # The size of the tile should remain equal.
                crop_bottom = padding[1] if sides.top else 0
                crop_right = padding[2] if sides.left else 0
                relevant_input = inpt[:, :,
                                      input_y:input_y + relevant_input_height - crop_bottom,
                                      input_x:input_x + relevant_input_width - crop_right]

                relevant_input = torch.nn.functional.pad(relevant_input, [padding[2] if sides.left else 0,
                                                                          padding[2] if sides.right else 0,
                                                                          padding[1] if sides.top else 0,
                                                                          padding[1] if sides.bottom else 0])

            # Calculate the kernel gradients with the new unseen gradient values
            relevant_grad = relevant_grad.contiguous()

            grad_weight = cpp_functions.backward(weight.shape,
                                                 relevant_grad.to(weight.dtype),
                                                 relevant_input.to(weight.dtype),
                                                 (0, 0),  # padding
                                                 stride[1:3], dilation, groups,
                                                 torch.backends.cudnn.benchmark,  # benchmark
                                                 torch.backends.cudnn.deterministic)  # deterministic

            if bias is not None:
                grad_bias = relevant_grad[0].sum((1, 2))

            del relevant_input
            del relevant_grad
        else:
            # if self.verbose and not hasattr(self, '_inefficient_tile_shape_warning'):
            # print("Warning: no new gradient values found. Tile size could be too small.")
            # self._inefficient_tile_shape_warning = True
            grad_weight = torch.zeros_like(weight)
            if bias is None: grad_bias = None
            else: grad_bias = torch.zeros_like(bias)

        if bias is not None:
            return grad_in, grad_weight, grad_bias, None, None, None, None, None, None, None, None, 
        else:
            return grad_in, grad_weight, None, None, None, None, None, None, None, None, None, 

conv2d = StreamingConv2dF.apply  # type:ignore

class StreamingConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(StreamingConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
        self.grad_lost = Lost(0, 0, 0, 0)
        self.tile_output_box = Box(0, 0, 0, 0, None)
        self.reset()

    def reset(self):
        self.seen_indices = Box(0, 0, 0, 0, None)
        self.input_loc = Box(0, 0, 0, 0, None)

    def forward(self, input):       
        return conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                      self.grad_lost, self.seen_indices, self.output_stride, self.input_loc)

B_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3

class StreamingCNN(object):
    '''Initialize Streaming CNN helper class. After initialization use the
    forward() and backward() function of this class to stream.
    Pseudocode example:

    ```python
    sCNN = StreamingCNN(stream_layers, tile_shape=(1, 3, 600, 600))
    str_output = sCNN.forward(image)
    final_output = final_layers(str_output)
    loss = criterion(final_output, labels)
    loss.backward()
    sCNN.backward(image, str_output.grad)
    ```

    Hooks are used to perform streaming, to use the stream_layers without
    streaming you can disable StreamingCNN with the disable() function.
    Subsequently, enable() enables it again. Streaming gets enabled by default
    after initialization.
    '''
    def __init__(self, stream_module, tile_shape, verbose=False, deterministic=False,
                 saliency=False, gather_gradients=False, replace_non_linearity=True, 
                 eps=1e-5, copy_to_gpu=True, dtype=None, statistics_on_cpu=False,
                 normalize_on_gpu=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 state_dict=None):
        '''
        Parameters:
            stream_module (torch.nn.Module): module containing the to be streamed layers
            tile_shape (tuple, NCHW): size of the to be streamed tiles
            verbose (bool): will log various debugging relevant information (default is False)
            deterministic (bool): whether to use the deterministic algorithms for cudnn
            saliency (bool): will gather the gradients of the input image (saliency map)
            gather_gradients (bool): will gather the gradients of the feature maps
            eps (float): epsilon error to compare floating values
        '''
        global H_DIM, W_DIM
        self.stream_module = stream_module
        self.verbose = verbose
        self.deterministic = deterministic
        self.eps = eps
        self.device = next(stream_module.parameters()).device
        self.dtype = next(stream_module.parameters()).dtype
        if dtype is not None: self.dtype = dtype
        self.tile_shape = tile_shape
        self.gather_input_gradient = saliency
        self.gather_gradient = gather_gradients
        self.replace_non_linearity = replace_non_linearity
        self.copy_to_gpu = copy_to_gpu
        self.statistics_on_cpu = statistics_on_cpu

        self.mean = torch.tensor(mean).cuda()[:, None, None]
        self.std = torch.tensor(std).cuda()[:, None, None]
        self.should_normalize = normalize_on_gpu

        self._tile_output_shape = None
        self._module_stats = {}
        self._backward_seen_indices = {}
        self._saved_tensors = {}
        self._current_tile_input_loc = None
        self._hooks = []

        if state_dict is None:
            self._configure()
        else:
            self.load_state_dict(state_dict)

    def _configure(self):
        if self.replace_non_linearity: self.convert_modules_model(self.stream_module)
        self.convert_modules_model(self.stream_module, from_mod=torch.nn.BatchNorm2d, to_mod=torch.nn.Sequential)

        # Save current model and cudnn flags, since we need to change them and restore later
        state_dict = self._save_parameters()
        old_deterministic_flag, old_benchmark_flag = self._set_cudnn_flags_to_determistic()
        self._reset_parameters_to_constant()

        # Add hooks to each layer to gather statistics
        self._add_hooks_for_statistics()

        # We need to temporary store statistics per layer to keep track of the
        # total output stride at each layer
        self._stats_per_grad_fn = {}

        # TODO; temp hack for tile sizes too big on gpu, 
        # we need float32 precision
        if self.statistics_on_cpu:
            self.stream_module = self.stream_module.cpu()
            self.device = torch.device('cpu')  # type:ignore

        # Create all-ones tile
        tile = torch.ones(self.tile_shape, dtype=self.dtype, requires_grad=True, device=self.device)

        self._gather_forward_statistics(tile)
        if self.verbose: print('')
        self._gather_backward_statistics(tile)

        # TODO; temp hack for tile sizes too big on gpu, 
        if self.statistics_on_cpu:
            self.stream_module = self.stream_module.cuda()
            self.device = torch.device('cuda')  # type:ignore

        # Remove all hooks and add hooks for correcting gradients
        # during streaming
        self._remove_hooks()
        self._add_hooks_for_streaming()
        self._restore_parameters(state_dict)
        self._convert_modules_for_streaming(self.stream_module)
        if self.replace_non_linearity: self.convert_modules_model(self.stream_module, back=True)

        # Remove temporary data
        self._saved_tensors = {}
        del self._stats_per_grad_fn

        # Zero the gradients
        for param in self.stream_module.parameters():
            if param.grad is not None: param.grad.data.zero_()

        self._set_cudnn_flags(old_deterministic_flag, old_benchmark_flag)
        del state_dict


    def _gather_backward_statistics(self, tile):
        # Forward pass with grads enabled
        torch.set_grad_enabled(True)
        output = self.stream_module(tile)

        # Gather backward statistics
        self._tile_output_shape = output.shape

        gradient = torch.zeros(*output.shape, dtype=self.dtype, device=self.device)
        gradient[:, :,
                 self.tile_output_lost.top:output.shape[H_DIM] - self.tile_output_lost.bottom,
                 self.tile_output_lost.left:output.shape[W_DIM] - self.tile_output_lost.right] = 1
        output.backward(gradient=gradient)

        # Calculate the output stride of the whole stream_module
        p_stats = self._prev_stats(output)
        if p_stats: self.output_stride = p_stats['output_stride'] * torch.tensor(p_stats['stride'])
        else: self.output_stride = torch.tensor([1, 1, 1])

        self.tile_gradient_lost = self._non_max_border_amount(tile.grad)

        if self.verbose: 
            print('\n', 'Input gradient lost', self.tile_gradient_lost)

    def _gather_forward_statistics(self, tile):
        torch.set_grad_enabled(False)
        output = self.stream_module(tile)
        self.tile_output_lost = self._non_max_border_amount(output)
        if self.verbose: print('\n', 'Output lost', self.tile_output_lost)

    def convert_modules_model(self, module, from_mod=torch.nn.ReLU6, to_mod=torch.nn.ReLU, back=False):
        mod = module
        if not back and isinstance(module, from_mod):
            mod = to_mod()
            # mod.previous_mod = module
        if back and isinstance(module, to_mod):
            mod = module.previous_mod
        for name, child in module.named_children():
            mod.add_module(name, self.convert_modules_model(child, from_mod, to_mod))
        del module
        return mod

    def _convert_modules_for_streaming(self, module):
        mod = module
        if isinstance(module, torch.nn.Conv2d):
            if module in self._module_stats:
                mod = StreamingConv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None)
                mod = mod.to(module.weight.device)
                mod = mod.to(module.weight.dtype)

                mod.weight.requires_grad = module.weight.requires_grad
                if module.bias is not None:
                    mod.bias.requires_grad = module.bias.requires_grad

                mod.load_state_dict(module.state_dict())  # copy params
                mod.grad_lost = self._module_stats[module]['grad_lost']
                mod.output_stride = self._module_stats[module]['output_stride']
                self._module_stats[mod] = self._module_stats[module]
                del self._module_stats[module]
        for name, child in module.named_children():
            mod.add_module(name, self._convert_modules_for_streaming(child))
        del module
        return mod

    def _reset_converted_modules(self, module):
        mod = module
        if isinstance(module, StreamingConv2d):
            mod = torch.nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, module.bias is not None)
            mod = mod.to(module.weight.device)
            mod = mod.to(module.weight.dtype)

            mod.weight.requires_grad = module.weight.requires_grad
            if module.bias is not None:
                mod.bias.requires_grad = module.bias.requires_grad

            mod.load_state_dict(module.state_dict())  # copy params
            self._module_stats[mod] = self._module_stats[module]
            del self._module_stats[module]
        for name, child in module.named_children():
            mod.add_module(name, self._reset_converted_modules(child))
        del module
        return mod

    def _reset_parameters_to_constant(self):
        for mod in self.stream_module.modules():
            if isinstance(mod, (torch.nn.Conv2d)):
                # to counter loating precision errors, we assign 1 to the weights and
                # normalize the output after the conv.
                torch.nn.init.constant_(mod.weight, 1)
                if mod.bias is not None:
                    torch.nn.init.constant_(mod.bias, 0)

        for m in self.stream_module.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.eval()

    def _set_cudnn_flags(self, deterministic_flag, benchmark_flag):
        torch.backends.cudnn.deterministic = deterministic_flag
        torch.backends.cudnn.benchmark = benchmark_flag

    def _set_cudnn_flags_to_determistic(self):
        deterministic_flag = torch.backends.cudnn.deterministic
        benchmark_flag = torch.backends.cudnn.benchmark
        self._set_cudnn_flags(True, False)
        return deterministic_flag, benchmark_flag

    def _save_parameters(self):
        state_dict = self.stream_module.state_dict()
        state_dict = copy.deepcopy(state_dict)
        return state_dict

    def _restore_parameters(self, state_dict):
        self.stream_module.load_state_dict(state_dict)

    def _non_max_border_amount(self, tensor):
        # Sum over the channels, useful for networks that treat certain channels
        # different (e.g., DenseNet)
        if tensor.dim() > 3: tensor = torch.sum(tensor, dim=1)[0]
        tensor = tensor / tensor.max()  # normalize
        tensor = (tensor > tensor.max() * (1-self.eps))
        non_zero = tensor.nonzero()
        top, left = non_zero.min(dim=0)[0]
        # for bottom and right we need to substract -1: correct index 3 is actually the 4th pixel
        bottom, right = torch.tensor([*tensor.size()], dtype=torch.long, device=self.device) - non_zero.max(dim=0)[0] - 1
        return Lost(int(top), int(left), int(bottom), int(right))

    def forward(self, image, result_on_cpu=False):
        """Perform forward pass with streaming.

        Parameters:
            image (torch.Tensor): CHW the image to stream
        """
        # The input image is likely quite small in terms of channels, for
        # performance reasons it is beneficial to copy to the GPU as a whole
        # instead of tile-by-tile.
        image = image
        if self.copy_to_gpu:
            image = image.to(self.device, non_blocking=True)

        tile_width, tile_height = self.tile_shape[W_DIM], self.tile_shape[H_DIM]

        # Size of valid output of a tile
        valid_output_height = (self._tile_output_shape[H_DIM] - self.tile_output_lost.top - self.tile_output_lost.bottom)
        valid_output_width = (self._tile_output_shape[W_DIM] - self.tile_output_lost.left - self.tile_output_lost.right)

        # We will keep track which part of the output of the whole image we
        # already filled with valid values from tile output.
        already_filled = Box(0, 0, 0, 0, None)

        # Calculate size of output that we would get by inferencing the
        # whole image.
        output_height = (image.shape[H_DIM] - self.tile_shape[H_DIM]) // self.output_stride[1] + self._tile_output_shape[H_DIM]
        output_width = (image.shape[W_DIM] - self.tile_shape[W_DIM]) // self.output_stride[2] + self._tile_output_shape[W_DIM]

        if result_on_cpu:
            device = torch.device('cpu')
        else:
            device = self.device
        output = torch.empty((image.shape[0], self._tile_output_shape[1], output_height, output_width), dtype=self.dtype, device=device).fill_(999)

        n_rows = math.ceil(float(output_height) / float(valid_output_height))
        n_cols = math.ceil(float(output_width) / float(valid_output_width))
        
        if image.shape[W_DIM] <= tile_width: n_cols = 1
        if image.shape[H_DIM] <= tile_height: n_rows = 1

        if self.gather_input_gradient:
            self.saliency_map = torch.zeros(image.shape, dtype=self.dtype, device=self.device)

        if self.verbose: print('Number of tiles in forward:', n_rows * n_cols)
        if self.verbose: iterator = tqdm(range(n_rows))
        else: iterator = range(n_rows)

        with torch.no_grad():
            for row in iterator:
                for col in range(n_cols):
                    # Coordinates of the output w.r.t. the output of full image
                    output_y = row * valid_output_height
                    output_x = col * valid_output_width

                    # Check if we are at borders, since we can not create
                    # overlap here and should not crop values.
                    sides_top = True if row == 0 else False
                    sides_left = True if col == 0 else False

                    sides_bottom = True if output_y * self.output_stride[1] + self.tile_shape[H_DIM] >= image.shape[H_DIM] else False
                    sides_right = True if output_x * self.output_stride[2] + self.tile_shape[W_DIM] >= image.shape[W_DIM] else False
                    sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                    # These values are used to crop invalid output values
                    lost = self._get_tile_lost_for_sides(sides)

                    # Since we need to stay at multiples of output stride we
                    # need to keep that into account when we are at the bottom
                    # and right side of the output.
                    if sides_bottom: output_y = (image.shape[H_DIM] - self.tile_shape[H_DIM]) // self.output_stride[1]
                    if sides_right:  output_x = (image.shape[W_DIM] - self.tile_shape[W_DIM]) // self.output_stride[2]

                    output_y = output_y if not sides.top else 0
                    output_x = output_x if not sides.left else 0
                    output_loc = Box(output_y + lost.top, -1, output_x + lost.left, -1, sides)

                    # Coordinates of the input w.r.t. the output of full image
                    tile_y = output_y * self.output_stride[1]
                    tile_x = output_x * self.output_stride[2]

                    # Extract tile and perform forward pass
                    tile = image[:, :,
                                 tile_y:tile_y + tile_height,
                                 tile_x:tile_x + tile_width]

                    # normalize on gpu for speed in dataloader
                    # does this reduce speed significantly?
                    if not self.copy_to_gpu:
                        tile = tile.to(self.device, non_blocking=True)

                    if self.should_normalize: tile = self._normalize_on_gpu(tile)
                    tile_output = self.stream_module(tile)

                    trimmed_output = tile_output[:, :,
                                                 lost.top:tile_output.shape[H_DIM] - lost.bottom,
                                                 lost.left:tile_output.shape[W_DIM] - lost.right]

                    new_output_box, updated_total_indices = self._new_value_indices(trimmed_output.shape, output_loc, already_filled)
                    already_filled = updated_total_indices

                    relevant_output = trimmed_output[:, :,
                                                     new_output_box.y:updated_total_indices.y + new_output_box.height,
                                                     new_output_box.x:new_output_box.x + new_output_box.width]

                    output[:, :, int(updated_total_indices.y):int(updated_total_indices.height), int(updated_total_indices.x - new_output_box.width):int(updated_total_indices.x)] = relevant_output

                    del tile

            assert sides_bottom and sides_right, "It seems like we could not reconstruct all output"  #type:ignore

        # mem management
        del relevant_output  # type:ignore
        del image
        self._saved_tensors = {}

        return output

    def backward(self, image, grad):
        """Perform backward pass with streaming.

        Parameters:
            image (torch.Tensor): the image (expects NCHW) that was used in the forward pass
            grad (torch.Tensor): this should be the gradient of the output of
                the stream_layers.
        """
        # The input image is likely quite small in terms of channels, for
        # performance reasons it is beneficial to copy to the GPU as a whole
        # instead of tile-by-tile.
        image = image
        if self.copy_to_gpu:
            image = image.to(self.device, non_blocking=True)
        grad = grad

        height = image.shape[H_DIM]
        width = image.shape[W_DIM]

        tile_height = self.tile_shape[H_DIM]
        tile_width = self.tile_shape[W_DIM]
        grad_lost = self.tile_gradient_lost

        output_height = self._tile_output_shape[H_DIM]
        output_width = self._tile_output_shape[W_DIM]

        valid_grad_height = (tile_height - grad_lost.top - grad_lost.bottom) // self.output_stride[1]
        valid_grad_height *= self.output_stride[1]
        valid_grad_width = (tile_width - grad_lost.left - grad_lost.right) // self.output_stride[2]
        valid_grad_width *= self.output_stride[2]

        n_rows = math.ceil(float(height - grad_lost.top - grad_lost.bottom) / float(valid_grad_height))
        n_cols = math.ceil(float(width - grad_lost.left - grad_lost.right) / float(valid_grad_width))

        if self.verbose:
            ideal_tile_size = height / float(n_rows) + grad_lost.top + grad_lost.bottom        
            next_ideal_tile_size = height / float(n_rows - 1) + grad_lost.top + grad_lost.bottom        
            print(ideal_tile_size, n_rows*n_cols, next_ideal_tile_size) 

        if image.shape[W_DIM] <= tile_width: n_cols = 1
        if image.shape[H_DIM] <= tile_height: n_rows = 1

        if self.gather_gradient:
            self.gradients = {}
        self._inputs = {}
        self._backward_seen_indices = {}

        if self.verbose: print('Number of tiles in backprop:', n_rows * n_cols)
        if self.verbose: iterator = tqdm(range(n_rows))
        else: iterator = range(n_rows)

        for row in iterator:
            for col in range(n_cols):
                # Since we determine output (gradient) coordinates based on input
                # coordinates. We need to divide by output stride.
                output_y = row * valid_grad_height // self.output_stride[1]
                output_x = col * valid_grad_width // self.output_stride[2]

                sides_top = True if row == 0 else False
                sides_left = True if col == 0 else False

                sides_bottom = True if output_y + output_height >= grad.shape[H_DIM] else False
                sides_right = True if output_x + output_width >= grad.shape[W_DIM] else False
                sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                # We are doing a forward pass
                lost = self._get_tile_lost_for_sides(sides)

                # If the tile is at the bottom or right side of the input image
                # than we need to shift back so that the tile fits (does not go
                # over the border)
                if sides_bottom: output_y = max(grad.shape[H_DIM] - output_height, 0)
                if sides_right: output_x = max(grad.shape[W_DIM] - output_width, 0)

                input_y = output_y * self.output_stride[1]
                input_x = output_x * self.output_stride[2]

                input_loc = Box(input_y, tile_height, input_x, tile_width, sides)

                tile = image[:, :,
                             input_y:input_y + tile_height,
                             input_x:input_x + tile_width]

                gradient = grad[:, :,
                                output_y:output_y + output_height,
                                output_x:output_x + output_width]

                self._saved_tensors = {}

                # Trim output and gradient
                trimmed_grad = gradient[:, :,
                                        lost.top:gradient.shape[H_DIM] - lost.bottom,
                                        lost.left:gradient.shape[W_DIM] - lost.right]

                if not self.copy_to_gpu:
                    tile = tile.to(self.device, non_blocking=True)

                for mod in self.stream_module.modules():
                    if isinstance(mod, StreamingConv2d):
                        mod.input_loc = input_loc

                # normalize on gpu for speed in dataloader
                # does this reduce speed significantly?
                if self.should_normalize: tile = self._normalize_on_gpu(tile)

                if self.dtype == torch.float16:
                    with autocast(): 
                        tile_output = self.stream_module(tile)
                else: 
                    tile_output = self.stream_module(tile)

                del tile # memory management

                trimmed_output = tile_output[:, :,
                         lost.top:tile_output.shape[H_DIM] - lost.bottom,
                         lost.left:tile_output.shape[W_DIM] - lost.right]

                # Do backward pass, fix gradient in hooks
                trimmed_output = trimmed_output.to(self.device, non_blocking=True)

                # Sometimes when training with variable input shapes,
                # the gradient size is a bit too big
                if trimmed_grad.shape[H_DIM] != trimmed_output.shape[H_DIM] or \
                   trimmed_grad.shape[W_DIM] != trimmed_output.shape[W_DIM]:
                    assert image.shape[H_DIM] < self.tile_shape[H_DIM] or \
                        image.shape[W_DIM] < self.tile_shape[W_DIM]
                    trimmed_grad = trimmed_grad[:, :,
                                                0:trimmed_output.shape[H_DIM],
                                                0:trimmed_output.shape[W_DIM]]

                trimmed_output.backward(trimmed_grad)

                # Memory management
                del tile_output
                del trimmed_grad
                del trimmed_output

        # Memory management
        self._saved_tensors = {}
        self._current_tile_input_loc = None

        for mod in self.stream_module.modules():
            if isinstance(mod, StreamingConv2d):
                mod.input_loc = None
                mod.reset()

        assert sides_right and sides_bottom, "It seems like we could not reconstruct all output"  # type:ignore

    def _get_tile_lost_for_sides(self, sides):
        lost_top = self.tile_output_lost.top if not sides.top else 0
        lost_bottom = self.tile_output_lost.bottom if not sides.bottom else 0
        lost_left = self.tile_output_lost.left if not sides.left else 0
        lost_right = self.tile_output_lost.right if not sides.right else 0
        lost = Lost(lost_top, lost_left, lost_bottom, lost_right)
        return lost

    def _normalize_on_gpu(self, tile):
        tile_norm = tile.to(self.dtype)
        del tile
        tile_norm.div_(255)
        tile_norm.sub_(self.mean)
        tile_norm.div_(self.std)
        tile = tile_norm
        return tile

    def disable(self):
        """Disable the streaming hooks"""
        self._remove_hooks()
        self._reset_converted_modules(self.stream_module)

    def enable(self):
        """Enable the streaming hooks"""
        self._remove_hooks()
        self._add_hooks_for_streaming()
        self._convert_modules_for_streaming(self.stream_module)

    def _add_hooks_for_statistics(self):
        def forw_lambda(module, inpt, outpt):
            self._forward_gather_statistics_hook(module, inpt, outpt)

        def back_lambda(module, grad_in, grad_out):
            return self._backward_gather_statistics_hook(module, grad_in, grad_out)

        self._add_hooks(forward_hook=forw_lambda, backward_hook=back_lambda)

    def _add_hooks_for_streaming(self):
        if self.gather_input_gradient:
            def back_lambda(module, grad_in, grad_out):
                return self._backward_saliency_hook(module, grad_in, grad_out)

            for mod in self.stream_module.modules():
                if isinstance(mod, (torch.nn.Conv2d)):
                    if mod.in_channels == 3:
                        back_handle = mod.register_backward_hook(back_lambda)
                        self._hooks.append(back_handle)

    def _add_hooks(self, forward_hook, backward_hook,
                   forward_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d),
                   back_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d)):
        for mod in self.stream_module.modules():
            if isinstance(mod, forward_modules):
                forw_handle = mod.register_forward_hook(forward_hook)
                self._hooks.append(forw_handle)
                if back_modules and isinstance(mod, back_modules):
                    back_handle = mod.register_backward_hook(backward_hook)
                    self._hooks.append(back_handle)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def _forward_gather_statistics_hook(self, module, inpt, output):
        stride, kernel_size, _ = _triple(module.stride), _triple(module.kernel_size), _triple(module.padding)

        if not torch.is_grad_enabled():  # type:ignore
            # Convert strided convolutions/pooling to average pool
            if isinstance(module, (torch.nn.MaxPool2d)) or \
                    (stride[0] > 1 and stride[0] > kernel_size[0]) or \
                    (stride[1] > 1 and stride[1] > kernel_size[1]) or \
                    (stride[2] > 1 and stride[2] > kernel_size[2]):
                # Pytorch documentation is explicitely against changing output in a forward hook
                # However, since we do not really need the graph or gradients to be correct
                # it shouldn't harm.
                if module.padding != 0:
                    padding = module.padding
                    if not isinstance(module.padding, tuple):
                        padding = [module.padding, module.padding]
                    padded_input = torch.nn.functional.pad(inpt[0], [padding[1], padding[1], padding[0], padding[0]])
                else:
                    padded_input = inpt[0]

                new_output = torch.nn.functional.avg_pool2d(padded_input, kernel_size[1:], stride[1:])
                new_output = torch.sum(new_output, dim=1)[0]
                new_output = (new_output > (1-self.eps) * new_output.max())
                new_output = new_output.expand_as(output[0])
                output[0] = new_output.type(self.dtype)

            # Sum all dimensions (useful for DenseNet like networks)
            lost = self._non_max_border_amount(output)

            # Make output between 0-1 again, so the values do not explode
            output.fill_(0)
            output[:,:,lost.top:output[0, 0].shape[0] - lost.bottom,
                     lost.left:output[0, 0].shape[1] - lost.right] = 1

            module_stats = {'lost': lost, 'stride': stride, 'module': module}
            if self.verbose: print(module, "\n", module_stats['lost'])

            self._saved_tensors[module] = inpt
            self._module_stats[module] = module_stats
        else:
            module_stats = self._module_stats[module]

            p_stats = self._prev_stats(output)
            if p_stats: output_stride = p_stats['output_stride'] * torch.tensor(p_stats['stride'])
            else: output_stride = torch.tensor([1, 1, 1])

            module_stats['output_stride'] = output_stride.clone().detach()

            self._stats_per_grad_fn[output.grad_fn] = module_stats
            self._module_stats[module] = module_stats

    def _backward_gather_statistics_hook(self, module, grad_in, grad_out):
        stride, kernel_size, _ = _triple(module.stride), _triple(module.kernel_size), _triple(module.padding)

        if grad_in[0] is not None:
            # We sum over the channels to deal with networks that do different operations
            # on groups of channels
            f_grad = torch.sum(grad_in[0], dim=1)[0]

            if isinstance(module, (torch.nn.MaxPool2d)):
                # MaxPool shifts indices around, which break the calculation to
                # find valid gradient values. To fix this we do an average pool
                # with the same kernel-size and stride and repeat using the stride.
                inpt = self._saved_tensors[module]
                padded_inpt = inpt[0]
                if module.padding != 0:
                    padded_inpt = torch.nn.functional.pad(inpt[0], [module.padding, module.padding,
                                                                    module.padding, module.padding], value=-1)

                new_outpt = torch.nn.functional.avg_pool2d(padded_inpt, kernel_size[1:], stride[1:])[0]
                new_outpt = torch.sum(new_outpt, dim=0)

                f_grad = torch.sum(grad_out[0], dim=1)[0]
                f_grad = f_grad * new_outpt
                f_grad = f_grad.cpu()
                f_grad = np.repeat(f_grad, stride[1], axis=0)
                f_grad = np.repeat(f_grad, stride[2], axis=1)
                grad = np.zeros(grad_in[0].shape[2:])
                grad[:f_grad.shape[0], :f_grad.shape[1]] = f_grad
                f_grad = torch.from_numpy(grad)
                f_grad = f_grad.to(self.device)

            grad_lost = self._non_max_border_amount(grad_out[0])

            if self.verbose: print(module, "\n", grad_lost)
            self._module_stats[module]['grad_lost'] = grad_lost

            valid_grad = (f_grad > (1-self.eps) * f_grad.max())

            # When kernel_size > stride we have some _overlap_ of gradients,
            # this overlap makes extra positions in the input gradient invalid
            if (stride[0] > 1 and kernel_size[0] > stride[0]) or \
                    (stride[1] > 1 and kernel_size[1] > stride[1]) or \
                    (stride[2] > 1 and kernel_size[2] > stride[2]):
                valid_lost = self._non_max_border_amount(f_grad)
                valid_grad.fill_(0)
                overlap_rows = kernel_size[1] - stride[1]
                overlap_cols = kernel_size[2] - stride[2]
                valid_grad[valid_lost.top + overlap_rows:
                           valid_grad.shape[0] - valid_lost.bottom - overlap_rows,
                           valid_lost.left + overlap_cols:
                           valid_grad.shape[1] - valid_lost.right - overlap_cols] = 1

            new_grad_in = valid_grad[None].expand(grad_in[0].shape[1], *valid_grad.shape)[None]
            new_grad_in = (new_grad_in.type(self.dtype) * 10 - 1)
            new_grad_in_lost = self._non_max_border_amount(new_grad_in)
            return (new_grad_in, *grad_in[1:])

    def _backward_saliency_hook(self, module: StreamingConv2d, grad_in, grad_out, is_bias=False, change_grad=True):
        stride: List[int] = _triple(module.stride)  # type:ignore

        # Trim gradient of invalid values
        sides = module.input_loc.sides
        grad_lost = module.grad_lost  # type: Lost

        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0
        lost = Lost(lost_top, lost_left, lost_bottom, lost_right)

        # Calculate which part of the gradient is 'new'
        new_output_box = module.tile_output_box
        updated_total_indices = module.seen_indices

        if module.in_channels == 3:
            valid_grad_in = grad_in[0][:, :,
                                       lost.top*stride[0]:grad_in[0].shape[2] - lost.bottom*stride[0],
                                       lost.left*stride[1]:grad_in[0].shape[3] - lost.right*stride[1]]

            relevant_input_grad = valid_grad_in[:, :,
                                                new_output_box.y*stride[0]:
                                                new_output_box.y*stride[0] + new_output_box.height*stride[0],
                                                new_output_box.x*stride[1]:
                                                new_output_box.x*stride[1] + new_output_box.width*stride[1]]

            self.saliency_map[:, :,
                              updated_total_indices.y * stride[0]:
                              updated_total_indices.height * stride[0],
                              updated_total_indices.x * stride[1] - relevant_input_grad.shape[3]:
                              updated_total_indices.x * stride[1]] = relevant_input_grad.detach().cpu()

            del relevant_input_grad
            del valid_grad_in

    @staticmethod
    def _new_value_indices(data_shape, data_indices, old_value_indices):
        """
        This helper functions assumes we reconstruct feature maps and
        gradients in tiles from top-left to bottom-right. Using current tile
        index and old_value_indices it finds the relative indices of `data`
        which are unique for this tile (not earlier seen in other tiles).
        """
        rel_top, rel_bottom, rel_left, rel_right = 0, 0, 0, 0

        old_values_y = old_value_indices.y
        old_values_x = old_value_indices.x
        old_values_height = old_value_indices.height

        # Check if new row
        if data_indices.x == 0:
            old_values_y = old_values_height
            old_values_height = data_indices.y + data_shape[H_DIM]
            old_values_x = 0

        # Check x-axis:
        # If this gradient is exactly on the border of old_value_indices
        # everything is new.
        if data_indices.x == old_values_x:
            rel_left = 0
            rel_right = data_shape[W_DIM]

        # If data_indices has some overlap with old_value_indices, trim unique
        # indices.
        else:
            assert old_values_x - data_indices.x >= 0, "Misses data in x-axis!"
            rel_left = old_values_x - data_indices.x
            rel_right = data_shape[W_DIM]

        # Check y-axis:
        # Equal to column logic (see above)
        if data_indices.y == old_values_y:
            rel_top = 0
            rel_bottom = data_shape[H_DIM]
        else:
            assert old_values_y - data_indices.y >= 0, "We miss data in y-axis"
            rel_top = old_values_y - data_indices.y
            rel_bottom = data_shape[H_DIM]

        # Update old-value-indices
        old_values_x += (rel_right - rel_left)

        assert rel_top >= 0, f"We miss data in y-axis before: {data_indices}"
        assert rel_left >= 0, f"We miss data in x-axis before: {data_indices}"

        new_value_indices = Box(rel_top, rel_bottom - rel_top, rel_left, rel_right - rel_left, None)
        old_value_indices = Box(int(old_values_y), int(old_values_height), int(old_values_x), 0, None)

        return new_value_indices, old_value_indices

    def _prev_stats(self, tensor):
        prev = tensor.grad_fn
        prev_stats = None
        while True:
            if prev in self._stats_per_grad_fn:
                prev_stats = self._stats_per_grad_fn[prev]
                break
            if hasattr(prev, 'next_functions') and len(prev.next_functions) > 0:
                prev = prev.next_functions[0][0]
            else:
                break
        return prev_stats

    def state_dict(self):
        named_stats = {
            'net_stats': {}
        }
        for name, module in self.stream_module.named_modules():
            if module in self._module_stats:
                named_stats['net_stats'][name] = self._module_stats[module]
        named_stats['output_stride'] = self.output_stride
        named_stats['tile_output_lost'] = self.tile_output_lost  # type:ignore
        named_stats['tile_gradient_lost'] = self.tile_gradient_lost  # type:ignore
        named_stats['tile_output_shape'] = self._tile_output_shape  # type:ignore
        return named_stats

    def load_state_dict(self, state):
        self.disable()

        self.output_stride = state['output_stride']
        self.tile_output_lost = state['tile_output_lost']
        self.tile_gradient_lost = state['tile_gradient_lost']
        self._tile_output_shape = state['tile_output_shape']

        for name, module in self.stream_module.named_modules():
            if name in state['net_stats']:
                self._module_stats[module] = state['net_stats'][name] 

        self.enable()
