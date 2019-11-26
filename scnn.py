"""
Author: Hans Pinckaers
MIT License
"""
import copy
import math
import os
from typing import NamedTuple, Union

import numpy as np
import torch
from torch.utils.cpp_extension import load
from tqdm import tqdm

# Load and compile cpp code to call cudnn conv2d backward function
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "conv2d_backward.cpp")
conv2d_cudnn = load(name="conv2d_backward", sources=[filename], verbose=False)

# Utility named tuples, makes code more readable
class Sides(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int

class Box(NamedTuple):
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

class Lost(NamedTuple):
    top: int
    left: int
    bottom: int
    right: int

    def __str__(self):
        return '(Lost top:%2.1f left:%2.1f bottom:%2.1f right:%2.1f)' \
            % (self.top, self.left, self.bottom, self.right)


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
                 saliency=False, gather_gradients=False, eps=1e-6):
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
        self.stream_module = stream_module
        self.verbose = verbose
        self.deterministic = deterministic
        self.eps = eps
        self.device = next(stream_module.parameters()).device
        self.dtype = next(stream_module.parameters()).dtype
        self.tile_shape = tile_shape
        self.gather_input_gradient = saliency
        self.gather_gradient = gather_gradients

        self._tile_output_shape = None
        self._module_stats = {}
        self._backward_seen_indices = {}
        self._saved_tensors = {}
        self._current_tile_input_loc = None
        self._hooks = []

        self.__configure()

    def __configure(self):
        # Save current model and cudnn flags, since we need to change them and restore later
        state_dict = self._save_parameters()
        old_deterministic_flag, old_benchmark_flag = self._set_cudnn_flags_to_determistic()
        self._reset_parameters_to_constant()

        # Add hooks to each layer to gather statistics
        self._add_hooks_for_statistics()

        # We need to temporary store statistics per layer to keep track of the
        # total output stride at each layer
        self._stats_per_grad_fn = {}

        # Create all-ones tile
        tile = torch.ones(self.tile_shape, dtype=self.dtype, requires_grad=True, device=self.device)

        self.__gather_forward_statistics(tile)
        if self.verbose: print('')
        self.__gather_backward_statistics(tile)

        # Remove all hooks and add hooks for correcting gradients 
        # during streaming
        self._remove_hooks()
        self._add_hooks_for_streaming()

        # Remove temporary data
        self._saved_tensors = {}
        del self._stats_per_grad_fn

        # Zero the gradients
        for param in self.stream_module.parameters():
            if param.grad is not None: param.grad.data.zero_()

        self._restore_parameters(state_dict)
        self._set_cudnn_flags(old_deterministic_flag, old_benchmark_flag)
        del state_dict

    def __gather_backward_statistics(self, tile):
        # Forward pass with grads enabled
        torch.set_grad_enabled(True)
        output = self.stream_module(tile)

        # Gather backward statistics
        self._tile_output_shape = output.shape

        gradient = torch.zeros(*output.shape, dtype=self.dtype, device=self.device)
        gradient[:, :, 
                 self.tile_output_lost.top:output.shape[2] - self.tile_output_lost.bottom,
                 self.tile_output_lost.left:output.shape[3] - self.tile_output_lost.right] = 1
        output.backward(gradient=gradient)

        # Calculate the output stride of the whole stream_module
        p_stats = self._prev_stats(output)
        self.output_stride = p_stats['output_stride'] * p_stats['stride'] if p_stats else [1, 1]

        self.tile_gradient_lost = self._non_max_border_amount(tile.grad)
        if self.verbose: print('\n', 'Input gradient lost', self.tile_gradient_lost)

    def __gather_forward_statistics(self, tile):
        torch.set_grad_enabled(False)
        output = self.stream_module(tile)
        self.tile_output_lost = self._non_max_border_amount(output)
        if self.verbose: print('\n', 'Output lost', self.tile_output_lost)

    def _reset_parameters_to_constant(self):
        for mod in self.stream_module.modules():
            if isinstance(mod, torch.nn.Conv2d):
                # to counter loating precision errors, we assign 1 to the weights and
                # normalize the output after the conv.
                torch.nn.init.constant_(mod.weight, 1)
                if mod.bias is not None:
                    torch.nn.init.constant_(mod.bias, 0)

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
        return Lost(top, left, bottom, right)

    def forward(self, image):
        """Perform forward pass with streaming.

        Parameters:
            image (torch.Tensor): CHW the image to stream
        """
        # The input image is likely quite small in terms of channels, for
        # performance reasons it is beneficial to copy to the GPU as a whole 
        # instead of tile-by-tile.
        image = image.to(self.device, non_blocking=True)[None]

        tile_width, tile_height = self.tile_shape[2:4]

        # Size of valid output of a tile
        valid_output_height = (self._tile_output_shape[2] - self.tile_output_lost.top - self.tile_output_lost.bottom)
        valid_output_width = (self._tile_output_shape[3] - self.tile_output_lost.left - self.tile_output_lost.right)

        # We will keep track which part of the output of the whole image we
        # already filled with valid values from tile output.
        already_filled = Box(0, 0, 0, 0, None)

        # Calculate size of output that we would get by inferencing the 
        # whole image.
        output_height = (image.shape[2] - self.tile_shape[2]) // self.output_stride[0] + self._tile_output_shape[2]
        output_width = (image.shape[3] - self.tile_shape[3]) // self.output_stride[1] + self._tile_output_shape[3]
        output_shape = IOShape(1, self._tile_output_shape[1], output_height, output_width)
        output = torch.empty(output_shape, dtype=self.dtype, device=self.device).fill_(999)

        n_rows = math.ceil(output_height / valid_output_height) + 1
        n_cols = math.ceil(output_width / valid_output_width) + 1

        if self.gather_input_gradient:
            self.saliency_map = torch.zeros(image.shape, dtype=self.dtype, device=self.device)

        with torch.no_grad():
            for row in range(n_rows):
                for col in range(n_cols):
                    # Coordinates of the output w.r.t. the output of full image
                    output_y = row * valid_output_height
                    output_x = col * valid_output_width

                    # Check if we are at borders, since we can not create
                    # overlap here and should not crop values.
                    sides_top = True if row == 0 else False
                    sides_left = True if col == 0 else False
                    sides_bottom = True if output_y * self.output_stride[0] + self.tile_shape[2] >= image.shape[2] else False
                    sides_right = True if output_x * self.output_stride[1] + self.tile_shape[3] >= image.shape[3] else False
                    sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                    # These values are used to crop invalid output values
                    lost_top = self.tile_output_lost.top if not sides_top else 0
                    lost_bottom = self.tile_output_lost.bottom if not sides_bottom else 0
                    lost_left = self.tile_output_lost.left if not sides_left else 0
                    lost_right = self.tile_output_lost.right if not sides_right else 0

                    # Since we need to stay at multiples of output stride we
                    # need to keep that into account when we are at the bottom
                    # and right side of the output.
                    if sides_bottom:
                        output_y = (image.shape[2] - self.tile_shape[2]) // self.output_stride[0]
                    if sides_right:
                        output_x = (image.shape[3] - self.tile_shape[3]) // self.output_stride[1]

                    output_y = output_y if not sides_top else 0
                    output_x = output_x if not sides_left else 0
                    output_loc = Box(output_y + lost_top, -1, output_x + lost_left, -1, sides)

                    # Coordinates of the input w.r.t. the output of full image
                    tile_y = output_y * self.output_stride[0]
                    tile_x = output_x * self.output_stride[1]

                    # Extract tile and perform forward pass
                    tile = image[:, :, tile_y:tile_y + tile_height, tile_x:tile_x + tile_width]
                    tile_output = self.stream_module(tile)

                    trimmed_output = tile_output[:, :,
                                                 lost_top:tile_output.shape[2] - lost_bottom,
                                                 lost_left:tile_output.shape[3] - lost_right]

                    new_output_box, updated_total_indices = self._new_value_indices(trimmed_output, output_loc, already_filled)
                    already_filled = updated_total_indices

                    relevant_output = trimmed_output[:, :, 
                                                     new_output_box.y:updated_total_indices.y + new_output_box.height,
                                                     new_output_box.x:new_output_box.x + new_output_box.width]
                    output[:, :,
                           int(updated_total_indices.y):int(updated_total_indices.height),
                           int(updated_total_indices.x - new_output_box.width):int(updated_total_indices.x)] = relevant_output

                    del tile

            assert sides_bottom and sides_right, "It seems like we could not reconstruct all output"

        # mem management
        del relevant_output
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
        image = image.to(self.device, non_blocking=True)[None]
        height = image.shape[2]
        width = image.shape[3]

        tile_height = self.tile_shape[2]
        tile_width = self.tile_shape[3]
        grad_lost = self.tile_gradient_lost

        output_height = self._tile_output_shape[2]
        output_width = self._tile_output_shape[3]

        valid_grad_height = (tile_height - grad_lost.top - grad_lost.bottom) // self.output_stride[0] 
        valid_grad_height *= self.output_stride[0]
        valid_grad_width = (tile_width - grad_lost.left - grad_lost.right) // self.output_stride[1]
        valid_grad_width *= self.output_stride[1]

        n_rows = int((height - grad_lost.top) // valid_grad_height + 1)
        n_cols = int((width - grad_lost.left) // valid_grad_width + 1)

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
                output_y = row * valid_grad_height // self.output_stride[0]
                output_x = col * valid_grad_width // self.output_stride[1]

                sides_top = True if row == 0 else False
                sides_left = True if col == 0 else False
                sides_bottom = True if output_y + output_height >= grad.shape[2] else False
                sides_right = True if output_x + output_width >= grad.shape[3] else False
                sides = Sides(sides_left, sides_top, sides_right, sides_bottom)

                # If the tile is at the bottom or right side of the input image
                # than we need to shift back so that the tile fits (does not go
                # over the border)
                if sides_bottom:
                    output_y = grad.shape[2] - output_height
                if sides_right:
                    output_x = grad.shape[3] - output_width

                input_y = output_y * self.output_stride[0]
                input_x = output_x * self.output_stride[1]

                tile = image[:, :, input_y:input_y + tile_height, input_x:input_x + tile_width]
                gradient = grad[:, :, output_y:output_y + output_height, output_x:output_x + output_width]

                self._current_tile_input_loc = Box(input_y, tile_height, input_x, tile_width, sides)
                self._saved_tensors = {}

                # Recover activations (gradient checkpointing)
                tile_output = self.stream_module(tile)

                # We are doing a forward pass
                lost_top = self.tile_output_lost.top if not sides_top else 0
                lost_bottom = self.tile_output_lost.bottom if not sides_bottom else 0
                lost_left = self.tile_output_lost.left if not sides_left else 0
                lost_right = self.tile_output_lost.right if not sides_right else 0

                # Trim output and gradient
                trimmed_output = tile_output[:, :,
                                             lost_top:tile_output.shape[2] - lost_bottom,
                                             lost_left:tile_output.shape[3] - lost_right]

                trimmed_grad = gradient[:, :,
                                        lost_top:gradient.shape[2] - lost_bottom,
                                        lost_left:gradient.shape[3] - lost_right]

                # Do backward pass, fix gradient in hooks
                trimmed_output.backward(trimmed_grad)

                # Memory management
                del tile
                del tile_output
                del trimmed_grad
                del trimmed_output

        # Memory management
        self._saved_tensors = {}
        self._current_tile_input_loc = None

        assert sides_right and sides_bottom, "It seems like we could not reconstruct all output"

    def disable(self):
        """Disable the streaming hooks"""
        self._remove_hooks()

    def enable(self):
        """Enable the streaming hooks"""
        self._remove_hooks()
        self._add_hooks_for_streaming()

    def _add_hooks_for_statistics(self):
        def forw_lambda(module, inpt, outpt):
            self._forward_gather_statistics_hook(module, inpt, outpt)

        def back_lambda(module, grad_in, grad_out):
            return self._backward_gather_statistics_hook(module, grad_in, grad_out)

        self._add_hooks(forward_hook=forw_lambda, backward_hook=back_lambda)

    def _add_hooks_for_streaming(self):
        def forw_lambda(module, inpt, outpt):
            self._forward_streaming_hook(module, inpt, outpt)

        def back_lambda(module, grad_in, grad_out):
            return self._backward_streaming_hook(module, grad_in, grad_out)

        self._add_hooks(forward_hook=forw_lambda, backward_hook=back_lambda,
                        back_modules=(torch.nn.Conv2d))

    def _add_hooks(self, forward_hook, backward_hook,
                   forward_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d), 
                   back_modules=(torch.nn.Conv2d, torch.nn.MaxPool2d)):
        for mod in self.stream_module.modules():
            if isinstance(mod, forward_modules):
                forw_handle = mod.register_forward_hook(forward_hook)
                self._hooks.append(forw_handle)
                if isinstance(mod, back_modules):
                    back_handle = mod.register_backward_hook(backward_hook)
                    self._hooks.append(back_handle)
    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def _forward_gather_statistics_hook(self, module, inpt, output):
        stride, kernel_size = self._stride_kernel_size_to_tuple(module)

        if not torch.is_grad_enabled():
            # Convert strided convolutions/pooling to average pool
            if isinstance(module, torch.nn.MaxPool2d) or (stride[0] > 1 and stride[0] > kernel_size[0]):
                # Pytorch documentation is explicitely against changing output in a forward hook
                # However, since we do not really need the graph or gradients to be correct
                # it shouldn't harm.
                if module.padding != 0:
                    padding = module.padding
                    if not isinstance(module.padding, tuple): padding = [module.padding, module.padding]
                    padded_input = torch.nn.functional.pad(inpt[0], (padding[1], padding[1], 
                                                                      padding[0], padding[0]))
                else:
                    padded_input = inpt[0]

                ks = int(kernel_size[0])
                st = int(stride[0])
                new_output = torch.nn.functional.avg_pool2d(padded_input, ks, st)
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
            if self.verbose: print("\n", module, "\n", module_stats['lost'])

            self._saved_tensors[module] = inpt
            self._module_stats[module] = module_stats
        else:
            module_stats = self._module_stats[module]

            p_stats = self._prev_stats(output)
            output_stride = p_stats['output_stride'] * p_stats['stride'] if p_stats else torch.tensor([1, 1])

            module_stats['output_stride'] = output_stride.clone().detach()

            self._stats_per_grad_fn[output.grad_fn] = module_stats
            self._module_stats[module] = module_stats

    def _backward_gather_statistics_hook(self, module, grad_in, grad_out):
        stride, kernel_size = self._stride_kernel_size_to_tuple(module)
        print(grad_out[0])
        exit()

        if grad_in[0] is not None:
            # We sum over the channels to deal with networks that do different operations
            # on groups of channels
            f_grad = torch.sum(grad_in[0], dim=1)[0]

            if isinstance(module, torch.nn.MaxPool2d):
                # MaxPool shifts indices around, which break the calculation to
                # find valid gradient values. To fix this we do an average pool
                # with the same kernel-size and stride and repeat using the stride.
                inpt = self._saved_tensors[module]
                padded_inpt = inpt[0]
                if module.padding != 0:
                    padded_inpt = torch.nn.functional.pad(inpt[0], (module.padding, module.padding, 
                                                                    module.padding, module.padding), value=-1)

                ks = int(kernel_size[0])
                st = int(stride[0])
                new_outpt = torch.nn.functional.avg_pool2d(padded_inpt, ks, st)[0]
                new_outpt = torch.sum(new_outpt, dim=0)

                f_grad = torch.sum(grad_out[0], dim=1)[0]
                f_grad = f_grad * new_outpt
                f_grad = f_grad.cpu()
                f_grad = np.repeat(f_grad, stride[0], axis=0)
                f_grad = np.repeat(f_grad, stride[1], axis=1)
                grad = np.zeros(grad_in[0].shape[2:])
                grad[:f_grad.shape[0], :f_grad.shape[1]] = f_grad
                f_grad = torch.from_numpy(grad)
                f_grad = f_grad.to(self.device)

            grad_lost = self._non_max_border_amount(grad_out[0])

            if self.verbose: print("\n", module, "\n", grad_lost)
            self._module_stats[module]['grad_lost'] = grad_lost

            valid_grad = (f_grad > (1-self.eps) * f_grad.max())

            # When kernel_size > stride we have some _overlap_ of gradients, 
            # this overlap makes extra positions in the input gradient invalid
            if stride[0] > 1 and kernel_size[0] > stride[0]:
                valid_lost = self._non_max_border_amount(f_grad)
                valid_grad.fill_(0)
                overlap = kernel_size[0] - stride[0]
                valid_grad[valid_lost.top + overlap:
                           valid_grad.shape[0] - valid_lost.bottom - overlap,
                           valid_lost.left + overlap:
                           valid_grad.shape[1] - valid_lost.right - overlap] = 1

            new_grad_in = valid_grad[None].expand(grad_in[0].shape[1], *valid_grad.shape)[None]
            return ((new_grad_in.type(self.dtype) * 10 - 1), *grad_in[1:])

    def _forward_streaming_hook(self, module, inpt, output):
        # Skip when not streaming
        if self._current_tile_input_loc is None: return

        # Save output per layer
        self._saved_tensors[module] = inpt

    def _backward_streaming_hook(self, module, grad_in, grad_out):
        # Skip when not streaming
        if self._current_tile_input_loc is None: return

        inpt = self._saved_tensors[module]
        grad_lost = self._module_stats[module]['grad_lost']  # Type: Lost
        grad = grad_out[0]

        # Trim gradient of invalid values
        sides = self._current_tile_input_loc.sides
        lost_top = grad_lost.top if not sides.top else 0
        lost_bottom = grad_lost.bottom if not sides.bottom else 0
        lost_left = grad_lost.left if not sides.left else 0
        lost_right = grad_lost.right if not sides.right else 0
        valid_grad = grad[:, :, lost_top:grad.shape[2] - lost_bottom,
                          lost_left:grad.shape[3] - lost_right]

        stride = module.stride if isinstance(module.stride, int) else module.stride[0]

        if module not in self._backward_seen_indices:
            self._backward_seen_indices[module] = Box(0, 0, 0, 0, None)

        output_stride = self._module_stats[module]['output_stride'] * self._module_stats[module]['stride']
        input_loc = self._current_tile_input_loc

        # Move the location according to how many pixels have been trimmed
        # this will be the location of the valid gradient of this layer in relation
        # to the actual gradient in a normal backpass
        data_loc_y = int(input_loc.y / output_stride[0]) + lost_top
        data_loc_x = int(input_loc.x / output_stride[1]) + lost_left
        data_loc = Box(data_loc_y, 0,
                       data_loc_x, 0,
                       input_loc.sides)

        # Calculate which part of the gradient is 'new'
        old_value_indices = self._backward_seen_indices[module]
        new_output_box, updated_total_indices = self._new_value_indices(valid_grad, data_loc, old_value_indices)
        self._backward_seen_indices[module] = updated_total_indices

        if new_output_box.height > 0 and new_output_box.width > 0:
            relevant_grad = valid_grad[:, :, new_output_box.y:new_output_box.y + new_output_box.height,
                                       new_output_box.x:new_output_box.x + new_output_box.width]

            # When debugging it can be useful to save the gradients
            if self.gather_gradient:
                if module in self.gradients: 
                    self.gradients[module].append(relevant_grad.clone().detach())
                else: 
                    self.gradients[module] = [relevant_grad.clone().detach()]

            input_y = (new_output_box.y + lost_top) * stride
            input_x = (new_output_box.x + lost_left) * stride

            # Accounting for padding:
            # the kernel locations are relative to the padded input, inpt[0] is not padded
            # this means that the corresponding input of the grad_loc is module.padding shifted to the left
            # we account for this:
            input_x -= module.padding[1]
            input_y -= module.padding[0]
            input_x = max(0, input_x)
            input_y = max(0, input_y)

            relevant_input_height = relevant_grad.shape[2] * stride + (module.kernel_size[0] - 1)
            relevant_input_width = relevant_grad.shape[3] * stride + (module.kernel_size[1] - 1)
            relevant_input = inpt[0][:, :,
                                     input_y:input_y + relevant_input_height,
                                     input_x:input_x + relevant_input_width]

            if self.gather_input_gradient:
                if module.in_channels == 3:
                    valid_grad_in = grad_in[0][:, :, lost_top:grad.shape[2] - lost_bottom,
                                               lost_left:grad.shape[3] - lost_right]
                    relevant_input_grad = valid_grad_in[:, :, new_output_box.y:new_output_box.y + new_output_box.height,
                                                        new_output_box.x:new_output_box.x + new_output_box.width]

                    self.saliency_map[:, :,
                                      updated_total_indices.y:
                                      updated_total_indices.height,
                                      updated_total_indices.x - relevant_input_grad.shape[3]:
                                      updated_total_indices.x] = relevant_input_grad

            # If layer has padding we need to pad based on if the current tile
            # is at the sides of the input.
            if (module.padding[0] > 0 or module.padding[1] > 0) and \
                    (sides.top or sides.left or sides.right or sides.bottom):
                # The size of the tile should remain equal. 
                crop_right = module.padding[1] if sides.left else 0
                crop_bottom = module.padding[0] if sides.top else 0
                relevant_input = inpt[0][:, :,
                     input_y:input_y + relevant_input_height - crop_bottom,
                     input_x:input_x + relevant_input_width - crop_right]

                relevant_input = torch.nn.functional.pad(relevant_input, (module.padding[1] if sides.left else 0,
                                                                          module.padding[1] if sides.right else 0,
                                                                          module.padding[0] if sides.top else 0,
                                                                          module.padding[0] if sides.bottom else 0))

            # Calculate the kernel gradients with the new unseen gradient values
            relevant_grad = relevant_grad.contiguous()
            grad_weight = conv2d_cudnn.backward(module.weight.shape,
                                                relevant_grad,
                                                relevant_input,
                                                (0, 0),  # padding
                                                module.stride,
                                                module.dilation,
                                                module.groups,
                                                not self.deterministic,  # benchmark
                                                self.deterministic)  # deterministic


            if module.bias is not None and len(grad_in) == 3:
                grad_bias = relevant_grad[0].sum((1, 2))

            del relevant_input
            del relevant_grad
        else:
            if self.verbose and not hasattr(self, '_inefficient_tile_shape_warning'):
                print("Warning: no new gradient values found. Tile size could be too small.")
                self._inefficient_tile_shape_warning = True

            grad_weight = grad_in[1].fill_(0)
            if len(grad_in) == 3:
                if module.bias is None: grad_bias = None
                else: grad_bias = grad_in[2].fill_(0)

        # Return the gradients
        if len(grad_in) == 3:
            if module.bias is None: return grad_in[0], grad_weight, None
            else: return grad_in[0], grad_weight, grad_bias
        elif len(grad_in) == 2:
            return grad_in[0], grad_weight
        else:
            return grad_in

    @staticmethod
    def _new_value_indices(data, data_indices, old_value_indices):
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
            old_values_height = data_indices[0] + data.shape[2]
            old_values_x = 0

        # Check x-axis:
        # If this gradient is exactly on the border of old_value_indices
        # everything is new.
        if data_indices.x == old_values_x:
            rel_left = 0
            rel_right = data.shape[3]

        # If data_indices has some overlap with old_value_indices, trim unique
        # indices.
        else:
            assert old_values_x - data_indices.x >= 0, "Misses data in x-axis!"
            rel_left = old_values_x - data_indices.x
            rel_right = data.shape[3]

        # Check y-axis:
        # Equal to column logic (see above)
        if data_indices.y == old_values_y:
            rel_top = 0
            rel_bottom = data.shape[2]
        else:
            assert old_values_y - data_indices[0] >= 0, "We miss data in y-axis"
            rel_top = old_values_y - data_indices.y
            rel_bottom = data.shape[2]

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

    @staticmethod
    def _stride_kernel_size_to_tuple(module):
        stride = module.stride
        kernel_size = module.kernel_size
        if not isinstance(module.stride, tuple):
            stride = [module.stride, module.stride]
        if not isinstance(module.kernel_size, tuple):
            kernel_size = (module.kernel_size, module.kernel_size)

        stride = torch.tensor(stride)
        return stride, kernel_size
