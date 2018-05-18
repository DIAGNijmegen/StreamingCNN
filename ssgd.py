"""
Author: Hans Pinckaers
April 11, 2018
"""
import math
from collections import namedtuple

import torch

from tqdm import tqdm
from IPython.core.debugger import set_trace

# Utility named tuples, makes code more readable
#
class Box(namedtuple('Box', 'y height x width sides')):
    def __str__(self):
        return '(Box x:%5.1f y:%5.1f width:%5.1f height:%5.1f)' % (self.x, self.y, self.width, self.height)

class Sides(namedtuple('Sides', 'left top right bottom')):
    def __str__(self):
        return '(Sides left:%r top:%r right:%r bottom:%r' % (self.left, self.top, self.right, self.bottom)

class IOShape(namedtuple('IOShape', 'batch channels height width')):
    def __str__(self):
        return '(IOShape batch:%2.1f channels:%2.1f height:%2.1f width:%2.1f)' % \
            (self.batch, self.channels, self.height, self.width)

class Lost(namedtuple('Lost', 'top left bottom right')):
    def __str__(self):
        return '(Lost top:%2.1f left:%2.1f bottom:%2.1f right:%2.1f)' % (self.top, self.left, self.bottom, self.right)

class LayerStats(object):
    """This class is responsible for calculating layer specific statistics,
    such as padding, output lost, gradient invalidated by convolution / zero-padding
    """
    def __init__(self, layer, padding, output_lost, downsamples, gradient_lost, output_shape, name, streaming=False):
        self.next = None
        self.previous = None
        self.padding = padding
        self.output_lost = output_lost
        self.gradient_lost = gradient_lost
        self.output_shape = output_shape
        self.downsamples = downsamples
        self.layer = layer
        self.name = name
        self.streaming = streaming

    @classmethod
    def stats_with_layer(cls, layer, input_shape, name):
        output_channels = input_shape[1]
        cur_stride = torch.FloatTensor(layer.stride)
        kernel_size = layer.kernel_size
        c_padding = layer.padding

        if isinstance(layer, torch.nn.MaxPool2d):
            cur_stride = torch.FloatTensor([layer.stride, layer.stride])
            kernel_size = (kernel_size, kernel_size)
        else:
            output_channels = layer.out_channels

        if isinstance(layer.padding, int):
            c_padding = [layer.padding, layer.padding]

        downsamples = cur_stride

        # Equations of the paper
        #
        lost_due_kernel_row = (kernel_size[0] - cur_stride[0]) / 2
        lost_due_stride_row = (input_shape[2] + c_padding[0] * 2 - kernel_size[0]) % cur_stride[0]
        lost_due_kernel_column = (kernel_size[1] - cur_stride[1]) / 2
        lost_due_stride_column = (input_shape[3] + c_padding[1] * 2 - kernel_size[1]) % cur_stride[1]

        p_left = math.floor(lost_due_kernel_row)
        p_right = math.ceil(lost_due_kernel_row) + lost_due_stride_row
        p_top = math.floor(lost_due_kernel_column)
        p_bottom = math.ceil(lost_due_kernel_column) + lost_due_stride_column

        lost_this_layer = Lost(top=p_top, left=p_left, bottom=p_bottom, right=p_right)
        grad_lost_this_layer = Lost(top=p_top * 2, left=p_left * 2, bottom=p_bottom * 2, right=p_right * 2)
        padding_this_layer = Lost(top=c_padding[1], left=c_padding[0], bottom=c_padding[1], right=c_padding[0])

        next_shape = [1, output_channels,
                      input_shape[2] - p_top - p_bottom + c_padding[0] * 2,
                      input_shape[3] - p_left - p_right + c_padding[1] * 2]

        next_shape[2] //= cur_stride[0]
        next_shape[3] //= cur_stride[1]

        next_shape = IOShape(1, next_shape[1], next_shape[2], next_shape[3])

        return cls(
            layer=layer,
            output_shape=next_shape,
            output_lost=lost_this_layer,
            padding=padding_this_layer,
            gradient_lost=grad_lost_this_layer,
            downsamples=downsamples,
            name=name
        )

    def calculate_input_shape(self, out_shape, valid=False, recursive=True, recursive_till="", gradient_lost=False):
        if not gradient_lost or self.previous is None:
            input_height = out_shape.height * self.downsamples[0] + self.output_lost.top + self.output_lost.bottom
            input_width = out_shape.width * self.downsamples[1] + self.output_lost.left + self.output_lost.right
        else:
            input_height = out_shape.height * self.downsamples[0] + self.gradient_lost.top + self.gradient_lost.bottom
            input_width = out_shape.width * self.downsamples[1] + self.gradient_lost.left + self.gradient_lost.right

        if not valid:
            input_height -= self.padding.top + self.padding.bottom
            input_width -= self.padding.left + self.padding.right

        shape = IOShape(batch=out_shape.batch,
                        channels=out_shape.channels,
                        height=input_height,
                        width=input_width)

        if recursive and self.previous is not None:
            return self.previous.calculate_input_shape(shape, valid, (self.previous is not recursive_till),
                                                       recursive_till, gradient_lost)
        else:
            return shape

    def calculate_output_shape(self, in_shape, output_layer):
        next_layer = self
        while next_layer is not None:
            in_shape = self.stats_with_layer(next_layer.layer, in_shape, next_layer.name).output_shape

            if next_layer == output_layer:
                break
            next_layer = next_layer.next

        return in_shape

    def calculate_input_coords_for_output(self, y: int, x: int, output_layer):
        total_downsamples = output_layer.downsampling_upto(self.name)
        tile_x = x * total_downsamples[1]
        tile_y = y * total_downsamples[0]
        return Box(y=max(tile_y, 0), height=0, x=max(tile_x, 0), width=0, sides=None)

    @property
    def total_downsampling(self) -> Lost:
        # should probably cache this
        return self.downsampling_upto()

    def downsampling_upto(self, until_layer=None):
        if self.name == until_layer or self.previous is None:
            return torch.FloatTensor([1., 1.])
        else:
            return self.downsamples * self.previous.downsampling_upto(until_layer)

    @property
    def total_padding(self) -> Lost:
        # should probably cache this
        if self.previous is not None:
            prev_lost = self.previous.total_padding
            return Lost(top=self.padding.top + prev_lost.top,
                        left=self.padding.left + prev_lost.left,
                        bottom=self.padding.bottom + prev_lost.bottom,
                        right=self.padding.right + prev_lost.right)
        else:
            return self.padding

    @property
    def total_output_lost(self) -> Lost:
        # should probably cache this
        if self.previous is not None:
            prev_lost = self.previous.total_output_lost
            return Lost(top=self.output_lost.top + math.ceil(prev_lost.top / self.downsamples[0]),
                        left=self.output_lost.left + math.ceil(prev_lost.left / self.downsamples[1]),
                        bottom=self.output_lost.bottom + math.ceil(prev_lost.bottom / self.downsamples[0]),
                        right=self.output_lost.right + math.ceil(prev_lost.right / self.downsamples[1]))
        else:
            return Lost(0, 0, 0, 0)

    def total_gradient_lost(self, output_layer) -> Lost:
        if self.name == output_layer or self.next is None:
            return Lost(0, 0, 0, 0)
        else:
            return self.next._total_gradient_lost(output_layer)

    def _total_gradient_lost(self, output_layer=None) -> Lost:
        # should probably cache this
        if self.name == output_layer:
            return self.gradient_lost
        elif self.next is not None:
            next_lost = self.next._total_gradient_lost(output_layer)
            return Lost(top=self.gradient_lost.top + next_lost.top * self.downsamples[0],
                        left=self.gradient_lost.left + next_lost.left * self.downsamples[1],
                        bottom=self.gradient_lost.bottom + next_lost.bottom * self.downsamples[0],
                        right=self.gradient_lost.right + next_lost.right * self.downsamples[1])
        else:
            return self.gradient_lost

    @staticmethod
    def _trim_tensor_with_lost(tensor, tile, lost):
        l_left = lost.left
        l_right = lost.right
        l_top = lost.top
        l_bottom = lost.bottom

        if tile.sides.left:
            l_left = 0
            # l_right += lost.left
        if tile.sides.top:
            l_top = 0
            # l_bottom += lost.top
        if tile.sides.right:
            # l_left = lost.left
            l_right = 0
        if tile.sides.bottom:
            # l_top = lost.top
            l_bottom = 0

        tensor = tensor[:, :,
                        int(l_top):int(tensor.shape[2] - l_bottom),
                        int(l_left):int(tensor.shape[3] - l_right)]
        return tensor, Lost(l_top, l_left, l_bottom, l_right)

    def trim_to_valid_output(self, output, tile):
        return self._trim_tensor_with_lost(output, tile, self.total_padding)

    def trim_to_valid_gradient(self, gradient, output_layer, tile):
        total_grad_lost = self.total_gradient_lost(output_layer)
        total_padding = self.total_padding

        total_lost = Lost(top=total_grad_lost.top + total_padding.top,
                          left=total_grad_lost.left + total_padding.left,
                          bottom=total_grad_lost.bottom + total_padding.bottom,
                          right=total_grad_lost.right + total_padding.right)

        return self._trim_tensor_with_lost(gradient, tile, total_lost)


class StreamingSGD(object):
    """
    The StreamingSGD class will run an image through the provided model in patches
    until the configured layer index, after that the feature map is run normally until
    the end of the network. The same happens in de backwards pass.
    """
    def __init__(self, model, stream_to_layer, input_shape, divide_in, cuda=False, verbose=False):
        """Configures the class

        Function calculates the coordinates of the forward and backward tiles.

        Args:
            stream_to_layer: An identifier of the layer indicating which layer we will switch to normal SGD
            input_shape: A shape (batch, channels, height, width) the model will be trained with
            divide_in: An integer indicating how many tiles the feature map will be divided in
            cuda: Optional argument (default is False), set to True if using cuda
            verbose: Optional argument, enable logging
        """
        self.model = model

        self._input_size = IOShape(*input_shape)
        self._stream_to_layer = stream_to_layer
        self._divide_in = divide_in
        self._cuda = cuda
        self._verbose = verbose

        # placeholder properties for batch training
        self._batch = False
        self._batch_count = 0

        self._tree, self._reverse_tree = self._create_sequential_tree(input_shape)
        self._first_layer = self._get_layers()[0][0]

        # placeholder attributes for backprop
        self._layer_outputs = {}
        self._layer_inputs = {}
        self._current_tile = None
        self._current_fmap_tile = None
        self._filled = {}
        self._layer_should_detach = False
        self._saved_gradients = {}

        self._add_hooks_sequential()

        # if self._verbose:
        #     for key, value in self._tree.items():
        #         print(value.output_shape, key)

        # Precalculate the coordinates of the tiles in the forward pass
        #
        self._forward_tiles, self._map_coords = self._calculate_tile_boxes()

        # if self._verbose:
        #     for ftiles, mtiles in zip(self._forward_tiles, self._map_coords):
        #         print(ftiles, mtiles)

        # Precalculate the coordinates of the tiles in the backward pass
        #
        self._back_tiles, self._grad_map_coords = self._calculate_tile_boxes(backwards=True)

        # if self._verbose:
        #     for btiles, mtiles in zip(self._back_tiles, self._grad_map_coords):
        #         print(btiles, mtiles)

        if self._verbose:
            print("Tile size forward:", (self._forward_tiles[0].height, self._forward_tiles[0].width))
            print("Tile size backward (for forward pass):", (self._back_tiles[0].height, self._back_tiles[0].width))

            # These memory reduction calculations are incorrect
            # We should also think about channels
            #
            print("*** Approximate memory reduction of streaming: {:2.1f}% ***".format(
                100 - self._back_tiles[0].height**2 / self._input_size.height**2 * 100))

    def forward(self, image):
        """Doing the forward pass

        Currently the image need to be able to be kept in RAM.

        Args:
            image: PyTorch tensor of the image in floats

        Returns:
            result: the final result of the network
            reconstructed feature map: a Pytorch Tensor / Variable of the reconstructed feature map
        """
        if self._verbose:
            print("Doing forward pass...")

        result, patches_output = self._forward_patches(image[None])
        return result, patches_output

    def backward(self, image, feature_map, loss, fill_gradients=False):
        """Doing the backward pass

        Args:
            image: PyTorch tensor of the image in floats (should be the same as used in forward pass)
            feature_map: PyTorch tensor with the reconstructed feature_map
            loss: PyTorch tensor / Variable with the calculated loss
            fill_gradients: Optional argument, if True will return
                the reconstructed gradients of the inputs of the layers

        Returns:
            list of gradients if fill_gradients is True, otherwise None
        """
        if self._verbose:
            print("Doing backward pass...")

        grad_embedding = torch.autograd.grad(loss, feature_map, only_inputs=False)[0]

        full_gradients = self._backward_tiles(image[None], grad_embedding, fill_gradients)
        return full_gradients

    def _calculate_tile_boxes(self, backwards=False):
        """
        This function calculates the coordinates of the tiles needed
        for the forward pass to reconstruct the feature map
        """
        last_layer_stats = self._tree[self._stream_to_layer]
        total_padding = last_layer_stats.total_padding
        output_shape = last_layer_stats.output_shape
        output_tile_shape = IOShape(batch=0, channels=0,
                                    height=output_shape.height // self._divide_in
                                    + total_padding.top + total_padding.bottom,
                                    width=output_shape.width // self._divide_in
                                    + total_padding.left + total_padding.right)

        tile_shape = last_layer_stats.calculate_input_shape(output_tile_shape, valid=True, gradient_lost=backwards)
        if backwards:
            map_tile_shape = self._tree[self._first_layer].calculate_output_shape(tile_shape, output_layer=last_layer_stats)
        else:
            map_tile_shape = output_tile_shape

        # The size of the patch/tile is feature map / divide_in
        #
        if output_shape.width % self._divide_in > 0 or output_shape.height % self._divide_in > 0:
            print("Check size of tiles:!", output_shape, " division asked not possible:", self._divide_in)

        if self._verbose and not backwards:
            print("Feature map to be reconstructed shape:", (output_shape.width, output_shape.height))
            print("Feature map divided in tile sizes:", (map_tile_shape.width, map_tile_shape.height))

        tile_boxes = []
        embed_boxes = []
        for y in range(0, int(output_shape.height), int(output_tile_shape.height)):
            for x in range(0, int(output_shape.width), int(output_tile_shape.width)):
                map_x = x - last_layer_stats.padding.left
                map_y = y - last_layer_stats.padding.top

                if backwards:
                    map_x -= last_layer_stats.total_output_lost.left
                    map_y -= last_layer_stats.total_output_lost.top

                map_x = max(0, map_x)
                map_y = max(0, map_y)

                tile = self._tree[self._first_layer].calculate_input_coords_for_output(
                    y=map_y, x=map_x, output_layer=last_layer_stats)

                # Keep track if we are at the sides (because we shouldn't crop the output here)
                #
                sides = Sides(left=(x == 0), top=(y == 0),
                              right=(tile.x + tile_shape.width >= self._input_size.width),
                              bottom=(tile.y + tile_shape.height >= self._input_size.height))

                if sides.bottom:
                    tile = Box(self._input_size.height - tile_shape.height, tile.height, tile.x, tile.width, tile.sides)
                    map_y = output_shape.height - map_tile_shape.height
                if sides.right:
                    tile = Box(tile.y, tile.height, self._input_size.width - tile_shape.width, tile.width, tile.sides)
                    map_x = output_shape.width - map_tile_shape.width

                tile_box = Box(tile.y, tile_shape.height, tile.x, tile_shape.width, sides)
                embed_box = Box(map_y, map_tile_shape.height, map_x, map_tile_shape.width, sides)

                tile_boxes.append(tile_box)
                embed_boxes.append(embed_box)

        # TODO: re-add output_lost check?
        return tile_boxes, embed_boxes

    def _forward_patches(self, image):
        """
        This function performs the streaming forward pass followed by
        the normal pass through the end of the network.
        """
        feature_map = None
        if self._verbose:
            iterator = tqdm(enumerate(self._forward_tiles), total=len(self._forward_tiles))
        else:
            iterator = enumerate(self._forward_tiles)

        # Reconstruct the feature map patch by patch
        #
        for i, tile in iterator:
            map_c = self._map_coords[i]

            # Fetch the relevant part of the full image
            #
            # TODO: try to remove ints in pytorch 0.4
            data = image[:, :,
                         int(tile.y):int(tile.y + tile.height),
                         int(tile.x):int(tile.x + tile.width)].clone()  # not sure if we need clone here?
            data.volatile = True

            if self._cuda:
                data = data.cuda()

            # Do the actual forward pass
            #
            self._layer_should_detach = False
            output = self.model.forward(data, self._stream_to_layer)
            tile_output, _ = self._tree[self._stream_to_layer].trim_to_valid_output(output, tile)
            output_size = self._tree[self._stream_to_layer].output_shape

            # Create (to be reconstructed) feature_map placeholder variable if it doesn't exists yet
            #
            if feature_map is None:
                feature_map = torch.autograd.Variable(torch.FloatTensor(1,
                                                                        int(output_size.channels),
                                                                        int(output_size.height),
                                                                        int(output_size.width)))
                if isinstance(data.data, torch.DoubleTensor):
                    feature_map = feature_map.double()

                if self._cuda:
                    feature_map = feature_map.cuda()

            # Save the output of the network in the relevant part of the feature_map
            #
            feature_map[:, :,
                        int(map_c.y):int(map_c.y + map_c.height),
                        int(map_c.x):int(map_c.x + map_c.width)] = tile_output

            tile_output = None  # trying memory management

        # From the feature map on we have to be able to generate gradients again
        #
        feature_map = torch.autograd.Variable(feature_map.data, requires_grad=True)

        # Run reconstructed feature map through the end of the network
        #
        final_output = self.model.forward(feature_map, start_at_layer=self._stream_to_layer)

        return final_output, feature_map

    @staticmethod
    def fill_tensor(data, data_loc: Box, tensor_shape: IOShape, already_filled: Box):
        # make relevant_gradient method more generalizable (could also be used for output in forward pass!
        # Check if this is a new row
        #
        rel_top = 0  # 0
        rel_bottom = 0  # 1
        rel_left = 0  # 2
        rel_right = 0  # 3

        already_filled_y = already_filled.y
        already_filled_x = already_filled.x
        already_filled_height = already_filled.height

        # check if new row
        if data_loc.x == 0:
            if data_loc.y + tensor_shape.height > already_filled_height:
                already_filled_y = already_filled_height
                already_filled_height = data_loc[0] + data.shape[2]
                already_filled_x = 0

        # Check x (column):
        #
        # If this gradient is exactly on the border we can use everything in x-axis
        #
        if data_loc.x == already_filled_x:
            rel_left = 0
            rel_right = data.shape[3]

        # If this gradient has some overlap with our current location check if we should use it
        #
        elif data_loc.x + tensor_shape.width > already_filled_x:
            if already_filled_x - data_loc.x < 0:
                print("Misses gradient in x-direction! Will probably crash...")
                print("We should be at:", already_filled_x, "but are at:", data_loc.x)
                print("Gradient size:", tensor_shape.width)
            rel_left = already_filled_x - data_loc.x
            rel_right = data.shape[3]

        # Check y (row):
        #
        # If this gradient is exactly on the border we can use everything in y-axis
        #
        if data_loc.y == already_filled_y:
            rel_top = 0
            rel_bottom = data.shape[2]

        # If this gradient has some overlap with our current location check if we should use it
        #
        elif data_loc.y + data.shape[2]:
            if already_filled_y - data_loc[0] < 0:
                print("We miss pieces of gradient in y-direction! Continuing...")
                print("We should be at:", already_filled_y, "but are at:", data_loc.y)
                return None
            rel_top = already_filled_y - data_loc.y
            rel_bottom = data.shape[2]

            # Check the bottom y-coord (if on the bottom we need to adjust)
            #
            if already_filled_y + (data.shape[2] - rel_top) > tensor_shape.height:
                rel_top += (already_filled_y + (data.shape[2] - rel_top)) \
                    - tensor_shape.height

        # Set new gradient location
        #
        already_filled_x += (rel_right - rel_left)

        # Check the right x-coord (if on the right edge we need to adjust)
        #
        if already_filled_x > tensor_shape.width:
            rel_left += (already_filled_x - tensor_shape.width)
            already_filled_x -= (already_filled_x - tensor_shape.width)

        rel_top = int(rel_top)
        rel_left = int(rel_left)
        rel_bottom = int(rel_bottom)
        rel_right = int(rel_right)

        if rel_top < 0:
            print("We miss gradients in y-axis before:", data_loc)
        if rel_left < 0:
            print("We miss gradients in x-axis before:", data_loc)

        return Box(rel_top, rel_bottom - rel_top, rel_left, rel_right - rel_left, None), \
            Box(already_filled_y, already_filled_height, already_filled_x, 0, None)

    def _backward_tiles(self, image, feature_map, fill_gradients=False):
        """
        This function is responsible for doing the backward pass per tile
        """
        if self._batch:
            self._save_batch_gradients()
            self._zero_gradient()

        if self._verbose:
            iterator = tqdm(enumerate(self._back_tiles), total=len(self._back_tiles))
        else:
            iterator = enumerate(self._back_tiles)

        self._filled = {}

        layerstats_last_layer = self._tree[self._stream_to_layer]
        first_backward_layer = layerstats_last_layer.name
        for i, tile in iterator:
            # Extract the tile
            #
            # TODO: test if clone() here is needed
            data = image[:, :,
                         int(tile.y):int(tile.y + tile.height),
                         int(tile.x):int(tile.x + tile.width)]

            if self._cuda:
                data = data.cuda()

            # Get the relevant part of the feature map for this tile
            #
            map_c = self._grad_map_coords[i]
            # TODO: test if clone() here is needed
            tile_map_grad = feature_map[:, :,
                                        int(map_c.y):int(map_c.y + map_c.height),
                                        int(map_c.x):int(map_c.x + map_c.width)]

            self._current_tile = tile
            self._current_fmap_tile = map_c

            # Do partial forward pass and backward pass
            # Hooks will be used to fetch relevant gradients / outputs
            #
            self._layer_should_detach = True
            self.model.forward(data, stop_at_layer=self._stream_to_layer)
            self._backward_sequential(first_backward_layer, tile_map_grad)

            # Needed for memory control
            #
            # del self._outputs
            del data
            del tile_map_grad

        if self._batch:
            self._sum_batch_gradients()
            self._batch_count += 1

        self._layer_should_detach = False

        if self._verbose:
            print("Everything filled:\n", self._check_gradient_size(self._filled))

        return None  # trimmed_full_gradient

    # --------------------------
    # Gradient utility functions
    #
    def _zero_gradient(self):
        """Zeros all the gradients of all the streaming Conv2d layers"""
        for key, layer in self._tree.items():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    layer.weight.grad.data.zero_()
                    layer.bias.grad.data.zero_()

    def _save_gradients(self):
        """Save all the gradients of all the streaming Conv2d layers"""
        for key, layerstat in self._tree.items():
            if isinstance(layerstat.layer, torch.nn.Conv2d) and layerstat.streaming:
                if layerstat.layer.weight.grad is not None:
                    self._saved_gradients[key] = (layerstat.layer.weight.grad.data.clone(),
                                                  layerstat.layer.bias.grad.data.clone())

    def _save_batch_gradients(self):
        """Save the valid batch gradients"""
        self._saved_batch_gradients = []
        for key, layer in self._tree.items():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    self._saved_batch_gradients[key] = (layer.weight.grad.data.clone(),
                                                        layer.bias.grad.data.clone())

    def _restore_gradients(self):
        """Restore the saved valid Conv2d gradients"""
        for key, layerstat in self._tree.items():
            if isinstance(layerstat.layer, torch.nn.Conv2d) and layerstat.streaming:
                if layerstat.layer.weight.grad is not None:
                    layerstat.layer.weight.grad.data.fill_(0)
                    layerstat.layer.bias.grad.data.fill_(0)

                    if key in self._saved_gradients:
                        gradient_tuple = self._saved_gradients[key]
                        layerstat.layer.weight.grad.data += gradient_tuple[0]
                        layerstat.layer.bias.grad.data += gradient_tuple[1]

    def _sum_batch_gradients(self):
        """Sum gradients within a batch"""
        if not self._saved_batch_gradients:
            return

        for key, layer in self._tree.items():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    gradient_tuple = self._saved_batch_gradients[key]
                    layer.weight.grad.data += gradient_tuple[0]
                    layer.bias.grad.data += gradient_tuple[1]

    def _apply_gradients(self, name, valid_grad):
        """Apply the relevant gradients"""
        self._tree[name].layer.backward(valid_grad)

    def start_batch(self):
        """Start a batch, this will sum all the gradients of the conv2d layers
        for every images backpropped after this function call."""
        self._batch = True
        self._batch_count = 0
        self._zero_gradient()

    def end_batch(self):
        """Stop current batch and divide all conv2d gradients with number of images in batch"""
        self._batch = False
        for key, layer in self._tree.items():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    layer.weight.grad.data /= self._batch_count
                    layer.bias.grad.data /= self._batch_count

    def _check_gradient_size(self, filled_grad_coords):
        correct = True
        for name, box in filled_grad_coords.items():
            size = self._tree[name].output_shape
            if size.height != box.height or size.width != box.x:
                correct = False
                print(name, "original gradient shape:", (size.width, size.height),
                      "reconstructed gradient shape: ", (box.x, box.height))
                break
        if not correct:
            print("!!!! Some gradient are smaller than they should be",
                  "this means we loose some information on the edges.",
                  "See log of configure() for information.\n")

        return correct

    # --------------------------
    # Model layer utility functions
    #
    def _backward_sequential(self, layer, gradient):
        output_layer = self._tree[self._stream_to_layer]
        while gradient is not None:
            layerstats = self._tree[layer]
            output = self._layer_outputs[layer]

            self._save_gradients()

            output.backward(gradient=gradient, retain_graph=True)

            # we carried the gradient to the input of current layer
            next_gradient = self._layer_inputs[layer].grad.clone()

            # apply right part of the gradient
            valid_grad, valid_lost = layerstats.trim_to_valid_gradient(gradient.clone(),
                                                                       self._stream_to_layer,
                                                                       self._current_tile)

            valid_output = output[:, :,
                                  int(valid_lost.top):int(output.shape[2] - valid_lost.bottom),
                                  int(valid_lost.left):int(output.shape[3] - valid_lost.right)]

            # calculate location of output in current layers input
            data_loc = layerstats.calculate_input_coords_for_output(
                y=self._current_fmap_tile.y,
                x=self._current_fmap_tile.x,
                output_layer=output_layer)

            if layer not in self._filled:
                self._filled[layer] = Box(0, 0, 0, 0, None)

            data_loc = Box(y=data_loc.y + valid_lost.top, height=0, x=data_loc.x + valid_lost.left, width=0, sides=None)

            relevant_box, grad_filled = self.fill_tensor(data=valid_grad,
                                                         data_loc=data_loc,
                                                         tensor_shape=self._tree[layer].output_shape,
                                                         already_filled=self._filled[layer])

            self._filled[layer] = grad_filled

            self._restore_gradients()

            if relevant_box.height > 0 and relevant_box.width > 0:
                relevant_grad = valid_grad[:, :, relevant_box.y:relevant_box.y + relevant_box.height,
                                           relevant_box.x:relevant_box.x + relevant_box.width]
                relevant_output = valid_output[:, :, relevant_box.y:relevant_box.y + relevant_box.height,
                                               relevant_box.x:relevant_box.x + relevant_box.width]

                relevant_output.backward(gradient=relevant_grad)

            gradient = next_gradient

            if layerstats.previous is None:
                break

            layer = layerstats.previous.name

    def _add_hooks_sequential(self):
        for name, layer in self._get_layers():
            layer.register_forward_pre_hook(self._forward_pre_hook)

            if name == self._stream_to_layer:
                layer.register_forward_hook(self._forward_hook)
                break

    def _forward_pre_hook(self, module, layer_input):
        # detach input inplace
        if self._layer_should_detach:
            name = self._reverse_tree[module]
            layerstats = self._tree[name]
            if layerstats.previous:
                self._layer_outputs[layerstats.previous.name] = layer_input[0].clone()
            layer_input[0].detach_()
            layer_input[0].requires_grad = True
            self._layer_inputs[name] = layer_input[0]

    def _forward_hook(self, module, layer_input, layer_output):
        # we need to save the output
        if self._layer_should_detach:
            name = self._reverse_tree[module]
            self._layer_outputs[name] = layer_output

    def _get_layers(self, modules=None):
        if modules is None:
            modules = self.model.named_children()

        layers = []
        for mod in modules:
            layers.append(mod)
            try:
                mod_layers = self._get_layers(mod[1].named_children())
                layers.extend(mod_layers)
            except StopIteration:
                pass

        return layers

    def _create_sequential_tree(self, input_shape):
        layer_dict = dict(self._get_layers())

        # loop through layers and do layer_stats, save in dict
        stats = {}
        reverse_stats = {}
        current_shape = input_shape
        prev_name = None
        streaming = True
        for name, layer in layer_dict.items():
            stats[name] = LayerStats.stats_with_layer(layer, current_shape, name)
            stats[name].streaming = streaming
            if name == self._stream_to_layer:
                streaming = False
            reverse_stats[layer] = name
            current_shape = stats[name].output_shape
            if prev_name is not None:
                stats[name].previous = stats[prev_name]
                stats[prev_name].next = stats[name]
            else:
                stats[name].previous = None

            prev_name = name

        return stats, reverse_stats
