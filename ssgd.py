"""
Author: Hans Pinckaers
April 11, 2018
"""
import math
import torch
from tqdm import tqdm
from IPython.core.debugger import set_trace
from collections import namedtuple

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
        return '(IOShape batch:%2.1f channels:%2.1f height:%2.1f width:%2.1f)' % (self.batch, self.channels, self.height, self.width)

class Lost(namedtuple('Lost', 'top left bottom right')):
    def __str__(self):
        return '(Lost top:%2.1f left:%2.1f bottom:%2.1f right:%2.1f)' % (self.top, self.left, self.bottom, self.right)

class LayerStats(object):
    """This class is responsible for calculating layer specific statistics,
    such as padding, output lost, gradient invalidated by convolution / zero-padding
    """
    def __init__(self, layer, padding, output_lost, downsamples, gradient_lost, output_shape):
        self.next = []
        self.previous = []
        self.padding = padding
        self.output_lost = output_lost
        self.gradient_lost = gradient_lost
        self.output_shape = output_shape
        self.downsamples = downsamples
        self.layer = layer

    @classmethod
    def stats_with_layer(cls, layer, input_shape):
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
                      input_shape[2] - p_left - p_right + c_padding[1] * 2]

        next_shape[2] //= cur_stride[0]
        next_shape[3] //= cur_stride[1]

        next_shape = IOShape(1, next_shape[1], next_shape[2], next_shape[3])

        return cls(
            layer=layer,
            output_shape=next_shape,
            output_lost=lost_this_layer,
            padding=padding_this_layer,
            gradient_lost=grad_lost_this_layer,
            downsamples=downsamples
        )

    def calculate_input_shape(self, output_shape, valid=False, recursive=True, gradient_lost=False):
        if not gradient_lost:
            input_height = output_shape.height * self.downsamples[0] + self.output_lost.top + self.output_lost.bottom
            input_width = output_shape.width * self.downsamples[1] + self.output_lost.left + self.output_lost.right
        else:
            input_height = output_shape.height * self.downsamples[0] + self.gradient_lost.top + self.gradient_lost.bottom
            input_width = output_shape.width * self.downsamples[1] + self.gradient_lost.left + self.gradient_lost.right

        if not valid:
            input_height -= self.padding.top + self.padding.bottom
            input_width -= self.padding.left + self.padding.right

        shape = IOShape(batch=output_shape.batch,
                        channels=output_shape.channels,
                        height=input_height,
                        width=input_width)

        if recursive and self.previous is not None:
            return self.previous.calculate_input_shape(shape, valid)
        else:
            return shape

    @property
    def total_downsampling(self):
        # should probably cache this
        if self.previous is not None:
            return self.downsamples * self.previous.total_downsampling
        else:
            return self.downsamples

    @property
    def total_padding(self):
        # should probably cache this
        if self.previous is not None:
            prev_lost = self.previous.total_padding
            return Lost(top=self.padding.top + prev_lost.top,
                        left=self.padding.left + prev_lost.left,
                        bottom=self.padding.bottom + prev_lost.bottom,
                        right=self.padding.right + prev_lost.right)
        else:
            return self.padding

    def total_gradient_lost(self):
        # should probably cache this
        if self.previous is not None:
            prev_lost = self.previous.gradient_lost
            return Lost(top=self.gradient_lost.top + prev_lost.top,
                        left=self.gradient_lost.left + prev_lost.left,
                        bottom=self.gradient_lost.bottom + prev_lost.bottom,
                        right=self.gradient_lost.right + prev_lost.right)
        else:
            return self.gradient_lost

    def _trim_tensor_with_lost(self, tensor, tile, lost):
        l_left = lost.left
        l_right = lost.right
        l_top = lost.top
        l_bottom = lost.bottom

        if tile.sides.left:
            l_left = 0
            l_right += lost.left
        if tile.sides.top:
            l_top = 0
            l_bottom += lost.top
        if tile.sides.right:
            l_left = lost.left
            l_right = 0
        if tile.sides.bottom:
            l_top = lost.top
            l_bottom = 0

        tensor = tensor[:, :,
                        int(l_left):int(tensor.shape[2] - l_right),
                        int(l_top):int(tensor.shape[3] - l_bottom)]
        return tensor, Lost(l_top, l_left, l_bottom, l_right)

    def trim_to_valid_output(self, output, tile):
        return self._trim_tensor_with_lost(output, tile, self.total_padding)

    def trim_to_valid_gradient(self, gradient, tile):
        return self._trim_tensor_with_lost(gradient, tile, self.total_gradient_lost)


class StreamingSGD(object):
    """
    The StreamingSGD class will run an image through the provided model in patches
    until the configured layer index, after that the feature map is run normally until
    the end of the network. The same happens in de backwards pass.
    """

    def __init__(self, model):
        self.model = model

    def configure(self, stream_to_layer, input_shape, divide_in, cuda=False, verbose=False):
        """Configures the class

        Function calculates the coordinates of the forward and backward tiles.

        Args:
            stream_to_layer: An identifier of the layer indicating which layer we will switch to normal SGD
            input_shape: A shape (batch, channels, height, width) the model will be trained with
            divide_in: An integer indicating how many tiles the feature map will be divided in
            cuda: Optional argument (default is False), set to True if using cuda
            verbose: Optional argument, enable logging
        """
        self._input_size = input_shape
        self._stream_to_layer = stream_to_layer
        self._divide_in = divide_in
        self._cuda = cuda
        self._verbose = verbose
        self._batch = False
        self._tree = self._create_sequential_tree(input_shape)

        self._add_hooks_sequential()

        if self._verbose:
            [print(value.output_shape, key) for key, value in self._tree.items()]

        if self._verbose:
            print("Calculating tile boxes...")

        # Precalculate the coordinates of the tiles in the forward pass
        #
        self._forward_tiles, self._map_coords = self._calculate_tile_boxes()

        if self._verbose:
            [print(box) for box in self._forward_tiles]

        # Precalculate the coordinates of the tiles in the backward pass
        #
        self._back_tiles, self._grad_map_coords = self._calculate_tile_boxes(backwards=True)

        if self._verbose:
            [print(box) for box in self._back_tiles]

        if self._verbose:
            print("Tile size forward:", (self._forward_tiles[0].height, self._forward_tiles[0].width))
            # print("Calculating gradient. Embedding size:", self._gradient_sizes[-1])
            print("Tile size backward (for forward pass):", (self._back_tiles[0].height, self._back_tiles[0].width))

            # These memory reduction calculations are incorrect
            # We should also think about channels
            #
            # print("*** Memory reduction of patches: {:2.1f}% ***".format(100 - self._back_patches[0][1][1]**2 / self._input_size[1]**2 * 100))
            # print("*** Memory reduction of embedding: {:2.1f}% ***".format(
            #    100 - (self._output_size[1]**2 * self._output_size[0]) / (self._input_size[1]**2 * self._input_size[2]) * 100))

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

        relevant_grads, full_gradients = self._backward_patches(image[None], grad_embedding, fill_gradients)
        return full_gradients

    def _calculate_tile_boxes(self, backwards=False):
        # print(valid_boxes)
        """
        This function calculates the coordinates of the tiles needed
        for the forward pass to reconstruct the feature map
        """
        last_layer_stats = self._tree[self._stream_to_layer]
        total_padding = last_layer_stats.total_padding
        output_shape = last_layer_stats.output_shape
        output_tile_shape = IOShape(batch=0, channels=0,
                                    height=output_shape.height // self._divide_in + total_padding.top + total_padding.bottom,
                                    width=output_shape.width // self._divide_in + total_padding.left + total_padding.right)

        tile_shape = output_tile_shape
        total_downsamples = last_layer_stats.total_downsampling
        tile_shape = last_layer_stats.calculate_input_shape(tile_shape, valid=True, recursive=True, gradient_lost=backwards)

        # The size of the patch/tile is feature map / divide_in
        #
        if output_shape.width % self._divide_in > 0 or output_shape.height % self._divide_in > 0:
            print("Check size of tiles:!", output_shape, " division asked not possible:", self._divide_in)

        if self._verbose:
            print("Embedding divided in tile sizes:", output_tile_shape, "\n")

        tile_boxes = []
        embed_boxes = []
        for y in range(0, int(output_shape.height), int(output_tile_shape.height)):
            for x in range(0, int(output_shape.width), int(output_tile_shape.width)):
                padding = last_layer_stats.padding
                embed_x = max(x - padding.left, 0)
                tile_x = embed_x * total_downsamples[1]
                embed_y = max(y - padding.top, 0)
                tile_y = embed_y * total_downsamples[0]

                # Keep track if we are at the sides (because we shouldn't crop the output here)
                #
                sides = Sides(left=(x == 0), top=(y == 0),
                              right=(tile_x + tile_shape.width >= output_shape.width),
                              bottom=(tile_y + tile_shape.height >= output_shape.height))

                tile_box = Box(tile_y, tile_shape.height, tile_x, tile_shape.width, sides)
                embed_box = Box(embed_y, output_tile_shape.height, embed_x, output_tile_shape.width, sides)

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
            output = self.model.forward(data, self._stream_to_layer, detach=False)
            tile_output, trimmed = self._tree[self._stream_to_layer].trim_to_valid_output(output, tile)
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
        feature_map.volatile = False
        feature_map.requires_grad = True

        # Run reconstructed feature map through the end of the network
        #
        final_output = self.model.forward(feature_map, start_index=self._stream_to_layer)

        return final_output, feature_map

    def fill_tensor(self, data, data_location, already_filled):
        # make relevant_gradient method more generalizable (could also be used for output in forward pass!
        return

    def _relevant_gradients(self, gradients, targed_map_coords, grad_crop_box, outputs, filled_grad_coords, fill_gradients=False, full_gradients=None):
        """
        Gradients should be in order of network layers
        We assume patches are backpropped from left to right, top to bottom

        In this function we keep track of which parts of the gradients of inputs to each layer we
        already reconstructed. It will only remember the 'relevant' part of the gradient per patch.
        We need this functions because there can be significant overlap in the gradients in some layers.
        """

        # filled_grad_coords is the array that keeps track of which gradients are filled
        #
        if filled_grad_coords is None:
            filled_grad_coords = torch.FloatTensor(len(gradients), 3).fill_(0)  # y, x, height

        # If we want to remember the reconstructed input gradient create the placeholder list
        #
        if fill_gradients is True:
            if full_gradients is None:
                full_gradients = []

        # List with the final reconstructed input gradients (for this patch)
        #
        rel_gradients = []

        # To calculate where we currently are in the input gradient, we need to keep track of the downsampling
        #
        prev_down = self._patch_output_lost[1][-1]

        # Current coordinate of the input gradient of the reconstructed feature map
        #
        c_coord = torch.FloatTensor(targed_map_coords[0:2])

        new_row = False
        for i, gradient in enumerate(gradients):
            # Current gradient locations
            #
            c_gradient_coords = filled_grad_coords[i]
            c_gradient_size = self._gradient_sizes[-(i + 1)]

            if i == 0:
                # We do not lose gradients in the first layer, since they come from the reconstructed feature map.
                c_gradient_lost = torch.FloatTensor([0, 0, 0, 0])
                c_padding = torch.FloatTensor([0, 0, 0, 0])
            else:
                # left right top bottom
                c_gradient_lost = self._patch_output_lost[0][-(i)] * 2
                c_padding = self._patch_output_lost[3][-(i)]

            # Calculate how much of the input is lost in the conv operation of this layer
            #
            c_gradient_down = prev_down / self._patch_output_lost[1][-(i + 1)]
            c_gradient_lost[0:2] /= self._patch_output_lost[1][-(i + 1)]
            c_gradient_lost[2:] /= self._patch_output_lost[1][-(i + 1)]

            prev_down = self._patch_output_lost[1][-(i + 1)]

            # Calculate current coords
            #
            c_coord *= c_gradient_down
            # if c_gradient_coords[1] > 0 and i == 0:
            #    set_trace()

            # If we already were at the border, the new gradient location is also there
            #
            if c_coord[0] <= 0:
                c_coord[0] = 0
            else:
                # otherwise we need to offset for input lost
                #
                c_coord[0] += c_gradient_lost[2] + c_padding[1]

            # Same holds for x-axes
            if c_coord[1] <= 0:
                c_coord[1] = 0
            else:
                c_coord[1] += c_gradient_lost[0] + c_padding[0]

            # if i == 0:
            # set_trace()
            # print(i, c_coord.tolist(), c_padding.tolist(), c_gradient_coords.tolist())

            # if i == 1:
            # print(c_padding.tolist(), c_coord.tolist(), c_gradient_size)

            # Calculate relevant gradient coords
            #
            rel_coords = [0, 0, 0, 0]  # y, h+y, x, w+x

            # Check if this is a new row
            #
            if c_coord[1] == 0:
                if i == 0:
                    new_row = True
                # Don't reset X - row if not needed
                #
                if new_row:
                    if c_coord[0] + gradient.shape[2] > c_gradient_coords[2]:
                        c_gradient_coords[0] = c_gradient_coords[2]
                        c_gradient_coords[2] = c_coord[0] + gradient.shape[2]
                        c_gradient_coords[1] = 0

            #####################
            # Check x (column): #
            #####################
            #
            # If this gradient is exactly on the border we can use everything in x-axis
            #
            if c_coord[1] == c_gradient_coords[1]:
                rel_coords[2] = 0
                rel_coords[3] = gradient.shape[3]

            # If this gradient has some overlap with our current location check if we should use it
            #
            elif c_coord[1] + gradient.shape[3] > c_gradient_coords[1]:
                if c_gradient_coords[1] - c_coord[1] < 0:
                    print("gradient i:", i, "misses gradient in x-direction! Will probably crash...")
                    print("We should be at:", c_gradient_coords[1], "but are at:", c_coord[1])
                    print("Gradient size:", gradient.shape)
                    continue
                rel_coords[2] = c_gradient_coords[1] - c_coord[1]
                rel_coords[3] = gradient.shape[3]

            ##################
            # Check y (row): #
            ##################
            #
            # If this gradient is exactly on the border we can use everything in y-axis
            #
            if c_coord[0] == c_gradient_coords[0]:
                rel_coords[0] = 0
                rel_coords[1] = gradient.shape[2]

            # If this gradient has some overlap with our current location check if we should use it
            #
            elif c_coord[0] + gradient.shape[2]:
                if c_gradient_coords[0] - c_coord[0] < 0:
                    print("We miss pieces of gradient in y-direction! Continuing...")
                    print("We should be at:", c_gradient_coords[0], "but are at:", c_coord[0])
                    continue
                rel_coords[0] = c_gradient_coords[0] - c_coord[0]
                rel_coords[1] = gradient.shape[2]

                # Check the bottom y-coord (if on the bottom we need to adjust)
                #
                if c_gradient_coords[0] + (gradient.shape[2] - rel_coords[0]) > c_gradient_size[1]:
                    rel_coords[0] += (c_gradient_coords[0] + (gradient.shape[2] - rel_coords[0])) \
                        - c_gradient_size[1]

            # Set new gradient location
            #
            c_gradient_coords[1] += (rel_coords[3] - rel_coords[2])

            # Check the right x-coord (if on the right edge we need to adjust)
            #
            if c_gradient_coords[1] > c_gradient_size[0]:
                rel_coords[2] += (c_gradient_coords[1] - c_gradient_size[0])
                c_gradient_coords[1] -= (c_gradient_coords[1] - c_gradient_size[0])

            rel_coords = [int(coord) for coord in rel_coords]

            if rel_coords[0] < 0:
                print("We miss gradients in y-axis before:", c_coord)
            if rel_coords[2] < 0:
                print("We miss gradients in x-axis before:", c_coord)

            # Clone the right gradients and save the corresponding output
            #
            if rel_coords[1] - rel_coords[0] == 0:
                continue
            if rel_coords[3] - rel_coords[2] == 0:
                continue

            rel_gradient = gradient[:, :, rel_coords[0]:rel_coords[1], rel_coords[2]:rel_coords[3]]
            rel_output = outputs[-(i + 1)][:, :, rel_coords[0]:rel_coords[1], rel_coords[2]:rel_coords[3]]

            ###################
            if c_coord[0] > 0:
                c_coord[0] -= c_padding[1]

            # Same holds for x-axes
            if c_coord[1] > 0:
                c_coord[1] -= c_padding[0]
            ####################

            # If we want to return the reconstructed input gradients to each layer, save them here
            #
            if fill_gradients:
                if i == len(full_gradients):
                    full_gradients.append(
                        torch.autograd.Variable(
                            torch.FloatTensor(1, rel_gradient.shape[1],
                                              int(self._gradient_sizes[-(i + 1)][0]),
                                              int(self._gradient_sizes[-(i + 1)][1]))))

                    full_gradients[i].fill_(0)

                # Check if in this part of the gradient we didn't already put values
                #
                #if np.count_nonzero(full_gradients[i - 1][:, :,
                #                                          int(c_gradient_coords[0]):
                #                                          int(c_gradient_coords[2]),
                #                                          int(c_gradient_coords[1] - rel_gradient.shape[3]):
                #                                          int(c_gradient_coords[1])]) > 0:
                #    print("Overwriting!")

                full_gradients[i][:, :,
                                  int(c_gradient_coords[0]):
                                  int(c_gradient_coords[2]),
                                  int(c_gradient_coords[1] - rel_gradient.shape[3]):
                                  int(c_gradient_coords[1])] = rel_gradient.clone()

            rel_gradients.append((rel_gradient.clone(), rel_output))

        if fill_gradients:
            return rel_gradients, filled_grad_coords, full_gradients
        else:
            return rel_gradients, filled_grad_coords, None

    def _backward_patches(self, image, feature_map, fill_gradients=False):
        """
        This function is responsible for doing the backward pass per tile
        """
        relevant_grads = []
        filled_grad_coords = None
        full_gradients = None

        if self._batch:
            self._save_batch_gradients()
            self._zero_gradient()

        if self._verbose:
            iterator = tqdm(enumerate(self._back_tiles), total=len(self._back_tiles))
        else:
            iterator = enumerate(self._back_tiles)

        for i, tile in iterator:
            self._save_gradients()

            # Extract the tile
            #
            data = image[:, :,
                         int(tile.y):int(tile.y + tile.height),
                         int(tile.x):int(tile.x + tile.width)].clone()  # TODO: test if clone() here is needed

            if self._cuda:
                data = data.cuda()

            # Get the relevant part of the feature map for this tile
            #
            map_c = self._grad_map_coords[i]
            tile_map_grad = feature_map[:, :,
                                        int(map_c.y):int(map_c.y + map_c.height),
                                        int(map_c.x):int(map_c.x + map_c.width)].clone()  # TODO: test if clone() here is needed

            # Do partial forward pass and backward pass
            # Hooks will be used to fetch relevant gradients / outputs
            #
            self.model.forward(data, stop_index=self._stream_to_layer)
            self.model.backward(gradient=tile_map_grad)

            # self._restore_gradients() TODO: restore gradient should be layer based

            # Redo backward pass with the relevant parts of the gradient
            #
            # self._apply_gradients() TODO: apply gradient should be layer based

            # Needed for memory control
            #
            del self._gradients
            del self._outputs
            del data
            del tile_map_grad

        if self._batch:
            self._sum_batch_gradients()
            self._batch_count += 1

        trimmed_full_gradient = []
        if full_gradients and self._verbose:
            for i, gradients in enumerate(full_gradients):
                for j, gradient in enumerate(gradients):
                    filled_shape = filled_grad_coords[i]
                    gradient = gradient[:, 0:int(filled_shape[2]), 0:int(filled_shape[1])]
                    trimmed_full_gradient.append(gradient)
                    print("Filled gradient size", gradient.shape)

        if self._verbose:
            print("\nFilled gradient sizes:")
            print(filled_grad_coords, "\n")

            print("Everything filled:\n", self._check_gradient_size(filled_grad_coords, self._gradient_sizes))

        return relevant_grads, trimmed_full_gradient

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
        self._saved_gradients = {}
        for key, layer in self._tree.items():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    self._saved_gradients[key] = (layer.weight.grad.data.clone(),
                                                  layer.bias.grad.data.clone())

    def _save_batch_gradients(self):
        """Save the valid batch gradients"""
        self._saved_batch_gradients = []
        for key, layer in self._tree.items():
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    self._saved_batch_gradients[key] = (layer.weight.grad.data.clone(),
                                                        layer.bias.grad.data.clone())

    def _restore_gradients(self, name):
        """Restore the saved valid Conv2d gradients"""
        layer = self._tree[name].layer
        if layer.weight.grad is not None:
            layer.weight.grad.data.fill_(0)
            layer.bias.grad.data.fill_(0)

            if len(self._saved_gradients) > 0:
                gradient_tuple = self._saved_gradients[name]
                layer.weight.grad.data += gradient_tuple[0]
                layer.bias.grad.data += gradient_tuple[1]

    def _sum_batch_gradients(self):
        """Sum gradients within a batch"""
        if len(self._saved_batch_gradients) == 0:
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

    def _check_gradient_size(self, filled_grad_coords, gradient_sizes):
        correct = True
        for i, size in enumerate(gradient_sizes):
            filled_size = filled_grad_coords[-(i + 1)]
            if size[0] > filled_size[2] and size[1] > filled_size[1]:
                correct = False
                break
        if not correct:
            print("!!!! Some gradient are smaller than they should be",
                  "this means we loose some information on the edges.",
                  "See log of configure() for information.\n")

        return correct

    # --------------------------
    # Model layer utility functions
    #
    def _add_hooks_sequential(self):
        for name, layerstats in self._get_layers():
            if name == self._stream_to_layer:
                break
            layerstats.layer.register_forward_pre_hook(lambda module, input: self._forward_pre_hook(module, input, name))
            layerstats.layer.register_forward_hook(lambda module, input, output: self._forward_hook(module, input, output, name))
            layerstats.layer.register_backward_hook(lambda module, grad_input, grad_output: self._backward_hook(module, grad_input, grad_output, name))
        return

    def _forward_pre_hook(self, module, input):
        # detach input inplace
        input.detach_()

    def _forward_hook(self, module, input, output):
        # we need to save the output
        name = self._reverse_tree[module]
        self._outputs[name] = self._tree[name].trim_to_valid_output(output, self._current_tile)

    def _backward_hook(self, module, grad_input, grad_output):
        name = self._reverse_tree[module]
        valid_grad = self._tree[name].trim_to_valid_gradient(grad_output.clone(), self._current_tile)
        self._restore_gradients(name)
        self._apply_gradient(name, valid_grad)

    def _get_layers(self, modules=None):
        if modules is None:
            modules = self.model.named_children()

        layers = []
        for m in modules:
            layers.append(m)
            try:
                mod_layers = self._get_layers(m[1].named_children())
                layers.extend(mod_layers)
            except StopIteration:
                pass

        return layers

    def _create_sequential_tree(self, input_shape):
        layer_dict = dict(self._get_layers())

        # loop through layers and do layer_stats, save in dict
        stats = {}
        current_shape = input_shape
        prev_name = None
        for name, layer in layer_dict.items():
            stats[name] = LayerStats.stats_with_layer(layer, current_shape)
            current_shape = stats[name].output_shape
            if prev_name is not None:
                stats[name].previous = stats[prev_name]
                stats[prev_name].next = stats[name]
            else:
                stats[name].previous = None

            prev_name = name

        return stats
