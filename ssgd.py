"""
Author: Hans Pinckaers
April 11, 2018
"""
import torch
import numpy as np
from tqdm import tqdm

class StreamingSGD(object):
    """
    The StreamingSGD class will run an image through the provided model in patches
    until the configured layer index, after that the feature map is run normally until
    the end of the network. The same happens in de backwards pass.
    """

    def __init__(self, model):
        self.model = model

    def configure(self, layers, stop_index, input_shape, divide_in, cuda=False, verbose=False):
        """Configures the class

        Function calculates the coordinates of the forward and backward tiles.

        Args:
            layers: modules of the model
            stop_index: An index of layers indicating which layer we will switch to normal SGD
            input_shape: A shape (batch, channels, height, width) the model will be trained with
            divide_in: An integer indicating how many tiles the feature map will be divided in
            cuda: Optional argument (default is False), set to True if using cuda
            verbose: Optional argument, enable logging
        """
        self._input_size = input_shape
        self._layers = layers
        self._stop_index = stop_index
        self._divide_in = divide_in
        self._cuda = cuda
        self._verbose = verbose
        self._batch = False

        if self._verbose:
            print("Calculating patch boxes...")

        # Calculate overlap / pixels lost per layer
        #
        self._full_output_lost = self._getreconstructioninformation(self._layers[:self._stop_index],
                                                                    input_shape=(1, input_shape[2], input_shape[0], input_shape[1]))

        # We keep track of the gradient sizes to check if we reconstruct all of it
        #
        self._gradient_sizes = self._calculate_gradient_sizes(self._full_output_lost, self._input_size, self._stop_index)

        # Precalculate the coordinates of the tiles in the forward pass
        #
        self._forward_patches_c, self._output_size, self._map_coords = self._forward_patches_coords()

        # Precalculate the coordinates of the tiles in the backward pass
        #
        self._back_patches, self._grad_map_coords, self._patch_output_lost = self._backward_patches_coords()

        if self._verbose:
            print("Patch size forward:", (self._forward_patches_c[0][1][1], self._forward_patches_c[0][1][3]), "\n")
            print("Calculating gradient. Embedding size:", self._gradient_sizes[-1])
            print("Done. \nBackward patch size (for forward pass):", (self._back_patches[0][1][1], self._back_patches[0][1][3]))

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

    def _valid_grads_with_input(self, output_lost, start_index, sides=(0, 0, 0, 0)):
        """
        This function gives 'crop' boxes for the gradients indicating which parts are valid gradients
        (meaning would be the same as in the full image)
        """
        valid_boxes = []
        v_x = 0
        v_y = 0
        v_w = 0
        v_h = 0

        prev_down = output_lost[1][-1]
        for i in range(len(output_lost[0]) - 1):
            grad_lost = output_lost[0][-(i + 1)]
            grad_lost = (grad_lost * 2)
            grad_down = output_lost[1][-(i + 2)]
            factor = prev_down / grad_down

            grad_lost[0:2] /= grad_down
            grad_lost[2:] /= grad_down

            v_x *= factor[0]
            v_w *= factor[0]
            v_y *= factor[1]
            v_h *= factor[1]

            v_x += grad_lost[0]
            v_w += grad_lost[1]
            v_y += grad_lost[2]
            v_h += grad_lost[3]

            if sides[0] == 1:
                v_x = 0
            if sides[1] == 1:
                v_y = 0
            if sides[2] == 1:
                v_w = 0
            if sides[3] == 1:
                v_h = 0

            prev_down = grad_down
            valid_boxes.append((int(v_y), int(v_h), int(v_x), int(v_w)))

        return valid_boxes

    def _valid_gradients(self, patch, patch_embedding_grad, stop_index, output_lost, sides):
        """
        This function performs the backward pass (and the partial forward pass to
        reconstruct the intermediate activations) and crops the gradients.
        """

        # Do partial forward pass and backward pass
        #
        patch_output = self.model.forward(patch, stop_index=stop_index, detach=True)
        self.model.backward(gradient=patch_embedding_grad)
        patch_gradients = [gradient for gradient in self.model.gradients]

        # Calculate cropbox for gradients
        #
        valid_gradients_loss = self._valid_grads_with_input(output_lost, stop_index, sides=sides)

        # Fetch the outputs of the patches
        #
        patch_outputs = [output for output in self.model.output]

        # List with the cropped gradients
        #
        correct = []

        for index in range(len(patch_gradients)):
            if index == 0:
                p_grad = patch_gradients[0].data
                correct.append(p_grad)
            else:
                # Fetch gradient 'lost' in this layer
                #
                g_lost = valid_gradients_loss[index - 1]

                # Crop the gradient
                #
                p_grad = patch_gradients[index]
                p_grad = p_grad[:, :,
                                g_lost[0]:p_grad.shape[2] - g_lost[1],
                                g_lost[2]:p_grad.shape[3] - g_lost[3]]

                # Crop the output
                #
                patch_output = patch_outputs[-(index + 1)]
                patch_outputs[-(index + 1)] = patch_output[:, :,
                                                           g_lost[0]:patch_output.shape[2] - g_lost[1],
                                                           g_lost[2]:patch_output.shape[3] - g_lost[3]]
                correct.append(p_grad)

        return correct, patch_outputs

    def _forward_patches_coords(self):
        """
        This function calculates the coordinates of the patches / tiles needed
        for the forward pass to reconstruct the feature map
        """

        # Sum the padding / output lost over the whole streaming part of the network
        #
        lost = np.array(self._full_output_lost[0][0:self._stop_index]).sum(axis=0).astype(int)

        # Fetch the last downsampling
        #
        down = self._full_output_lost[1][self._stop_index - 1].astype(int)

        # With the lost + down it is possible to calculate the output_size
        # (= size of feature map to be reconstructed)
        #
        output_size = ((self._input_size[0] - lost[0] - lost[1]) // down[0],
                       (self._input_size[1] - lost[2] - lost[3]) // down[1])

        # The size of the patch/tile is feature map / divide_in
        #
        patch_size = (output_size[0] // self._divide_in, output_size[1] // self._divide_in)
        if output_size[0] % self._divide_in > 0 or output_size[1] % self._divide_in > 0:
            print("Check size patches!", output_size[0], "division asked not possible:", self._divide_in)

        if self._verbose:
            print("Embedding divided in patch sizes:", patch_size, "\n")

        embedding_coords = []
        boxes = []
        for y in range(0, output_size[1], patch_size[1]):
            for x in range(0, output_size[0], patch_size[0]):
                # Calculate which part of the full image we need for this part of the feature map
                #
                box = self._input_box_for_output((x, y, patch_size[0], patch_size[1]), self._full_output_lost, self._stop_index)

                # Keep track if we are at the sides (because we shouldn't crop the output here)
                #
                sides = [0, 0, 0, 0]  # left - top - right - bottom
                sides[0] = 1 if x == 0 else 0
                sides[2] = 1 if x + patch_size[0] >= output_size[0] else 0
                sides[1] = 1 if y == 0 else 0
                sides[3] = 1 if y + patch_size[1] >= output_size[1] else 0

                embedding_coords.append((y, int(y + patch_size[1]), x, int(x + patch_size[0])))
                boxes.append((sides, box))

        return boxes, output_size, embedding_coords

    def _output_box_for_input(self, input_coords, output_lost, stop_index):
        """
        This function calculates the coordinates of the feature map for a given input box
        """
        lost = np.array(output_lost[0][0:stop_index]).sum(axis=0).astype(int)
        down = output_lost[1][stop_index - 1].astype(int)
        lost = lost.astype(int)

        # We always need a multiple of downsampling, otherwise downsampling layers
        # will not begin in the correct corner for example with down = 2,
        # we will want index 0 or index 2 of an image, index 1 will give wrong results
        #
        p_x = input_coords[0] / down[0]
        p_y = input_coords[1] / down[1]
        p_w = (input_coords[2] - lost[0] - lost[1]) / down[0]
        p_h = (input_coords[3] - lost[3] - lost[2]) / down[1]

        # top bottom left right
        return (int(p_y), int(p_y + p_h), int(p_x), int(p_x + p_w))

    def _input_box_for_output(self, output_coords, output_lost, stop_index):
        """
        This function calculates the coordinates of the input for a given feature map box
        """
        lost = np.array(output_lost[0][0:stop_index]).sum(axis=0).astype(int)
        down = output_lost[1][stop_index - 1].astype(int)
        lost = lost.astype(int)

        p_x = output_coords[0] * down[0]
        p_y = output_coords[1] * down[1]
        p_w = output_coords[2] * down[0] + lost[0] + lost[1]
        p_h = output_coords[3] * down[1] + lost[2] + lost[3]

        return (p_y, p_y + p_h, p_x, p_x + p_w)  # top bottom left right

    def _forward_patches(self, image):
        """
        This function performs the streaming forward pass followed by
        the normal pass through the end of the network.
        """
        feature_map = None
        if self._verbose:
            iterator = tqdm(enumerate(self._forward_patches_c), total=len(self._forward_patches_c))
        else:
            iterator = enumerate(self._forward_patches_c)

        # Reconstruct the feature map patch by patch
        #
        for i, patch in iterator:
            coords = patch[1]
            map_c = self._map_coords[i]

            # Fetch the relevant part of the full image
            #
            data = image[:, :, coords[0]:coords[1], coords[2]:coords[3]].clone()  # not sure if we need clone here?
            data.volatile = True
            if self._cuda:
                data = data.cuda()

            # Do the actual forward pass
            #
            patch_output = self.model.forward(data, self._stop_index, detach=False)
            c = patch_output.shape[1]

            # Create (to be reconstructed) feature_map placeholder variable if it doesn't exists yet
            #
            if feature_map is None:
                feature_map = torch.autograd.Variable(torch.FloatTensor(1, c,
                                                                        int(self._output_size[1]),
                                                                        int(self._output_size[0])))
                if isinstance(data.data, torch.DoubleTensor):
                    feature_map = feature_map.double()

                if self._cuda:
                    feature_map = feature_map.cuda()

            # Save the output of the network in the relevant part of the feature_map
            #
            feature_map[:, :, map_c[0]:map_c[1], map_c[2]:map_c[3]] = patch_output.clone()
            patch_output = None  # trying memory management

        # From the feature map on we have to be able to generate gradients again
        #
        feature_map.volatile = False
        feature_map.requires_grad = True

        # Run reconstructed feature map through the end of the network
        #
        final_output = self.model.forward(feature_map, start_index=self._stop_index)

        return final_output, feature_map

    def _calculate_gradient_sizes(self, output_lost, input_size, stop_index):
        """
        This utility function calculates the size of the gradients per layer
        (basically equals output size)
        """
        sizes = []
        size = input_size
        prev_down = (1., 1.)
        for i in range(stop_index):
            down = output_lost[1][i] / prev_down
            lost = np.copy(output_lost[0][i])
            lost[0:2] /= output_lost[1][i]
            lost[2:] /= output_lost[1][i]
            size = ((size[0] - lost[0] - lost[1]) // down[0],
                    (size[1] - lost[2] - lost[3]) // down[1])
            sizes.append(size)

            prev_down = output_lost[1][i]

        return sizes

    def _relevant_gradients(self, gradients, targed_map_coords, outputs, filled_grad_coords, fill_gradients=False, full_gradients=None):
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
            filled_grad_coords = np.zeros((len(gradients), 3))  # y, x, height

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
        c_coord = targed_map_coords[0:2]

        new_row = False
        for i, gradient in enumerate(gradients):
            # Current gradient locations
            #
            c_gradient_coords = filled_grad_coords[i]
            c_gradient_size = self._gradient_sizes[-(i + 1)]

            if i == 0:
                # We do not lose gradients in the first layer, since they come from the reconstructed feature map.
                c_gradient_lost = [0, 0, 0, 0]
            else:
                # left right top bottom
                c_gradient_lost = self._patch_output_lost[0][-(i)] * 2

            # Calculate how much of the input is lost in the conv operation of this layer
            #
            c_gradient_down = prev_down / self._patch_output_lost[1][-(i + 1)]
            c_gradient_lost[0:2] /= self._patch_output_lost[1][-(i + 1)]
            c_gradient_lost[2:] /= self._patch_output_lost[1][-(i + 1)]
            prev_down = self._patch_output_lost[1][-(i + 1)]

            # Calculate current coords
            #
            c_coord *= c_gradient_down

            # If we already were at the border, the new gradient location is also there
            #
            if c_coord[0] <= 0:
                c_coord[0] = 0
            else:
                # otherwise we need to offset for input lost
                #
                c_coord[0] += c_gradient_lost[2]

            # Same holds for x-axes
            if c_coord[1] <= 0:
                c_coord[1] = 0
            else:
                c_coord[1] += c_gradient_lost[0]

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
                    print("We miss pieces of gradient in x-direction! Continuing...")
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
                    print("We miss pieces of gradient in x-direction! Continuing...")
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

            if rel_coords[3] - rel_coords[2] == 0:
                continue
            if rel_coords[1] - rel_coords[0] == 0:
                continue

            if rel_coords[0] < 0:
                print("We miss gradients in y-axis before:", c_coord)
            if rel_coords[2] < 0:
                print("We miss gradients in x-axis before:", c_coord)

            # Clone the right gradients and save the corresponding output
            #
            rel_gradient = gradient[:, :, rel_coords[0]:rel_coords[1], rel_coords[2]:rel_coords[3]]
            rel_output = outputs[-(i + 1)][:, :, rel_coords[0]:rel_coords[1], rel_coords[2]:rel_coords[3]]

            # If we want to return the reconstructed input gradients to each layer, save them here
            #
            if fill_gradients and i > 0:
                if i > len(full_gradients):
                    full_gradients.append(
                        torch.autograd.Variable(
                            torch.FloatTensor(1, rel_gradient.shape[1],
                                              int(self._gradient_sizes[-(i + 1)][0]),
                                              int(self._gradient_sizes[-(i + 1)][1]))))

                    full_gradients[i - 1].fill_(0)

                # Check if in this part of the gradient we didn't already put values
                #
                #if np.count_nonzero(full_gradients[i - 1][:, :,
                #                                          int(c_gradient_coords[0]):
                #                                          int(c_gradient_coords[2]),
                #                                          int(c_gradient_coords[1] - rel_gradient.shape[3]):
                #                                          int(c_gradient_coords[1])]) > 0:
                #    print("Overwriting!")

                full_gradients[i - 1][:, :,
                                      int(c_gradient_coords[0]):
                                      int(c_gradient_coords[2]),
                                      int(c_gradient_coords[1] - rel_gradient.shape[3]):
                                      int(c_gradient_coords[1])] = rel_gradient

            rel_gradients.append((rel_gradient.clone(), rel_output))

        if fill_gradients:
            return rel_gradients, filled_grad_coords, full_gradients
        else:
            return rel_gradients, filled_grad_coords, None

    def _backward_patches_coords(self):
        """
        This function calculates the coordinates of the patches / tiles needed
        for the backward pass to reconstruct the relevant activations
        """
        # Check if we can divide the feature map in the configured amount
        #
        if self._output_size[0] % self._divide_in > 0 or self._output_size[1] % self._divide_in > 0:
            print("!!!! Check unequal size embedding!", self._output_size[0], "division asked:", self._divide_in)

        # Calculate forward patch sizes for gradient checkpointing
        #
        # NOTE: output lost * 2 is not always correct,
        # if we loose pixels due to the kernel
        # not fitting perfectly of the whole image we also multiply those.
        # This propably results in bigger input patches than needed.
        # Due to deadlines we leave it like this for now.
        #
        box_size = [int(np.ceil(self._output_size[0] / self._divide_in)),
                    int(np.ceil(self._output_size[1] / self._divide_in))]

        # The first layer overlap
        #
        first_lost = self._full_output_lost[0][0]

        # This loop does the actual calculation,
        # there is probably an easier approach than looping here.
        #
        size = [box_size[0], box_size[1]]
        prev_down = self._full_output_lost[1][-1]
        for i in range(len(self._full_output_lost[0]) - 1):
            grad_lost = self._full_output_lost[0][-(i + 1)]

            # We lose twice the amount of valid gradient (see paper)
            #
            grad_lost = (grad_lost * 2)

            # Keep track of downsampling factor
            #
            grad_down = self._full_output_lost[1][-(i + 1)]
            factor = prev_down / grad_down
            size *= factor

            grad_lost[0:2] /= grad_down
            grad_lost[2:] /= grad_down

            # Sum the amount of gradient lost to the current size
            #
            size[0] += grad_lost[0] + grad_lost[1]
            size[1] += grad_lost[2] + grad_lost[3]

            prev_down = grad_down

        # Since we do not need to reconstruct the gradient of the first layer
        # we can use the normal overlap here
        #
        size[0] += first_lost[0] + first_lost[1]
        size[1] += first_lost[2] + first_lost[3]

        # Here we check if the convolutions fit the same between the tiles and
        # the full input image
        #
        new_output_lost = self._getreconstructioninformation(self.model.layers[:self._stop_index],
                                                             input_shape=(1, self._input_size[2], int(size[1]), int(size[0])),
                                                             stop_index=self._stop_index)

        adjust_step = 0
        if not np.all(np.array(self._full_output_lost[0])[:self._stop_index] == np.array(new_output_lost[0])):
            print("!!!! New patch size has different output loss profile !!!! This could cause problems \
                  with required patch sizes. Further testing needed.")

            full_output_arr = np.array(self._full_output_lost[0])[:self._stop_index]
            patch_output_arr = np.array(new_output_lost[0])
            diff_c = np.where(full_output_arr != patch_output_arr)
            max_diff = 0
            for c in diff_c:
                diff = np.max(patch_output_arr[c] - full_output_arr[c])
                diff *= np.max(self._full_output_lost[1][c[0]])
                if diff > max_diff:
                    max_diff = diff

            print("!!!! We are losing", max_diff, "px information of the input image because of different loss profile")

            for li, l in enumerate(new_output_lost[0]):
                print("new:", l, "old:",  self._full_output_lost[0][li])

            # TODO: Test this better
            # We need to add a certain margin because output lost can differ and
            # thus we can miss edges of gradient
            #
            adjust_step = 1  # we should be able to calculate this, but for now assume 1 extra pixel for embedding

        size = [int(s) for s in size]
        lost = np.array(new_output_lost[0][1:self._stop_index]).sum(axis=0).astype(int)
        lost = (lost).astype(int)

        first_lost = new_output_lost[0][0]

        patches = []
        grad_embedding_coords = []
        down = new_output_lost[1][self._stop_index - 1].astype(int)
        for y in range(0, self._output_size[1], box_size[1] - adjust_step):
            for x in range(0, self._output_size[0], box_size[0] - adjust_step):
                # Adjust the x and y with the total amount lost at the x and y
                # This would be the coordinates of the tile
                #
                p_x = x * down[0] - lost[0]
                p_y = y * down[1] - lost[2]

                p_w = size[0]
                p_h = size[1]

                # We need to keep track which tiles are on the edge,
                # since we shouldn't crop gradients there.
                #
                sides = [0, 0, 0, 0]  # left - top - right - bottom
                if p_x < 0:
                    p_x = 0
                    sides[0] = 1
                if p_x + p_w >= self._input_size[0]:
                    sides[2] = 1
                    p_x = self._input_size[0] - p_w
                if p_y < 0:
                    p_y = 0
                    sides[1] = 1
                if p_y + p_h >= self._input_size[1]:
                    sides[3] = 1
                    p_y = self._input_size[1] - p_h

                # Calculate coordinates of the feature map of this tile
                #
                grad_embed = self._output_box_for_input((p_x, p_y, p_w, p_h), new_output_lost, self._stop_index)
                grad_embedding_coords.append((grad_embed, (grad_embed[0], grad_embed[2])))
                patches.append((sides, (p_y, p_y + p_h, p_x, p_x + p_w)))

        return patches, grad_embedding_coords, new_output_lost

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
            iterator = tqdm(enumerate(self._back_patches), total=len(self._back_patches))
        else:
            iterator = enumerate(self._back_patches)

        for i, (sides, coords) in iterator:
            self._save_gradients()

            # Extract the tile
            #
            patch = image[:, :, coords[0]:coords[1], coords[2]:coords[3]].clone()  # TODO: test if clone() here is needed
            if self._cuda:
                patch = patch.cuda()

            # Get the relevant part of the feature map for this tile
            #
            map_c = self._grad_map_coords[i][0]
            patch_map_grad = feature_map[:, :, map_c[0]:map_c[1], map_c[2]:map_c[3]].clone()  # TODO: test if clone() here is needed

            # Do the actual partial forward pass and backward pass of the tile
            #
            val_gradients, outputs = self._valid_gradients(patch,
                                                           patch_map_grad,
                                                           self._stop_index,
                                                           self._patch_output_lost,
                                                           sides)

            # The coordinates of the reconstructed gradients w.r.t. the original input gradients
            #
            current_grad_pos = self._grad_map_coords[i][1]

            # Extract the relevant parts of the gradients
            #
            rel_grads, filled_grad_coords, full_gradients = self._relevant_gradients(val_gradients,
                                                                                     current_grad_pos,
                                                                                     outputs,
                                                                                     filled_grad_coords,
                                                                                     fill_gradients=fill_gradients,
                                                                                     full_gradients=full_gradients)
            self._restore_gradients()

            # Redo backward pass with the relevant parts of the gradient
            #
            self._apply_gradients(rel_grads)

            # Needed for memory control
            #
            del rel_grads
            del val_gradients
            del outputs
            del patch
            del patch_map_grad

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
        for i, layer in enumerate(self.model.layers[:self._stop_index]):
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    layer.weight.grad.data.zero_()
                    layer.bias.grad.data.zero_()

    def _save_gradients(self):
        """Save all the gradients of all the streaming Conv2d layers"""
        self._saved_gradients = []
        for i, layer in enumerate(self.model.layers[:self._stop_index]):
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    self._saved_gradients.append((layer.weight.grad.data.clone(),
                                                  layer.bias.grad.data.clone()))

    def _save_batch_gradients(self):
        """Save the valid batch gradients"""
        self._saved_batch_gradients = []
        for i, layer in enumerate(self.model.layers[:self._stop_index]):
            if isinstance(layer, torch.nn.Conv2d):
                if layer.weight.grad is not None:
                    self._saved_batch_gradients.append((layer.weight.grad.data.clone(),
                                                        layer.bias.grad.data.clone()))

    def _restore_gradients(self):
        """Restore the saved valid Conv2d gradients"""
        j = -1
        for i, layer in enumerate(self.model.layers[:self._stop_index]):
            if isinstance(layer, torch.nn.Conv2d):
                j += 1
                if layer.weight.grad is not None:
                    layer.weight.grad.data.fill_(0)
                    layer.bias.grad.data.fill_(0)

                    if len(self._saved_gradients) > 0:
                        gradient_tuple = self._saved_gradients[j]

                        layer.weight.grad.data += gradient_tuple[0]
                        layer.bias.grad.data += gradient_tuple[1]

    def _sum_batch_gradients(self):
        """Sum gradients within a batch"""
        if len(self._saved_batch_gradients) == 0:
            return

        j = -1
        for i, layer in enumerate(self.model.layers[:self._stop_index]):
            if isinstance(layer, torch.nn.Conv2d):
                j += 1
                if layer.weight.grad is not None:
                    gradient_tuple = self._saved_batch_gradients[j]
                    layer.weight.grad.data += gradient_tuple[0]
                    layer.bias.grad.data += gradient_tuple[1]

    def _apply_gradients(self, gradients):
        """Apply the relevant gradients"""
        for i, (grad, out) in enumerate(gradients):
            out.backward(grad)

    def start_batch(self):
        """Start a batch, this will sum all the gradients of the conv2d layers
        for every images backpropped after this function call."""
        self._batch = True
        self._batch_count = 0
        self._zero_gradient()

    def end_batch(self):
        """Stop current batch and divide all conv2d gradients with number of images in batch"""
        self._batch = False
        for i, layer in enumerate(self.model.layers):
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
            print("!!!! Some gradient are smaller than they should be, \
                  this means we loose some information on the edges.",
                  "See log of configure() for information.\n")

        return correct

    def _getreconstructioninformation(self, layers, input_shape, stop_index=-1):
        # TODO: we could also return the actual output sizes,
        # could make some code in this class easier

        # NOTE: lost = left right top bottom
        #
        lost = []
        downsamples = []

        last_shape = input_shape

        # TODO: add support for more types of layers (e.g. average layer)
        for i, l in enumerate(layers):
            if i == stop_index:
                break

            lost_this_layer = np.array([0, 0, 0, 0])

            # For the transposed convolutions the output size increases
            #
            if isinstance(l, torch.nn.ConvTranspose2d):
                # TODO: not tested
                #
                # Currently only valid padding with a filter_size of 2 and stride 2 is supported
                #
                if l.kernel_size != (2, 2):
                    print('Filter_size not supported', str(l.kernel_size), str(l))
                elif l.stride != (2, 2):
                    print('Stride not supported', str(l.stride), str(l))

                # A valid padding with filter size of 2 and stride of 2 results in a output size
                # that is double the size of the input image. No pixels are lost.
                #
                downsamples.append(np.array([0.5, 0.5]))
            elif isinstance(l, torch.nn.UpsamplingBilinear2d):
                downsamples.append(np.array([0.5, 0.5]))
            else:
                cur_stride = np.array(l.stride)
                kernel_size = l.kernel_size

                if isinstance(l, torch.nn.MaxPool2d):
                    cur_stride = np.array([l.stride, l.stride])
                    kernel_size = (kernel_size, kernel_size)
                else:
                    output_channels = l.out_channels

                # Equations of the paper
                #
                lost_due_kernel_row = (kernel_size[0] - cur_stride[0]) / 2
                lost_due_stride_row = (last_shape[2] - kernel_size[0]) % cur_stride[0]

                lost_due_kernel_column = (kernel_size[1] - cur_stride[1]) / 2
                lost_due_stride_column = (last_shape[3] - kernel_size[1]) % cur_stride[1]

                p_left = np.floor(lost_due_kernel_row)
                p_right = np.ceil(lost_due_kernel_row) + lost_due_stride_row

                p_top = np.floor(lost_due_kernel_column)
                p_bottom = np.ceil(lost_due_kernel_column) + lost_due_stride_column

                lost_this_layer = np.array([p_left, p_right, p_top, p_bottom])
                downsamples.append(cur_stride)

                # Different way of calculating total pixels lost
                #
                total_lost_row = last_shape[2] - (np.floor((last_shape[2] - kernel_size[0]) / cur_stride[0]) + 1) * cur_stride[0]
                total_lost_column = last_shape[3] - (np.floor((last_shape[3] - kernel_size[1]) / cur_stride[1]) + 1) * cur_stride[1]

                # Check if reconstructions would be correct:
                #
                if total_lost_row != p_left + p_right or total_lost_column != p_bottom + p_top:
                    print("Invalid reconstruction, total lost row:", total_lost_row, "total lost column:",
                          total_lost_column, "lost_this_layer", lost_this_layer)
                    print("Layer info", l, "kernel size:", kernel_size, "stride", cur_stride, "last_shape", last_shape)

                next_shape = [1, output_channels, last_shape[2] - p_top - p_bottom, last_shape[2] - p_left - p_right]
                next_shape[2] //= cur_stride[0]
                next_shape[3] //= cur_stride[1]

                if self._verbose:
                    print(next_shape, cur_stride, lost_this_layer, l.__class__.__name__)

            last_shape = next_shape
            lost.append(lost_this_layer)

        # Convert to float for potential upsampling
        #
        downsamples = [np.array([float(x), float(y)]) for x, y in downsamples]
        lost = [x.astype(float) for x in lost]

        for i in range(1, len(downsamples)):
            downsamples[i] *= downsamples[i - 1]
            lost[i][0:2] *= downsamples[i - 1][0]
            lost[i][2:] *= downsamples[i - 1][1]

        return lost, downsamples, np.array(lost).sum(axis=0).astype(int).tolist(), \
            downsamples[-1].astype(int).tolist()
