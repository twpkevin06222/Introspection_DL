import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from tensorflow.keras import Model, Sequential

class Activations:

    def __init__(self):
        pass

    def layers_dict(self, model):
        '''
        :param model: deep learning model

        :return:
            Dictionary with 'key': layer names, value: layer information
        '''
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        return layer_dict

    def feature_maps(self, model, layer_name, inps):
        '''
        This function visualize the intermediate activations of the filters within the layers
        :param model: deep learning model
        :param layer_name: desired layer name, if forgotten, please refer to layers_dict function
        :param inps: feed the network with input, such as images, etc. input dimension
                     should be 4.

        :return:
            feature maps of the layer specified by layer name,
            with dimension ( batch, row size, column size, channels)
        '''
        assert inps.ndim == 4, "Input tensor dimension not equal to 4!"
        #retrieve key value from layers_dict
        layer_dict = self.layers_dict(model)

        #layer output with respect to the layer name
        layer_output = layer_dict[layer_name].output
        viz_model = Model(inputs=model.inputs, outputs=layer_output)
        feature_maps = viz_model.predict(inps)

        print('Shape of feature maps:', feature_maps.shape)
        #shape (batch, row size, column size, channels)
        return feature_maps

    def plot_feature_maps(self, inps, row_num, col_num):
        '''
        This function can only plot the feature maps of a model
        :param inps: feature maps
        :param row_num: number of rows for the plot
        :param col_num: number of columns for the plot

        :return:
            grid plot of size (row_num * col_num)
        '''
        assert inps.ndim == 4, "Input tensor dimension not equal to 4!"

        print("Number of feature maps in layer: ", inps.shape[-1])

        fig, axes = plt.subplots(row_num, col_num)
        fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

        for i, ax in enumerate(axes.flat):
            img = inps[0, :, :, i]

            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

	    plt.tight_layout()
        plt.show()

    def acti_max(self, layer_name, inp_img, model, n_iter,
                 feature_index, learning_rate='default',
                 smoothing='standard', clip_min_val=0,
                 clip_max_val=1, verbose=True):
        '''
        Compute activation maximization with gradient ascent by taking
        the gradient of loss w.r.t input image
        :param layer_name: Particular layer for visualization
        :param inp_img: Input image to optimize
        :param model: Deep Network model
        :param n_iter: Number of iteration for optimization
        :param feature_index: The index of the feature maps in the layer
        :param learning_rate: Learning rate for gradient ascent, DEFAULT learning rate is used
        :param smoothing: 'standard' => no smoothing is use
                          'gaussian' => gaussian blur is implemented
        :param clip_min_val: Minimum value for clipping the optimized image
        :param clip_max_val: Maximum value for clipping the optimized image
        :param verbose: Print steps, loss and learning rate
        :return:
            optimized image after gradient ascent with
            shape (1, image width, image height, color channel)
        '''
        # retrieve output tensors w.r.t layer name
        layer_dict = self.layers_dict(model)
        layer_output = layer_dict[layer_name].output

        # build custom model for visualizing particular layer
        viz_model = tf.keras.Model(inputs=model.inputs, outputs=layer_output)

        assert feature_index <= layer_output.shape[-1], "Feature index out of bound!"

        # iteratively apply gradient ascent
        for i in range(n_iter):
            with tf.GradientTape() as tape:
                # cast input image to type tf.float32 for gradient tape
                inp_img = tf.cast(inp_img, dtype=tf.float32)
                # ensure the tensor is being traced by tape
                tape.watch(inp_img)

                # output feature w.r.t input image
                l_output = viz_model(inp_img)

                # if the feature belongs to dense layer
                if tf.rank(layer_output).numpy() == 2:
                    loss = tf.reduce_mean(l_output[:, feature_index])
                # if the feature belongs to convolutional layer
                else:
                    loss = tf.reduce_mean(l_output[:, :, :, feature_index])

            # compute the gradient loss function w.r.t input image
            grads = tape.gradient(loss, inp_img)

            # normalization trick for gradient
            grads /= (tf.math.sqrt(tf.reduce_mean(tf.math.square(grads))) + 1e-5)

            # default learning rate
            if learning_rate == 'default':
                learning_rate = 1.0 / (tf.math.reduce_std(grads) + 1e-8)

            # apply gradient ascent with with or without gaussian-blur
            if smoothing == 'standard':
                inp_img += np.clip(learning_rate * grads, clip_min_val, clip_max_val)

            if smoothing == 'gaussian':
                inp_img += scipy.ndimage.gaussian_filter(np.clip(learning_rate * grads, clip_min_val, clip_max_val), 2)

            if verbose:
                if i % 10 == 0:
                    print('Step: {}, Loss: {}, Learning rate: {}'.format(i, loss.numpy(), learning_rate.numpy()))
                    print()
        # output shape (1,image width,image height, color channel)
        return inp_img