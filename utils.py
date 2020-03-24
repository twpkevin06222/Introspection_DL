import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import os
import re

def get_layer_name(model, key_letter=None):
    '''
    Return a list of layer names from the model 
    
    Parameter: 
    model: model used 
    key_word: to retrieve layer name with certain key letter
    
    Return:
    a list containing layers name
    '''
    name_list = []
    if key_letter==None:
        for layer in model.layers:
            name_list.append(layer.name)
    else:
        regex = re.compile(".*".join(key_letter), re.IGNORECASE)
        for layer in model.layers:
            if regex.search(layer.name):
                name_list.append(layer.name)
    
    return name_list


def layer_get_weights(model, keyword):
    '''
    Retrieving weights of model filtered by keyword:
    Parameters:
    model: model class
    keyword: keyword to be filtered according to name of layer

    Return:
    List of layers that contains the corresponding layer's name
    and layer's weight
    '''

    filters_list = []
    name_list = []
    for layer in model.layers:
        if str(keyword) not in layer.name:
            continue
        filters, biases = layer.get_weights()

        print(layer.name, filters.shape)

        filters_list.append(filters)
        name_list.append(layer.name)

    return name_list, filters_list


def min_max_norm(inp):
    '''
    Min. max. normalization of filter to values between 0 - 1 for visualization

    Parametes:
    inp: inp of layer with weights
    '''
    v_min, v_max = inp.min(), inp.max()
    v_new = (inp - v_min) / (v_max - v_min)

    return v_new


def vis_filters(layer_name, layer, n_filters, n_cols=3, interpolation="nearest", n_channel=None, cmap='gray'):
    '''
    Function to visualize filters, based on layer name,
    number of filters and feature channels

    Parameters:
        layer_name : Name of the corresponding layer
        layer: Layer of the filters, where dimension [w, h , channel, number of filters]
        n_filters : Number of filters to visualize
        n_cols: Number of columns in the plot grid
        interpolation: Choise of interpolations
        n_channel: Number of channel to visualize
        cmap: For reference, please check cmap type from matplotlib
    '''
    assert n_filters % n_cols == 0, \
        'Unable to plot non-rectangular grid! Please check n_filters or n_cols value.'
    fig, axes = plt.subplots(int(n_filters / n_cols), n_cols, figsize=(5, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    if layer.shape[2] == 1:
        # for the 1st convolutional layer with third dimension equal to 1, convert 4D tensor into 3D tensor
        f_new = tf.squeeze(layer)

        for i, ax in enumerate(axes.flat):
            ax.imshow(f_new[:, :, i], interpolation=interpolation, cmap=cmap)
            ax.set_xlabel('Filter_{}'.format(i), fontsize=16)

            # removes ticks for axes
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout(h_pad=1.8)

    else:
        # for the rest of the convolutional layers
        for i, ax in enumerate(axes.flat):
            # get the filter
            f_new = layer[:, :, :, i]

            for j in range(n_channel):
                ax.imshow(f_new[:, :, i], interpolation=interpolation, cmap=cmap)
                ax.set_xlabel('Channel_{}/ \n Filter_{}'.format(j, i), fontsize=12)

                # removes ticks for axes
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout(h_pad=2.0)

    print('Layer name: {}'.format(layer_name))
    plt.show()
