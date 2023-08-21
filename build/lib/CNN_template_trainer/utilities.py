"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras import layers
from tensorflow.keras.regularizers import l2
from keras.models import Sequential


__all__ = ["poisson_likelihood_gaussian", "tensor_poisson_likelihood", "rotate_translate", "create_model_cnn"]

def rotate_translate(pixel_pos_x, pixel_pos_y, x_trans, y_trans, phi):
    """
    Function to perform rotation and translation of pixel lists
    Parameters
    ----------
    pixel_pos_x: ndarray
        Array of pixel x positions
    pixel_pos_y: ndarray
        Array of pixel x positions
    x_trans: float
        Translation of position in x coordinates
    y_trans: float
        Translation of position in y coordinates
    phi: float
        Rotation angle of pixels
    Returns
    -------
        ndarray,ndarray: Transformed pixel x and y coordinates
    """

    cosine_angle = np.cos(phi[..., np.newaxis])
    sin_angle = np.sin(phi[..., np.newaxis])

    pixel_pos_trans_x = (x_trans - pixel_pos_x) * cosine_angle - (
        y_trans - pixel_pos_y
    ) * sin_angle

    pixel_pos_trans_y = (pixel_pos_x - x_trans) * sin_angle + (
        pixel_pos_y - y_trans
    ) * cosine_angle
    
    return pixel_pos_trans_x, pixel_pos_trans_y

def poisson_likelihood_gaussian(image, prediction, spe_width=0.5, ped=1):
    """_summary_

    Args:
        image (_type_): _description_
        prediction (_type_): _description_
        spe_width (float, optional): _description_. Defaults to 0.5.
        ped (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    image = np.asarray(image)
    prediction = np.asarray(prediction)
    spe_width = np.asarray(spe_width)
    ped = np.asarray(ped)
    
    sq = 1. / np.sqrt(2 * np.pi * (np.power(ped, 2)
                                + prediction * (1 + np.power(spe_width, 2))))
    
    diff = np.power(image - prediction, 2.)
    denom = 2 * (np.power(ped, 2) + prediction * (1 + np.power(spe_width, 2)))
    expo = np.asarray(np.exp(-1 * diff / denom))
    
    # If we are outside of the range of datatype, fix to lower bound
    min_prob = np.finfo(expo.dtype).tiny
    expo[expo < min_prob] = min_prob
    
    return -2 * np.log(sq * expo)

def tensor_poisson_likelihood(image, prediction, spe_width=0.5, ped=1):
    """_summary_

    Args:
        image (_type_): _description_
        prediction (_type_): _description_
        spe_width (float, optional): _description_. Defaults to 0.5.
        ped (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    prediction = tf.clip_by_value(prediction, 1e-6, 1e9)

    sq = 1. / K.sqrt(2. * np.pi * (K.square(ped)
                                + prediction * (1. + K.square(spe_width))))

    diff = K.square(image - prediction)
    denom = 2. * (K.square(ped) + prediction * (1 + K.square(spe_width)))
    expo = K.exp(-1 * diff / denom)
    expo = tf.clip_by_value(expo, 1e-20, 100)

    return K.mean(-2 * K.log(sq * expo))

def create_model_cnn(cnn_input_shape, filters=50, number_of_layers=14):
    """_summary_

    Args:
        cnn_input_shape (_type_): _description_
        filters (int, optional): _description_. Defaults to 50.
        number_of_layers (int, optional): _description_. Defaults to 14.

    Returns:
        _type_: _description_
    """

    # OK first we have out CNN input layer, it's time distributed to account for the multiple telescope types

    l2_reg = l2(0.0005)

    # Fit image pixels using a multi layer perceptron
    model = Sequential()
    model.add(layers.Conv2D(filters=filters, activation="relu", kernel_size=(1, 1), padding="same", kernel_regularizer=l2_reg, input_shape=cnn_input_shape))
    # We make a very deep network
    for n in range(number_of_layers):
        model.add(layers.Conv2D(filters=filters, activation="relu", kernel_size=(1, 1), padding="same", kernel_regularizer=l2_reg))

    model.add(layers.Conv2D(filters=1, activation="linear",
                                                       kernel_size=(1, 1), padding="same"))

    return model
