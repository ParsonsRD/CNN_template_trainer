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


__all__ = ["rotate_translate", "create_model"]

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

def create_model(input_shape):
    """_summary_

    Args:
        cnn_input_shape (_type_): _description_
        filters (int, optional): _description_. Defaults to 50.
        number_of_layers (int, optional): _description_. Defaults to 14.

    Returns:
        _type_: _description_
    """

    model = Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(64, activation='swish'))
    model.add(layers.Dense(64, activation='swish'))

    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='swish'))
    model.add(layers.Dense(128, activation='swish'))

    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(150, activation='swish'))
    model.add(layers.Dense(150, activation='swish'))

    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='swish'))
    model.add(layers.Dense(128, activation='swish'))
    
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='swish'))
    model.add(layers.Dense(64, activation='swish'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
