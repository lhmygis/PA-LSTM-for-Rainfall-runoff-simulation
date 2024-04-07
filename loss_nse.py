import keras.backend as K
import numpy as np
from scipy import stats
import logging
import tensorflow as tf


def nse_loss(y_true, y_pred):

    #y_pred_swap = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]

    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    y_pred_swap = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)


    print("y_pred:", y_pred)
    print("y_pred_swap:", y_pred_swap)
    print("y_true:", y_true)



    numerator = K.sum(K.square(y_pred_swap - y_true), axis=1)  #分子
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    return numerator / denominator


def nse_metrics(y_true, y_pred):
    #y_pred_swap = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[3288,150,1] ->  [150,3288,1]

    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)
    y_pred_swap = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)

    numerator = K.sum(K.square(y_pred_swap - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    rNSE = numerator / denominator

    return 1.0 - rNSE
