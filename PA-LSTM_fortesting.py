import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers, callbacks
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation, LSTM
import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import ops

## Import libraries developed by this study
from libs_tf.Class_PA-LSTM import PRNN_Layer, ScaleLayer, LSTM_Layer
from libs_tf.camel_datahandle import DataforIndividual
from libs_tf import loss_nse

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.experimental.output_all_intermediates(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
np.seterr(divide='ignore',invalid='ignore')

attrs_path = 'D:\\MyGIS\\hydro_dl_project\\camels_data\\datafor_rm\\attributedata_671.csv'
working_path = 'D:\\MyGIS\\hydro_dl_project'

basin_climate1_id = [
'11532500']


training_start = '1980-10-01'
training_end = '2000-09-30'

testing_start = '2000-10-01'
testing_end = '2010-09-30'



def generate_train_test(train_set, test_set, wrap_length, attrs_file, basin_id):

    attrs_path = attrs_file

    if basin_id.startswith('0'):
        basin_id = basin_id[1:]

    else:
        basin_id = basin_id

    print(basin_id)

    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('gauge_id')
    rows_bool = (static_x.index == int(basin_id))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)
    print("static_x_np_shape:", static_x_np.shape)

    local_static_x = static_x_np[rows_int, :]  # basin_id index in attrs_path
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)
    print("local_static_x_test:", local_static_x_for_test)
    print("local_static_x_test_shape:", local_static_x_for_test.shape)

    train_x_np = train_set.values[:, :-1]
    train_y_np = train_set.values[:, -1:]
    test_x_np = test_set.values[:, :-1]
    test_y_np = test_set.values[:, -1:]

    wrap_number_train = (train_set.shape[0] - wrap_length) // 365 + 1

    local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = local_static_x_for_train.repeat(wrap_number_train, axis=0)
    print("local_static_x_train_shape:", local_static_x_for_train.shape)

    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))

    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)

    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i * 365:(wrap_length + i * 365), :]
        train_y[i, :, :] = train_y_np[i * 365:(wrap_length + i * 365), :]

    return train_x, train_y, local_static_x_for_train, test_x, test_y, local_static_x_for_test

def create_model(input_xd_shape, input_xs_shape, seed):
    xd_input_forprnn = Input(shape=input_xd_shape, batch_size=1, name='Input_xd1')
    xs_input = Input(shape=input_xs_shape, batch_size=1, name='Input_xs')

    hydro_output = PRNN_Layer(mode='normal', name='Hydro')(xd_input_forprnn)
    print("hydro_output",hydro_output)


    xd_hydro = Concatenate(axis=-1, name='Concat')([xd_input_forprnn, hydro_output])
    print("xd_hydro",xd_hydro)

    xd_hydro_scale = ScaleLayer(name='Scale')(xd_hydro)
    print("xd_hydro_scale",xd_hydro_scale)

    ealstm_hn, ealstm_cn = LSTM_Layer(input_xd = 9, input_xs = 27, hidden_size=128, seed=seed)([xd_hydro_scale,xs_input])
    fc2_out = Dense(units=1)(ealstm_hn)

    fc2_out = K.permute_dimensions(fc2_out, pattern=(1,0,2))  # for test model

    model = Model(inputs = [xd_input_forprnn, xs_input], outputs = fc2_out)
    return model

def train_model(model, train_xd, train_xs, train_y, ep_number, lrate, save_path):

    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)

    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                 restore_best_weights=True)

    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)

    tnan = callbacks.TerminateOnNaN()


    model.compile(loss=loss_nse.nse_loss, metrics=[loss_nse.nse_metrics], optimizer=tf.keras.optimizers.Adam(learning_rate=lrate))

    history = model.fit(x = [train_xd, train_xs], y = train_y, epochs=ep_number, batch_size=1,
                        callbacks=[save, es, reduce, tnan])
    return history

def normalize_xs(data):
    train_mean = np.mean(data, axis=1, keepdims=True)
    train_std = np.std(data, axis=1, keepdims=True)
    train_scaled = (data - train_mean) / train_std
    return train_scaled, train_mean, train_std

def normalize_minmax_xs(data):
    data_min = np.min(data)
    print("data_min:",data_min)
    data_max = np.max(data)
    data_scaled = (data - data_min) / (data_max - data_min)
    return data_scaled

def normalize_y(data):
    train_mean = np.mean(data, axis=-2, keepdims=True)
    train_std = np.std(data, axis=-2, keepdims=True)
    train_scaled = (data - train_mean) / train_std
    return train_scaled, train_mean, train_std



def nse_metrics(y_true,y_pred):

    y_true = K.constant(y_true)
    y_pred = y_pred # Omit values in the spinup period (the first 365 days)

    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)  [150,3288,1]
    y_pred = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)
    print("y_true_shape:",y_true.shape)
    print("y_pred_shape:",y_pred.shape)

    numerator = K.sum(K.square(y_true - y_pred), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    rNSE = numerator / denominator

    return 1.0 - rNSE

def test_model(model, test_xd, test_xs, save_path):

    model.load_weights(save_path, by_name=True)
    pred_y = model.predict(x = [test_xd,test_xs], batch_size=1)
    return pred_y

for i in range(len(basin_climate1_id)):
    tf.keras.backend.clear_session()
    K.clear_session()
    ops.reset_default_graph()
    hydrodata = DataforIndividual(working_path, basin_climate1_id[i]).load_data()

    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]

    print(f"The training data set is from {training_start} to {training_end}, with a shape of {train_set.shape}")
    print(f"The testing data set is from {testing_start} to {testing_end}, with a shape of {test_set.shape}")

    wrap_length = 2190  # 6-years for a train
    train_x, train_y, local_train, test_x, test_y, local_test = generate_train_test(train_set, test_set,
                                                                                    wrap_length=wrap_length,
                                                                                    attrs_file=attrs_path,
                                                                                    basin_id=basin_climate1_id[i])

    print(f'The shape of train_x, train_y, test_x, and test_y after wrapping by {wrap_length} days are:')
    print(f'{train_x.shape}, {train_y.shape},{local_train.shape}, {test_x.shape},{test_y.shape}, and {local_test.shape}')

    Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
    save_palstm = f'{working_path}/results/California_PA-LSTM_Models/pa_lstm_{basin_climate1_id[i]}.h5'

    model = create_model(input_xd_shape=(test_x.shape[1], test_x.shape[2]), input_xs_shape = (local_train.shape[1]) , seed=200)

    flow = test_model(model=model, test_xd=test_x, test_xs = local_test,
                      save_path=save_palstm)

    nse_test = nse_metrics(test_y, flow)
    print(f"{basin_climate1_id[i]}_nse_test:", K.eval(nse_test))


