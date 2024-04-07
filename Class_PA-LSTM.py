import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation, LSTM
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()


class PRNN_Layer(Layer):

    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(PRNN_Layer, self).__init__(**kwargs)


    def build(self, input_shape):
        #print('******',input_shape)    [None,2190,5]


        self.f = self.add_weight(name='f', shape=(1,),
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                 trainable=True)
        self.smax = self.add_weight(name='smax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1 / 15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qmax = self.add_weight(name='qmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.ddf = self.add_weight(name='ddf', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                   trainable=True)
        self.tmin = self.add_weight(name='tmin', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.tmax = self.add_weight(name='tmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)


        super(PRNN_Layer, self).build(input_shape)

    def heaviside(self, x):


        return (K.tanh(5 * x) + 1) / 2


    def rainsnowpartition(self, p, t, tmin):

        tmin = tmin * -3  # (-3.0, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]


    def snowbucket(self, s0, t, ddf, tmax):

        ddf = ddf * 5  # (0, 5.0)
        tmax = tmax * 3  # (0, 3.0)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt


    def soilbucket(self, s1, pet, f, smax, qmax):

        f = f / 10  # (0, 0.1)
        smax = smax * 1500  # (100, 1500)
        qmax = qmax * 50  # (10, 50)

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)
        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        # q = f((qsurb + qsurf, xt, p)) + n (xt, w, b)

        return [et, qsub, qsurf]

    def step_do(self, step_in, states):
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]


        [_ps, _pr] = self.rainsnowpartition(p, t, self.tmin)

        _m = self.snowbucket(s0, t, self.ddf, self.tmax)

        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        # _q = f((_qsurb + _qsurf, xt, p)) + ( n (xt, w, b)  - 观测量  ）
        #  q  = f((_qsurb + _qsurf, xt, p))+ NN ( 物理部分 -  观测 )

        # Water balance equations
        _ds0 = _ps - _m
        # _ds1 = _pr + _m - _et - _q
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]


    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, pet), axis=-1)

        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # Recursively calculate state variables by using RNN
        # return 3 outputs:
        # last_output (the latest output of the rnn, through last time g() & *V & +b)
        # output (all outputs [wrap_number_train, wrap_length, output], through all time g() & *V & +b)
        # new_states(latest states returned by the step_do function, without through last time g() & *V & +b)
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        # Calculate final process variables
        m = self.snowbucket(s0, tmean, self.ddf, self.tmax)
        [et, qsub, qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)


        if self.mode == "normal":
            return K.concatenate([m, et, qsurf, qsub], axis=-1)
        elif self.mode == "analysis":
            return K.concatenate([s0, m, et, qsurf, qsub, s1], axis=-1)


    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 4)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 6)

class ScaleLayer(Layer):


    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, inputs):
        #met = [wrap_number_train, wrap_length, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        print("***************:",inputs.shape)
        prec = inputs[:, :, 0:1]

        t = inputs[:,:,1:2]

        other_met = inputs[:,:,2:-4]

        #flow= [wrap_number_train, wrap_length, 1('Q(mm)')]
        flow = inputs[:, :, -4:]
        print("flow_calculatedby_fir_rnncel:",flow)

        #[wrap_number_train, 1, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        self.prec_center = K.mean(prec, axis=-2, keepdims=True)

        #[wrap_number_train, 1, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        self.prec_scale = K.std(prec, axis=-2, keepdims=True)

        #[wrap_number_train,  wrap_length, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        self.prec_scaled = (prec - self.prec_center) / self.prec_scale

        #[wrap_number_train, 1, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        self.other_met_center = K.mean(other_met, axis=-2, keepdims=True)

        #[wrap_number_train, 1, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        self.other_met_scale = K.std(other_met, axis=-2, keepdims=True)

        #[wrap_number_train,  wrap_length, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        self.other_met_scaled = (other_met - self.other_met_center) / self.other_met_scale

        #self.flow_center = K.mean(flow, axis=-2, keepdims=True)
        #self.flow_scale = K.std(flow, axis=-2, keepdims=True)
        #self.flow_scaled = (flow - self.flow_center) / self.flow_scale

        return K.concatenate([self.prec_scaled,t, self.other_met_scaled, flow], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

class LSTM_Layer(Layer):
    def __init__(self, input_xd, hidden_size, seed=200, **kwargs):
        self.input_xd = input_xd
        self.hidden_size = hidden_size
        self.seed = seed
        super(LSTM_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_ih = self.add_weight(name='w_ih', shape=(self.input_xd, 4 * self.hidden_size),
                                    initializer=initializers.Orthogonal(seed=self.seed - 5),
                                    trainable=True)

        self.w_hh = self.add_weight(name='w_hh',
                                    shape=(self.hidden_size, 4 * self.hidden_size),
                                    initializer=initializers.Orthogonal(seed=self.seed + 5),
                                    trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(4 * self.hidden_size,),
                                    # initializer = 'random_normal',
                                    initializer=initializers.Constant(value=0),
                                    trainable=True)

        self.shape = input_shape
        self.reset_parameters()
        super(LSTM_Layer, self).build(input_shape)

    def reset_parameters(self):
        # self.w_ih.initializer = initializers.Orthogonal(seed=self.seed - 5)
        # self.w_sh.initializer = initializers.Orthogonal(seed=self.seed + 5)

        w_hh_data = K.eye(self.hidden_size)
        # bias_s_batch = K.repeat_elements(bias_s_batch, rep=sample_size_d, axis=0)
        w_hh_data = K.repeat_elements(w_hh_data, rep=4, axis=1)
        self.w_hh = w_hh_data

        # self.bias.initializer = initializers.Constant(value=0)
        # self.bias_s.initializer = initializers.Constant(value=0)

    def call(self, inputs_x):
        forcing = inputs_x  # [batch, seq_len, dim]
        print('forcing_shape:', forcing.shape)
        # attrs = inputs_x[1]     #[batch, dim]
        # print('attrs_shape:',attrs.shape)

        forcing_seqfir = K.permute_dimensions(forcing, pattern=(1, 0, 2))  # [seq_len, batch, dim]
        print('forcing_seqfir_shape:', forcing_seqfir.shape)

        # attrs_seqfir = K.permute_dimensions(attrs, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        # print('attrs_seqfir_shape:',attrs_seqfir.shape)

        seq_len = forcing_seqfir.shape[0]
        print('seq_len:', seq_len)
        batch_size = forcing_seqfir.shape[1]
        print('batch_size:', batch_size)

        # init_states = [K.zeros((K.shape(forcing)[0], 2))]
        # h0, c0 = [K.zeros(shape= (sample_size_d,self.hidden_size)),K.zeros(shape= (sample_size_d,self.hidden_size))]
        h0 = K.zeros(shape=(batch_size, self.hidden_size))
        c0 = K.zeros(shape=(batch_size, self.hidden_size))
        h_x = (h0, c0)

        h_n, c_n = [], []

        bias_batch = K.expand_dims(self.bias, axis=0)
        bias_batch = K.repeat_elements(bias_batch, rep=batch_size, axis=0)
        print("bias_batch:", bias_batch.shape)

        # bias_s_batch = K.expand_dims(self.bias_s, axis=0)
        # bias_s_batch = K.repeat_elements(bias_s_batch, rep=batch_size, axis=0)

        # i = K.sigmoid(K.dot(attrs, self.w_sh) + bias_s_batch)

        for t in range(seq_len):
            h_0, c_0 = h_x


            gates = ((K.dot(h_0, self.w_hh) + bias_batch) + K.dot(forcing_seqfir[t], self.w_ih))
            f, i, o, g = tf.split(value=gates, num_or_size_splits=4, axis=1)

            next_c = K.sigmoid(f) * c_0 + K.sigmoid(i) * K.tanh(g)
            next_h = K.sigmoid(o) * K.tanh(next_c)

            h_n.append(next_h)
            c_n.append(next_c)

            h_x = (next_h, next_c)

        h_n = K.stack(h_n, axis=0)
        c_n = K.stack(c_n, axis=0)

        return h_n, c_n
































