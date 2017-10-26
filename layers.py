from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from scipy.fftpack import idct


# create DCT filter basis as numpy matrix, one filter per column
def dct_filters(filter_size=(3,3)):
    N = filter_size[0] * filter_size[1]
    filters = np.zeros((N,N-1), np.float32)
    for i in range(1,N):
        d = np.zeros(filter_size, np.float32)
        d.flat[i] = 1
        filters[:, i-1] = idct(idct(d, norm='ortho').T, norm='ortho').real.flatten()
    return filters


# tensorflow: convert PSFs to OTFs
# psf: tensor with shape [height, width, channels_in, channels_out]
# img_shape: pair of integers
def psf2otf(psf, img_shape):
    # shape and type of the point spread function(s)
    psf_shape = tf.shape(psf)
    psf_type = psf.dtype

    # coordinates for 'cutting up' the psf tensor
    midH = tf.floor_div(psf_shape[0], 2)
    midW = tf.floor_div(psf_shape[1], 2)

    # slice the psf tensor into four parts
    top_left     = psf[:midH, :midW, :, :]
    top_right    = psf[:midH, midW:, :, :]
    bottom_left  = psf[midH:, :midW, :, :]
    bottom_right = psf[midH:, midW:, :, :]

    # prepare zeros for filler
    zeros_bottom = tf.zeros([psf_shape[0] - midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)
    zeros_top    = tf.zeros([midH, img_shape[1] - psf_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)

    # construct top and bottom row of new tensor
    top    = tf.concat([bottom_right, zeros_bottom, bottom_left], 1)
    bottom = tf.concat([top_right,    zeros_top,    top_left],    1)

    # prepare additional filler zeros and put everything together
    zeros_mid = tf.zeros([img_shape[0] - psf_shape[0], img_shape[1], psf_shape[2], psf_shape[3]], dtype=psf_type)
    pre_otf = tf.concat([top, zeros_mid, bottom], 0)
    # output shape: [img_shape[0], img_shape[1], channels_in, channels_out]

    # fast fourier transform, transposed because tensor must have shape [..., height, width] for this
    otf = tf.fft2d(tf.cast(tf.transpose(pre_otf, perm=[2,3,0,1]), tf.complex64))

    # output shape: [channels_in, channels_out, img_shape[0], img_shape[1]]
    return otf


class Pad(Layer):
    def __init__(self, border=0, mode='REPLICATE', **kwargs):
        assert border >= 0
        assert mode in ['REPLICATE', 'ZEROS']
        self.border = border
        self.mode = mode
        super(Pad, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2*self.border if input_shape[1] is not None else None,
                input_shape[2] + 2*self.border if input_shape[2] is not None else None,
                input_shape[3])

    def call(self, x, mask=None):
        if self.mode == 'REPLICATE':
            # hack: iterate 1-pixel symmetric padding to get replicate padding
            for i in range(self.border):
                x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], "SYMMETRIC")
        elif self.mode == 'ZEROS':
            x = tf.pad(x, [[0,0], [self.border, self.border], [self.border, self.border], [0,0]], "CONSTANT")
        return x


class Crop(Layer):
    def __init__(self, border=0, **kwargs):
        assert border >= 0
        self.border = border
        super(Crop, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] - 2*self.border if input_shape[1] is not None else None,
                input_shape[2] - 2*self.border if input_shape[2] is not None else None,
                input_shape[3])

    def call(self, x, mask=None):
        return x[:, self.border:-self.border, self.border:-self.border, :] if self.border > 0 else x


class FourierDeconvolution(Layer):
    def __init__(self, filter_size, stage, **kwargs):
        self.filter_size = filter_size
        self.stage = stage
        super(FourierDeconvolution, self).__init__(**kwargs)

    def build(self, input_shapes):
        # construct filter basis B and define filter weights variable
        B = dct_filters(self.filter_size)
        self.B = K.variable(B, name='B', dtype='float32')
        self.nb_filters = B.shape[1]
        self.filter_weights = K.variable(np.eye(self.nb_filters), name='filter_weights', dtype='float32')
        self.trainable_weights = [self.filter_weights]
        super(FourierDeconvolution, self).build(input_shapes)

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs, mask=None):
        padded_inputs, adjustments, observations, blur_kernels, lambdas = inputs

        imagesize = tf.shape(padded_inputs)[1:3]
        kernelsize = tf.shape(blur_kernels)[1:3]
        padding = tf.floor_div(kernelsize,2)

        mask_int = tf.ones((imagesize[0]-2*padding[0],imagesize[1]-2*padding[1]), dtype=tf.float32)
        mask_int = tf.pad(mask_int, [[padding[0],padding[0]],[padding[1],padding[1]]], mode='CONSTANT')
        mask_int = tf.expand_dims(mask_int, 0)

        filters = tf.matmul(self.B, self.filter_weights)
        filters = tf.reshape(filters, [self.filter_size[0],self.filter_size[1],1,self.nb_filters])

        filter_otfs = psf2otf(filters, imagesize)
        otf_term = tf.reduce_sum(tf.square(tf.abs(filter_otfs)), axis=1)

        k = tf.expand_dims(tf.transpose(blur_kernels, [1,2,0]), -1)
        k_otf = psf2otf(k, imagesize)[:,0,:,:]

        if self.stage > 1:
            # boundary adjustment
            Kx_fft       = tf.fft2d(tf.cast(padded_inputs[:,:,:,0], tf.complex64)) * k_otf
            Kx           = tf.to_float(tf.ifft2d(Kx_fft))
            Kx_outer     = (1.0 - mask_int) * Kx
            y_inner      = mask_int * observations[:,:,:,0]
            y_adjusted   = y_inner + Kx_outer
            dataterm_fft = tf.fft2d(tf.cast(y_adjusted, tf.complex64)) * tf.conj(k_otf)
        else:
            # standard data term
            observations_fft = tf.fft2d(tf.cast(observations[:,:,:,0], tf.complex64))
            dataterm_fft = observations_fft * tf.conj(k_otf)

        lambdas = tf.expand_dims(lambdas, -1)

        adjustment_fft = tf.fft2d(tf.cast(adjustments[:,:,:,0], tf.complex64))
        numerator_fft  = tf.cast(lambdas, tf.complex64) * dataterm_fft + adjustment_fft

        KtK = tf.square(tf.abs(k_otf))
        denominator_fft = lambdas * KtK + otf_term
        denominator_fft = tf.cast(denominator_fft, tf.complex64)

        frac_fft = numerator_fft / denominator_fft
        return tf.expand_dims(tf.to_float(tf.ifft2d(frac_fft)), -1)