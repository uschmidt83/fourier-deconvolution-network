from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np

from keras.models import Model
from keras.layers import Input, Lambda, Dense, Conv2D
from layers import Pad, Crop, FourierDeconvolution


def _get_inputs(img_shape=(None,None,1),kernel_shape=(None,None)):
    x_in = Input(shape=img_shape,    name="x_in")
    y    = Input(shape=img_shape,    name="y")
    k    = Input(shape=kernel_shape, name="k")
    s    = Input(shape=(1,),         name="s")
    return x_in, y, k, s


def model_stage(stage):
    assert 1 <= stage
    x_in, y, k, s = _get_inputs()

    # MLP for noise-adaptive regularization weight
    layer = Lambda(lambda u: 1/(u*u), name="1_over_s_squared")(s)
    for i in range(3):
        layer = Dense(16, activation='elu', name="dense%d"%(i+1))(layer)
    lamb = Dense(1, activation='softplus', name="lambda")(layer)

    # CNN for regularization in numerator
    layer = Pad(20, 'REPLICATE', name='x_in_padded')(x_in)
    nconvs = 5
    for i in range(nconvs):
        layer = Conv2D(32, (3,3), activation='elu', padding='same', name='conv%d'%(i+1))(layer)
    layer = Conv2D(1, (3,3), activation='linear', padding='same', name='conv%d'%(nconvs+1))(layer)
    x_adjustment = Crop(20, name='x_adjustment')(layer)

    # FFT-based update equation (also contains linear filters in denominator)
    x_out = FourierDeconvolution((5,5), stage, name="x_out")([x_in, x_adjustment, y, k, lamb])

    return Model([x_in, y, k, s], x_out)


def model_stacked(n_stages,weights=None):
    assert weights is None or len(weights) == n_stages
    x0, y, k, s = _get_inputs()

    if n_stages == 1:
        m = model_stage(1)
        m.load_weights(weights[0])
        return m
    else:
        outputs = []
        for t in range(n_stages):
            stage = model_stage(t+1)
            if weights is not None:
                stage.load_weights(weights[t])
            outputs.append(stage([(outputs[-1] if t>0 else x0), y, k, s]))

        return Model([x0, y, k, s], outputs)