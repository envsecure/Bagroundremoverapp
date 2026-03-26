import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Concatenate, Input

from tensorflow.keras.layers import  UpSampling2D

from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

import tensorflow as tf
from .ASPP import ASPP
from .SAE import SqueezeAndExcite

def deeplabv3_plus(shape):
    inputs = Input(shape)
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    image_features = encoder.get_layer("conv4_block6_out").output
    x_a =ASPP(image_features)
    x_a=UpSampling2D((4,4),interpolation="bilinear")(x_a)
    
    x_b = encoder.get_layer("conv2_block2_out").output
    X_b=Conv2D(filters=48,kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b=BatchNormalization()(x_b)
    x_b=Activation("relu")(x_b)
    
    x=Concatenate()([x_a,x_b])
    x=Conv2D(256,3,use_bias="False",padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=SqueezeAndExcite(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, 1)(x)
    x = Activation("sigmoid")(x)
    model = Model(inputs, x)
    return model
