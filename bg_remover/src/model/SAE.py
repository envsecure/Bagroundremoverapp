import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import  GlobalAveragePooling2D
from tensorflow.keras.layers import Dense,Reshape,Multiply

def SqueezeAndExcite(inputs,ratio=8):
    b,_,_,channel=inputs.shape
    se_shape = (1, 1, channel)
    x=GlobalAveragePooling2D()(inputs)
    se = Reshape(se_shape)(x)
    dep1=Dense(channel//ratio,activation="relu",use_bias=False)(se)
    dep2=Dense(channel,activation="sigmoid",use_bias=False)(dep1)
    ans=Multiply()([inputs,dep2])
    return ans