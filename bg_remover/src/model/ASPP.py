import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation,  Concatenate

from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D

def ASPP(inputs):
    shape=inputs.shape
    y1=AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y1=Conv2D(256,1,padding="same",use_bias="False")(y1)
    y1=BatchNormalization()(y1)
    y1=Activation("relu")(y1)
    y1=UpSampling2D((shape[1], shape[2]))(y1)
    
    y2=Conv2D(256,1,use_bias="False",padding="same")(inputs)
    y2=BatchNormalization()(y2)
    y2=Activation("relu")(y2)

    y3=Conv2D(256,3,use_bias="False",padding="same",dilation_rate=6)(inputs)
    y3=BatchNormalization()(y3)
    y3=Activation("relu")(y3)

    y4=Conv2D(256,3,use_bias="False",padding="same",dilation_rate=12)(inputs)
    y4=BatchNormalization()(y4)
    y4=Activation("relu")(y4)

    y5=Conv2D(256,3,use_bias="False",padding="same",dilation_rate=18)(inputs)
    y5=BatchNormalization()(y5)
    y5=Activation("relu")(y5)

    y6=Concatenate()([y1,y2,y3,y4,y5])
    y6=Conv2D(256,1,padding="same",use_bias="False")(y6)
    y6=BatchNormalization()(y6)
    y6=Activation("relu")(y6)
     
    
    return y6 
    