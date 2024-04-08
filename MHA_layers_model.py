# DEEP cONV NET USING OPTIMAL N-KERNEL CONV3D LAYERS WITH INTERMEDIATE DENSE LAYER FEATURES CONCAT'ED AND MHA (ATTENTION)(NEW)
# - INTER-LAYER SHARING/CONNECTIONS BETWEEN DUAL STREAMS
# RECORDED ACC ON CLIPPED DATA: 74% - 83%

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.regularizers import l1, l2, l1_l2
from sklearn.utils import class_weight
from tensorflow.keras import mixed_precision

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

#mixed_precision.set_global_policy('mixed_float16')

#one-hot code labels
lb = LabelBinarizer()
le = LabelEncoder()
choice = '3d'
temporal_length,img_rows, img_cols, colors, colors_2 = 8,128,128,3,3

#x1 = tf.keras.layers.Lambda(CustomAugment())(x1)
class CustomAugment(object):
    def __call__(self, image):        
        # Random flips and grayscale with some stochasticity
        img = self._random_apply(tf.image.flip_left_right, image, p=0.6)
        img = self._random_apply(self._color_drop, img, p=0.9)
        return img

    def _color_drop(self, x):
        image = tf.image.rgb_to_grayscale(x)
        image = tf.tile(x, [1, 1, 1, 3])
        return x
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)

def Fully_connected(x, hidden_units, dropout_rate, l_func=tf.nn.leaky_relu):
    for units in hidden_units:
        x = Dense(units, activation=l_func, activity_regularizer=l2(0.0002))(x)
        x = Dropout(dropout_rate)(x)
    return x

def attention(x, num_heads=1, num=4, embed_dim=64, hidden_units=128):
    for _ in range(num):
        # Layer normalization 1.
        x = LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.2)(x, x)
        # Skip connection 1.
        x2 = Add()([attention_output, x])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = Fully_connected(x3, hidden_units=[hidden_units], dropout_rate=0.1)
        # Skip connection 2.
        x = Concatenate()([x3, x2])
    return x

def Squeeze_excitation_layer(input_x, out_dim=80):
            squeeze = GlobalAveragePooling3D()(input_x)
            excitation = Fully_connected(squeeze, [out_dim*2], 0.2)
            excitation = Fully_connected(excitation, [out_dim], 0.2,)
            excitation = tf.reshape(excitation, [-1,1,1,1,out_dim])
            scale = Multiply()([input_x, excitation])
            return scale
        
def attention_2(x,x_, num_heads=1, num=3, embed_dim=64, hidden_units=128):
    for _ in range(num):
        # Layer normalization 1.
        x = BatchNormalization()(x)
        x_ = BatchNormalization()(x_)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.3)(x,x_)
        # Skip connection 1.
        x2 = Concatenate()([x, attention_output, x_]);# x2 = Concatenate()([x2, x])
        #x2 = Squeeze_excitation_layer(attention_output, out_dim=int(attention_output.shape[-1]))
        # Layer normalization 2.
        x3 = BatchNormalization()(x2)
        # MLP.
        #x3 = Fully_connected(x3, hidden_units=[embed_dim], dropout_rate=0.25)
        # Skip connection 2.
        x = Concatenate()([x3, attention_output])
        x,x_ = x, x
    return x

def incept_blkv3(x_o, filter):
    x_ = Conv3D(filter, 1, strides=1, padding='same', activation=tf.nn.relu, )(x_o)
    x = Conv3D(filter, (1,1,7), strides=1, padding='same', activation=tf.nn.relu, )(x_)
    x = Conv3D(filter, (7,1,1), strides=1, padding='same', activation=tf.nn.relu, )(x)
    x1 = Conv3D(filter, 3, strides=1, padding='same', activation=tf.nn.relu, )(x)
    x2 = Conv3D(filter, 3, strides=1, padding='same', activation=tf.nn.relu, )(x_)
    x3 = MaxPooling3D(3, strides=1, padding='same')(x_o)
    x = Concatenate()([x1,x2,x3])
    return x

norm_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
norm_layer_2 = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)

def AE_model_3_4(fusiontype='late_fusion', method='concat', norm_layer=norm_layer, norm_layer_2=norm_layer_2): #not working yet..........................
  input_shape = (temporal_length, img_rows, img_cols, colors)
  input_shape_2 = (temporal_length, img_rows, img_cols, colors_2)
  input = Input(shape=input_shape)
  input_2 = Input(shape=input_shape_2)
  scale_, offset = 1./255., 0
  # method 1: late fusion of multi kernel network
  
  if fusiontype == 'late_fusion':
     x1=input
     x1= Cropping3D((0,16,0))(x1)
     #x1 = tf.keras.layers.Lambda(CustomAugment())(x1)
     #x1 = tf.keras.layers.experimental.preprocessing.Rescaling(scale_,offset)(x1)
     x1 = norm_layer (x1)
        
     # second stream
     x2=input_2
     x2= Cropping3D((0,16,0))(x2)
     #x2 = tf.keras.layers.Lambda(CustomAugment())(x2)
     x2 = tf.keras.layers.experimental.preprocessing.Rescaling(scale_,offset)(x2); 
     #x2 = norm_layer_2 (x2); opt_flo=x2
    
     #==========================================================================================================================
        
     filter, n = 16, 7
     
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n-2, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     
    
     #'''#filter, n = 128, 7
     x2 = Conv3D(filter, n-2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2); print(x1.shape, x2.shape)#'''
    
     #'''
     #x1 = attention_2(x1,x1, num_heads=1, num=1, hidden_units=int(filter))
     #x2 = attention_2(x2,x2, num_heads=1, num=1, hidden_units=int(filter))    
     x = Add()([x1, x2]); #x=Fully_connected(x, [filter], 0.2, l_func=tf.nn.leaky_relu)  #x = ConvLSTM2D(x.shape[-1], 1, 1, activation=tf.nn.leaky_relu, return_sequences=True)(x); x = BatchNormalization()(x)
     x = incept_blkv3(x, int(filter*1.5))
     #x = attention_2(x,x, num_heads=1, num=1, hidden_units=int(filter*1.5))#'''
     x1, x2 = x, x
    
     #==========================================================================================================================
    
     filter, n = 32, 5
     
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n-2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     
    
     #'''#filter, n = 64, 5
     x2 = Conv3D(filter, n-2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
    
     #x1 = attention_2(x1,x1, num_heads=1, num=1, hidden_units=filter)
     #x2 = attention_2(x2,x2, num_heads=1, num=1, hidden_units=filter)
     x = Add()([x1, x2]); #x=Fully_connected(x, [filter], 0.2, l_func=tf.nn.leaky_relu)
     x = incept_blkv3(x, int(filter*1.5))
     x1, x2 = x, x 
        
     #==========================================================================================================================
       
     filter, n = 64, 3
     
     #x1 = Conv3D(int(filter*1.5), n, strides=(1,2,2), padding='same', activation=tf.nn.gelu)(x1); #x1 = Dropout(0.2)(x1)
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu,)(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n+2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     

     #'''#filter, n = 32, 3
     x2 = Conv3D(filter, n+2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
        
     #'''
     #x1 = attention_2(x1,x1, num_heads=1, num=1, hidden_units=int(filter))
     #x2 = attention_2(x2,x2, num_heads=1, num=1, hidden_units=int(filter)) 
     x = Add()([x1, x2]); #x=Fully_connected(x, [filter], 0.2, l_func=tf.nn.leaky_relu)  #x = ConvLSTM2D(x.shape[-1], 1, 1, activation=tf.nn.leaky_relu, return_sequences=True)(x); x = BatchNormalization()(x)
     x = incept_blkv3(x, int(filter*1.5))
     #x = attention_2(x,x, num_heads=1, num=1, hidden_units=int(filter*1.5))#'''
     x1, x2 = x, x
        
     #==========================================================================================================================
       
     filter, n = 128, 1
    
     x1 = Conv3D(filter, n+2, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
    
     #---------------------------------------------------------------------------------------------------------------     
    
     #'''#filter, n = 16, 3
     x2 = Conv3D(filter, n, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu,)(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n+2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
        
     x1 = attention_2(x1,x1, num_heads=1, num=1, hidden_units=filter)
     x2 = attention_2(x2,x2, num_heads=1, num=1, hidden_units=filter)
     x = Add()([x1, x2]); #x=Fully_connected(x, [filter,], 0.2, l_func=tf.nn.leaky_relu) 
     x = incept_blkv3(x, int(filter*1.5))
     #x = attention_2(x,x, num_heads=1, num=1, hidden_units=int(filter*1.5))
     #==========================================================================================================================

     #x = Reshape([x.shape[1]*x.shape[2]*x.shape[3],x.shape[4]])(x); x = Bidirectional(LSTM(512,))(x); #(256,)
     
     # fusion and decoding
     if method == 'concat':
        #x = BatchNormalization()(x)
        #x = Flatten()(x); 
        x= Flatten()(x)
        x = Dropout(0.2)(x)

        mergedOutput = Dense(512, activation=tf.nn.leaky_relu)(x) #256
        #mergedOutput = Dropout(0.2)(mergedOutput)
        
        mergedOutput = Dense(64, activation=tf.nn.leaky_relu)(mergedOutput) #32
        #mergedOutput = Dropout(0.2)(mergedOutput)
        
        mergedOutput = Dense(8, activation=tf.nn.softmax)(mergedOutput)
        #mergedOutput = tf.keras.layers.Dense(8)(mergedOutput)
        #mergedOutput = tf.keras.layers.Activation('softmax', dtype='float32')(mergedOutput)
        model = models.Model(inputs=[input,input_2], outputs=mergedOutput) 

     model.summary()
     return model
    
def AE_model_3_4_5(fusiontype='late_fusion', method='concat'): #not working yet..........................
  input_shape = (temporal_length, img_rows, img_cols, colors)
  input_shape_2 = (temporal_length, img_rows, img_cols, colors_2)
  input = Input(shape=input_shape)
  input_2 = Input(shape=input_shape_2)
  scale_, offset = 1./255., 0
  # method 1: late fusion of multi kernel network
  
  if fusiontype == 'late_fusion':
     x1=input
     x1= Cropping3D((0,16,0))(x1)
     #x1 = tf.keras.layers.Lambda(CustomAugment())(x1)
     x1 = tf.keras.layers.experimental.preprocessing.Rescaling(scale_,offset)(x1)
        
     # second stream
     x2=input_2
     x2= Cropping3D((0,16,0))(x2)
     #x2 = tf.keras.layers.Lambda(CustomAugment())(x2)
     x2 = tf.keras.layers.experimental.preprocessing.Rescaling(scale_,offset)(x2); opt_flo=x2
    
     #==========================================================================================================================
        
     filter, n = 16, 7
     
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n-2, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     
    
     #'''#filter, n = 128, 7
     x2 = Conv3D(filter, n-2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2); print(x1.shape, x2.shape)#'''
    
     #'''
     x1 = incept_blkv3(x1, int(filter*1.5))
     x2 = incept_blkv3(x2, int(filter*1.5))
    
     #==========================================================================================================================
    
     filter, n = 32, 5
     
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n-2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     
    
     #'''#filter, n = 64, 5
     x2 = Conv3D(filter, n-2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
    
     x1 = incept_blkv3(x1, int(filter*1.5))
     x2 = incept_blkv3(x2, int(filter*1.5))
        
     #==========================================================================================================================
       
     filter, n = 64, 3
     
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu,)(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n+2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     

     #'''#filter, n = 32, 3
     x2 = Conv3D(filter, n+2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
        
     x1 = incept_blkv3(x1, int(filter*1.5))
     x2 = incept_blkv3(x2, int(filter*1.5))
        
     #==========================================================================================================================
       
     filter, n = 128, 1
    
     x1 = Conv3D(filter, n+2, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
    
     #---------------------------------------------------------------------------------------------------------------     
    
     x2 = Conv3D(filter, n, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu,)(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n+2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
        
     x1 = attention_2(x1,x1, num_heads=1, num=1, hidden_units=filter)
     x2 = attention_2(x2,x2, num_heads=1, num=1, hidden_units=filter)
     x1 = incept_blkv3(x1, int(filter*1.5))
     x2 = incept_blkv3(x2, int(filter*1.5))
        
     x = Add()([x1, x2]); #x=Fully_connected(x, [filter,], 0.2, l_func=tf.nn.leaky_relu) 

     #==========================================================================================================================
     
     # fusion and decoding
     if method == 'concat':
        #x = BatchNormalization()(x)
        #x = Flatten()(x); 
        x= Flatten()(x)
        x = Dropout(0.2)(x)

        mergedOutput = Dense(512, activation=tf.nn.leaky_relu)(x) #256
        #mergedOutput = Dropout(0.2)(mergedOutput)
        
        mergedOutput = Dense(64, activation=tf.nn.leaky_relu)(mergedOutput) #32
        #mergedOutput = Dropout(0.2)(mergedOutput)
        
        mergedOutput = Dense(8, activation=tf.nn.softmax)(mergedOutput)
        #mergedOutput = tf.keras.layers.Dense(8)(mergedOutput)
        #mergedOutput = tf.keras.layers.Activation('softmax', dtype='float32')(mergedOutput)
        model = models.Model(inputs=[input,input_2], outputs=mergedOutput) 

     model.summary()
     return model
    
def AE_model_3_4_6(fusiontype='late_fusion', method='concat'): #not working yet..........................
  input_shape = (temporal_length, img_rows, img_cols, colors)
  input_shape_2 = (temporal_length, img_rows, img_cols, colors_2)
  input = Input(shape=input_shape)
  input_2 = Input(shape=input_shape_2)
  scale_, offset = 1./255., 0
  # method 1: late fusion of multi kernel network
  
  if fusiontype == 'late_fusion':
     x1=input
     x1= Cropping3D((0,16,0))(x1)
     #x1 = tf.keras.layers.Lambda(CustomAugment())(x1)
     x1 = tf.keras.layers.experimental.preprocessing.Rescaling(scale_,offset)(x1)
        
     # second stream
     x2=input_2
     x2= Cropping3D((0,16,0))(x2)
     #x2 = tf.keras.layers.Lambda(CustomAugment())(x2)
     x2 = tf.keras.layers.experimental.preprocessing.Rescaling(scale_,offset)(x2); opt_flo=x2
    
     #==========================================================================================================================
        
     filter, n = 16, 7
     
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n-2, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     
    
     #'''#filter, n = 128, 7
     x2 = Conv3D(filter, n-2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2); print(x1.shape, x2.shape)#'''
    
     #'''
     #x1 = incept_blkv3(x1, int(filter*1.5))
     #x2 = incept_blkv3(x2, int(filter*1.5))
    
     #==========================================================================================================================
    
     filter, n = 32, 5
     
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n-2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     
    
     #'''#filter, n = 64, 5
     x2 = Conv3D(filter, n-2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
    
     #x1 = incept_blkv3(x1, int(filter*1.5))
     #x2 = incept_blkv3(x2, int(filter*1.5))
        
     #==========================================================================================================================
       
     filter, n = 64, 3
     
     x1 = Conv3D(filter, n, strides=2, padding='same', activation=tf.nn.leaky_relu,)(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n+2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
     #---------------------------------------------------------------------------------------------------------------     

     #'''#filter, n = 32, 3
     x2 = Conv3D(filter, n+2, strides=2, padding='same', activation=tf.nn.leaky_relu, )(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
        
     #x1 = incept_blkv3(x1, int(filter*1.5))
     #x2 = incept_blkv3(x2, int(filter*1.5))
        
     #==========================================================================================================================
       
     filter, n = 128, 1
    
     x1 = Conv3D(filter, n+2, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu, )(x1)
     x1 = Dropout(0.2)(x1)
     x1 = Conv3D(int(filter*1.5), n, strides=1, padding='same', activation=tf.nn.leaky_relu)(x1) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x1 = BatchNormalization()(x1); #print(x1.shape, x2.shape)#'''
    
     #---------------------------------------------------------------------------------------------------------------     
    
     x2 = Conv3D(filter, n, strides=(1,2,2), padding='same', activation=tf.nn.leaky_relu,)(x2)
     x2 = Dropout(0.2)(x2)
     x2 = Conv3D(int(filter*1.5), n+2, strides=1, padding='same', activation=tf.nn.leaky_relu)(x2) #TimeDistributed(SeparableConv2D(int(filter*1.5), (n,n), strides=(2,2), padding='same', activation=tf.nn.leaky_relu))(x2)#
     x2 = BatchNormalization()(x2)#'''
        
     x1 = attention_2(x1,x1, num_heads=1, num=1, hidden_units=filter)
     x2 = attention_2(x2,x2, num_heads=1, num=1, hidden_units=filter)
     #x1 = incept_blkv3(x1, int(filter*1.5))
     #x2 = incept_blkv3(x2, int(filter*1.5))
        
     x = Add()([x1, x2]); #x=Fully_connected(x, [filter,], 0.2, l_func=tf.nn.leaky_relu) 

     #==========================================================================================================================
     
     # fusion and decoding
     if method == 'concat':
        #x = BatchNormalization()(x)
        #x = Flatten()(x); 
        x= Flatten()(x)
        x = Dropout(0.2)(x)

        mergedOutput = Dense(512, activation=tf.nn.leaky_relu)(x) #256
        #mergedOutput = Dropout(0.2)(mergedOutput)
        
        mergedOutput = Dense(64, activation=tf.nn.leaky_relu)(mergedOutput) #32
        #mergedOutput = Dropout(0.2)(mergedOutput)
        
        mergedOutput = Dense(8, activation=tf.nn.softmax)(mergedOutput)
        #mergedOutput = tf.keras.layers.Dense(8)(mergedOutput)
        #mergedOutput = tf.keras.layers.Activation('softmax', dtype='float32')(mergedOutput)
        model = models.Model(inputs=[input,input_2], outputs=mergedOutput) 

     model.summary()
     return model
    
###################################################-----###############################################################################################
model = AE_model_3_4(fusiontype='late_fusion', method='concat', norm_layer=norm_layer); name='supervised_3d_conv.h5'; bs = 32
##############################################====######################################################################

opt = optimizers.SGD(0.001) # 0.002-71% | 0.001 - 72% | 0.00075 - 71 | 0.0005 - 73

METRICS = ['accuracy'] #, tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(name='prc', curve='PR')]

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=METRICS)# loss=losses, loss_weights=lossWeights  # binary_crossentropy
model.summary()
