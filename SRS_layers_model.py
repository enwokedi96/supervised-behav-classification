    # DUAL STREAM CLASSIC INCEPTION BLOCKS VER. 1 (inception_3) AND VER. 2 (inception_4)
    # - INTER-LAYER SHARING/CONNECTIONS BETWEEN DUAL STREAMS
    # RECORDED ACC ON CLIPPED DATA: 76.97 (VER. 1) AND 77.63 (VER. 2)

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

    #mixed_precision.set_global_policy('mixed_float16')

    def Fully_connected(x, hidden_units, dropout_rate, l_func=tf.nn.relu):
        for units in hidden_units:
            x = Dense(units, activation=l_func)(x)
            x = Dropout(dropout_rate)(x)
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
            x3 = Fully_connected(x3, hidden_units=[hidden_units], dropout_rate=0.2)
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
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.2)(x,x_)
            # Skip connection 1.
            x2 = Concatenate()([attention_output, x_]); x2 = Concatenate()([x2, x])
            #x2 = Squeeze_excitation_layer(attention_output, out_dim=int(attention_output.shape[-1]))
            # Layer normalization 2.
            x3 = BatchNormalization()(x2)
            # MLP.
            x3 = Fully_connected(x3, hidden_units=[embed_dim], dropout_rate=0.2)
            # Skip connection 2.
            x = Concatenate()([x3, attention_output])
            x,x_ = x, x
        return x

    def inception_3(input_img,input_img_2,activation='relu',opt='Xtra_joint_processing'): # activation='relu'

      ### 1st layer
      layer_1 = Conv3D(8, 1, padding='same', activation=activation)(input_img)
      layer_1 = Conv3D(12, 3, padding='same', activation=activation)(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(8, 1, padding='same', activation=activation)(input_img)
      layer_2 = Conv3D(12, 5, padding='same', activation=activation)(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(input_img)
      layer_3 = Conv3D(12, 1, padding='same', activation=activation)(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(8, 1, padding='same', activation=activation)(input_img_2)
      layer_1_2 = Conv3D(12, 3, padding='same', activation=activation)(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(8, 1, padding='same', activation=activation)(input_img_2)
      layer_2_2 = Conv3D(12, 5, padding='same', activation=activation)(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(input_img_2)
      layer_3_2 = Conv3D(12, 1, padding='same', activation=activation)(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]); 
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 12)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=4, activation='relu')(mid)

      ### 2nd layer
      layer_1 = Conv3D(16, 1, padding='same', activation=activation)(mid)
      layer_1 = Conv3D(24, 3, padding='same', activation=activation)(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(16, 1, padding='same', activation=activation)(mid)
      layer_2 = Conv3D(24, 5, padding='same', activation=activation)(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(24, 1, padding='same', activation=activation)(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3]) #; print(mid_1.shape)

      mid_2 = Cropping3D((2,24,32))(input_img_2) #Cropping3D((2,24,32))# MaxPooling3D(3, strides=4, padding='same')(input_img_2)
      layer_1_1 = Conv3D(16, 1, padding='same', activation=activation)(mid_2)
      layer_1_2 = Conv3D(24, 3, padding='same',  activation=activation)(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(16, 1, padding='same', activation=activation)(mid_2)
      layer_2_2 = Conv3D(24, 5, padding='same', activation=activation)(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(24, 1, padding='same', activation=activation)(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) #; print(mid.shape); 
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 24)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid); print(mid.shape) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 3rd layer
      layer_1 = Conv3D(32, 1, padding='same', activation=activation)(mid)
      layer_1 = Conv3D(48, 3, padding='same', activation=activation)(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(32, 1, padding='same', activation=activation)(mid)
      layer_2 = Conv3D(48, 5, padding='same', activation=activation)(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(48, 1, padding='same', activation=activation)(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=4, padding='same')(input_img_2)
      layer_1_1 = Conv3D(32, 1, padding='same', activation=activation)(mid_2)
      layer_1_2 = Conv3D(48, 3, padding='same',  activation=activation)(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(32, 1, padding='same', activation=activation)(mid_2)
      layer_2_2 = Conv3D(48, 5, padding='same',  activation=activation)(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(48, 1, padding='same',  activation=activation)(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 48)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 4th layer
      layer_1 = Conv3D(64, 1, padding='same', activation=activation)(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation=activation)(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(64, 1, padding='same', activation=activation)(mid)
      layer_2 = Conv3D(96, 5, padding='same', activation=activation)(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation=activation)(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=8, padding='same')(input_img_2)
      layer_1_1 = Conv3D(64, 1, padding='same', activation=activation)(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation=activation)(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(64, 1, padding='same', activation=activation)(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation=activation)(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation=activation)(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 96)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 5th layer
      layer_1 = Conv3D(64, 1, padding='same', activation=activation)(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation=activation)(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(64, 1, padding='same', activation=activation)(mid)
      layer_2 = Conv3D(96 , 5, padding='same', activation=activation)(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation=activation)(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=16, padding='same')(input_img_2)
      layer_1_1 = Conv3D(64, 1, padding='same', activation=activation)(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation=activation)(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(64, 1, padding='same', activation=activation)(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation=activation)(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation=activation)(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 96)
      mid=MaxPooling3D(3, strides=2, padding='same')(mid) 

      #================================================================================================
      layer_1 = Conv3D(128, 1, padding='same', activation=activation)(mid)
      layer_1 = Conv3D(128, 3, padding='same', activation=activation)(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(128, 1, padding='same', activation=activation)(mid)
      layer_2 = Conv3D(128, 5, padding='same', activation=activation)(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(128, 1, padding='same', activation=activation)(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid = Concatenate()([layer_1, layer_2, layer_3])
     #
      return mid

    def inception_3_revised(input_img,input_img_2,opt='Xtra_joint_processing'):

      ### 1st layer
      layer_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img)
      layer_1 = Conv3D(12, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(8, 1, padding='same', activation='relu')(input_img)
      layer_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(input_img)
      layer_3 = Conv3D(12, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img_2)
      layer_1_2 = Conv3D(12, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img_2)
      layer_2_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(input_img_2)
      layer_3_2 = Conv3D(12, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]); 
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 12)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=4, activation='relu')(mid)

      ### 2nd layer
      layer_1 = Conv3D(16, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(24, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(16, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(24, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3]) #; print(mid_1.shape)

      mid_2 = MaxPooling3D(3, strides=2, padding='same')(input_img_2) #Cropping3D((2,24,32))(input_img_2) #Cropping3D((2,24,32))(input_img_2) # MaxPooling3D(3, strides=4, padding='same')(input_img_2)
      layer_1_1 = Conv3D(16, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(24, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(16, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(24, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) #; print(mid.shape); 
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 24)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid); print(mid.shape) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 3rd layer
      layer_1 = Conv3D(32, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(48, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(32, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(48, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(48, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=4, padding='same')(input_img_2)
      layer_1_1 = Conv3D(32, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(48, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(32, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(48, 5, padding='same',  activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(48, 1, padding='same',  activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 48)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 4th layer
      layer_1 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=8, padding='same')(input_img_2)
      layer_1_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 96)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 5th layer
      layer_1 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(96 , 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=16, padding='same')(input_img_2)
      layer_1_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': mid = incept_blkv3(mid, 96)
      mid=MaxPooling3D(3, strides=2, padding='same')(mid) 

      #================================================================================================
      layer_1 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(128, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(128, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(128, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid = Concatenate()([layer_1, layer_2, layer_3])
     #
      return mid

    def inception_3_5(input_img,input_img_2):

      ### 1st layer
      layer_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img)
      layer_1 = Conv3D(12, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(8, 1, padding='same', activation='relu')(input_img)
      layer_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(input_img)
      layer_3 = Conv3D(12, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img_2)
      layer_1_2 = Conv3D(12, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img_2)
      layer_2_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(input_img_2)
      layer_3_2 = Conv3D(12, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2]); print(mid_2.shape)

      mid = mid_1 # Concatenate()([mid_1,mid_2]); 
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=4, activation='relu')(mid)

      ### 2nd layer
      layer_1 = Conv3D(16, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(24, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(16, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(24, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3]) #; print(mid_1.shape)

      mid_2 = MaxPooling3D(3, strides=2, padding='same')(mid_2) #Cropping3D((2,24,32))(input_img_2) #Cropping3D((2,32,32))(input_img_2) # MaxPooling3D(3, strides=4, padding='same')(input_img_2)
      layer_1_1 = Conv3D(16, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(24, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(16, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(24, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = mid_1 # Concatenate()([mid_1,mid_2]) #; print(mid.shape); 
      mid = MaxPooling3D(3, strides=2, padding='same')(mid); print(mid.shape) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 3rd layer
      layer_1 = Conv3D(32, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(48, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(32, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(48, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(48, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=2, padding='same')(mid_2) #(input_img_2)
      layer_1_1 = Conv3D(32, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(48, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(32, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(48, 5, padding='same',  activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(48, 1, padding='same',  activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = mid_1#Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 4th layer
      layer_1 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=2, padding='same')(mid_2)#(input_img_2)
      layer_1_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = mid_1 # Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 5th layer
      layer_1 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(96 , 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=2, padding='same')(mid_2) #input_img_2)
      layer_1_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid=MaxPooling3D(3, strides=2, padding='same')(mid) 

      #================================================================================================
      layer_1 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(128, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(128, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(128, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid = Concatenate()([layer_1, layer_2, layer_3])
     #
      return mid


    def inception_4(input_img,input_img_2):

      ### 1st layer
      layer_1 = Conv3D(12, 1, padding='same', activation='relu')(input_img)
      layer_1 = Conv3D(12, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(12, 1, padding='same', activation='relu')(input_img)
      layer_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(input_img)
      layer_3 = Conv3D(12, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(12, 1, padding='same', activation='relu')(input_img_2)
      layer_1_2 = Conv3D(12, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(12, 1, padding='same', activation='relu')(input_img_2)
      layer_2_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(input_img_2)
      layer_3_2 = Conv3D(12, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]); 
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=4, activation='relu')(mid)

      ### 2nd layer
      layer_1 = Conv3D(24, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(24, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(24, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(24, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3]) #; print(mid_1.shape)

      mid_2 = MaxPooling3D(3, strides=2, padding='same')(input_img_2); mid_2 = Concatenate()([mid,mid_2]);
      layer_1_1 = Conv3D(24, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(24, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(24, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(24, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) #; print(mid.shape); 
      mid = MaxPooling3D(3, strides=2, padding='same')(mid); print(mid.shape) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 3rd layer
      layer_1 = Conv3D(48, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(48, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(48, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(48, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(48, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=4, padding='same')(input_img_2); mid_2 = Concatenate()([mid,mid_2]);
      layer_1_1 = Conv3D(48, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(48, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(48, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(48, 5, padding='same',  activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(48, 1, padding='same',  activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 4th layer
      layer_1 = Conv3D(96, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(96, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=8, padding='same')(input_img_2); mid_2 = Concatenate()([mid,mid_2]);
      layer_1_1 = Conv3D(96, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(96, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)

      ### 5th layer
      layer_1 = Conv3D(192, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(192, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(192, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(192 , 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(192, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=16, padding='same')(input_img_2); mid_2 = Concatenate()([mid,mid_2]);
      layer_1_1 = Conv3D(192, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(192, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(192, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(192, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(192, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid=MaxPooling3D(3, strides=2, padding='same')(mid) 

      #================================================================================================
      '''
      layer_1 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(128, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(128, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(128, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid = Concatenate()([layer_1, layer_2, layer_3])
     #'''
      return mid
    
    def inception_4_v2_1(input_img,input_img_2):

      ### 1st layer
      layer_1 = Conv3D(12, 1, padding='same', activation='relu')(input_img)
      layer_1 = Conv3D(12, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(12, 1, padding='same', activation='relu')(input_img)
      layer_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(input_img)
      layer_3 = Conv3D(12, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(12, 1, padding='same', activation='relu')(input_img_2)
      layer_1_2 = Conv3D(12, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(12, 1, padding='same', activation='relu')(input_img_2)
      layer_2_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(input_img_2)
      layer_3_2 = Conv3D(12, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = mid_1 # Concatenate()([mid_1,mid_2]); 
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=4, activation='relu')(mid)
      mid_2 = MaxPooling3D(3, strides=2, padding='same')(mid_2)

      ### 2nd layer
      layer_1 = Conv3D(24, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(24, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(24, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(24, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3]) #; print(mid_1.shape)

      layer_1_1 = Conv3D(24, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(24, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(24, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(24, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = mid_1 # Concatenate()([mid_1,mid_2]) #; print(mid.shape); 
      mid = MaxPooling3D(3, strides=2, padding='same')(mid); print(mid.shape) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)
      mid_2 = MaxPooling3D(3, strides=2, padding='same')(mid_2)

      ### 3rd layer
      layer_1 = Conv3D(48, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(48, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(48, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(48, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(48, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(48, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(48, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(48, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(48, 5, padding='same',  activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(48, 1, padding='same',  activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = mid_1 # Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)
      mid_2 = MaxPooling3D(3, strides=2, padding='same')(mid_2)

      ### 4th layer
      layer_1 = Conv3D(96, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(96, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(96, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(96, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = mid_1 #Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)
      mid_2 = MaxPooling3D(3, strides=2, padding='same')(mid_2)

      ### 5th layer
      layer_1 = Conv3D(192, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(192, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(192, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(192 , 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(192, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(192, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(192, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(192, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(192, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(192, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]) ; print(mid.shape)
      mid=MaxPooling3D(3, strides=2, padding='same')(mid) 

      #================================================================================================
      '''
      layer_1 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(128, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(128, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(128, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid = Concatenate()([layer_1, layer_2, layer_3])
     #'''
      return mid 
    
    def inception_3_residual(input_img,input_img_2,opt='Xtra_joint_processing'):

      ### 1st layer
      layer_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img)
      layer_1 = Conv3D(12, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(8, 1, padding='same', activation='relu')(input_img)
      layer_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(input_img)
      layer_3 = Conv3D(12, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      layer_1_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img_2)
      layer_1_2 = Conv3D(12, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(8, 1, padding='same', activation='relu')(input_img_2)
      layer_2_2 = Conv3D(12, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(input_img_2)
      layer_3_2 = Conv3D(12, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2]); 
      if opt=='Xtra_joint_processing': 
        #mid_ = incept_blkv3(mid, 12)
        mid_= attention(mid, num_heads=1, num=2, embed_dim=16, hidden_units=16); #mid_ = Concatenate()([mid_,mid__]);
        mid_= experimental.preprocessing.Rescaling(1,0)(mid_) 
        mid_ = MaxPooling3D(6, strides=2, padding='same')(mid_); skip = mid_
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=4, activation='relu')(mid)
        
      ### 2nd layer
      layer_1 = Conv3D(16, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(24, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(16, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(24, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3]) #; print(mid_1.shape)

      mid_2 = Cropping3D((2,24,32))(input_img_2) #Cropping3D((2,24,32))(input_img_2) # MaxPooling3D(3, strides=4, padding='same')(input_img_2)
      layer_1_1 = Conv3D(16, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(24, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(16, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(24, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(24, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2,skip]) #; print(mid.shape); 
      if opt=='Xtra_joint_processing': 
        #mid_ = incept_blkv3(mid, 24)
        mid_= attention(mid, num_heads=1, num=2, embed_dim=16, hidden_units=16);#mid_ = Concatenate()([mid_,mid__]);
        mid_=experimental.preprocessing.Rescaling(0.75,0)(mid_) 
        mid_ = MaxPooling3D(6, strides=2, padding='same')(mid_); skip = mid_
      mid = MaxPooling3D(3, strides=2, padding='same')(mid); print(mid.shape) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)
        
      ### 3rd layer
      layer_1 = Conv3D(32, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(48, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(32, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(48, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(48, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=4, padding='same')(input_img_2)
      layer_1_1 = Conv3D(32, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(48, 3, padding='same',  activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(32, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(48, 5, padding='same',  activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(48, 1, padding='same',  activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2,skip]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': 
        #mid_ = incept_blkv3(mid, 48)
        mid_= attention(mid, num_heads=1, num=2, embed_dim=16, hidden_units=16); #mid_ = Concatenate()([mid_,mid__]);
        mid_=experimental.preprocessing.Rescaling(0.5,0)(mid_) 
        mid_ = MaxPooling3D(6, strides=2, padding='same')(mid_); skip = mid_
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)
        
      ### 4th layer
      layer_1 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=8, padding='same')(input_img_2)
      layer_1_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2,skip]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': 
        #mid_ = incept_blkv3(mid, 96)
        mid_= attention(mid, num_heads=1, num=2, embed_dim=16, hidden_units=16); #mid_ = Concatenate()([mid_,mid__]);
        mid_=experimental.preprocessing.Rescaling(0.25,0)(mid_) 
        mid_ = MaxPooling3D(6, strides=2, padding='same')(mid_); skip = mid_
      mid = MaxPooling3D(3, strides=2, padding='same')(mid) #Conv3D(int(mid.shape[-1]), 1, padding='same', strides=2, activation='relu')(mid)
        
      ### 5th layer
      layer_1 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(96, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(64, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(96 , 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(96, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid_1 = Concatenate()([layer_1, layer_2, layer_3])

      mid_2 = MaxPooling3D(3, strides=16, padding='same')(input_img_2)
      layer_1_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_1_2 = Conv3D(96, 3, padding='same', activation='relu')(layer_1_1); #layer_1_2 = BatchNormalization()(layer_1_2)
      layer_2_1 = Conv3D(64, 1, padding='same', activation='relu')(mid_2)
      layer_2_2 = Conv3D(96, 5, padding='same', activation='relu')(layer_2_1); #layer_2_2 = BatchNormalization()(layer_2_2)
      layer_3_1 = MaxPooling3D(3, strides=1, padding='same')(mid_2)
      layer_3_2 = Conv3D(96, 1, padding='same', activation='relu')(layer_3_1); #layer_3_2 = BatchNormalization()(layer_3_2)
      mid_2 = Concatenate()([layer_1_2, layer_2_2, layer_3_2])

      mid = Concatenate()([mid_1,mid_2,skip]) ; print(mid.shape)
      if opt=='Xtra_joint_processing': 
        #mid_ = incept_blkv3(mid, 96)
        mid_= attention(mid, num_heads=1, num=2, embed_dim=16, hidden_units=16); #mid_ = Concatenate()([mid_,mid__]);
        mid_=experimental.preprocessing.Rescaling(1,0)(mid_) 
        mid_ = MaxPooling3D(6, strides=2, padding='same')(mid_); skip = mid_ # Concatenate()([mid,mid_])
      mid=MaxPooling3D(3, strides=2, padding='same')(mid) 

      #================================================================================================
      layer_1 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_1 = Conv3D(128, 3, padding='same', activation='relu')(layer_1); #layer_1 = BatchNormalization()(layer_1)
      layer_2 = Conv3D(128, 1, padding='same', activation='relu')(mid)
      layer_2 = Conv3D(128, 5, padding='same', activation='relu')(layer_2); #layer_2 = BatchNormalization()(layer_2)
      layer_3 = MaxPooling3D(3, strides=1, padding='same')(mid)
      layer_3 = Conv3D(128, 1, padding='same', activation='relu')(layer_3); #layer_3 = BatchNormalization()(layer_3)
      mid = Concatenate()([layer_1, layer_2, layer_3])
     #
      return mid

 

    ##########################################################################################################################
    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
    norm_layer_2 = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
        
    def create_SRS(norm_layer, norm_layer_2):
        inp,inp_ = Input(shape=(8,128,128,3)), Input(shape=(8,128,128,3))
        inp_diff = Input(shape=(8,32,32,1))

        scale_, offset = 1/255., 0

        x1, x2 = inp, inp_
        x1, x2 = Cropping3D((0,16,0))(x1), Cropping3D((0,16,0))(x2)
        
        x1 = norm_layer(x1)
        #x1 = experimental.preprocessing.Rescaling(scale_,offset)(x1)
        
        #x2 = norm_layer(x2)
        x2 = experimental.preprocessing.Rescaling(scale_,offset)(x2)
        

        x = inception_3(x1,x2,opt='', activation='relu') #opt for inception_3: Xtra_joint_processing or 0
        #x = inception_4(x1,x2)

        x =  Flatten()(x) #GlobalAveragePooling3D|Flatten()(x); 
        #x = Reshape([x.shape[1]*x.shape[2]*x.shape[3],x.shape[4]])(x); x = Bidirectional(LSTM(512,))(x)
        dim = x.shape[1]

        x = Dropout(0.2)(x)

        x = Dense(512, activation=tf.nn.leaky_relu)(x)

        x = Dense(64, activation=tf.nn.leaky_relu)(x)

        x = Dense(8, activation=tf.nn.softmax)(x)

        return models.Model(inputs=[inp,inp_], outputs=x) 

    model = create_SRS(norm_layer, norm_layer_2)
    optimizer = optimizers.SGD(0.001) #Adam(0.0002,0.5) # should by 0.0005
    METRICS = ['accuracy']#, tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(name='prc', curve='PR')]

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=METRICS)# loss=losses, loss_weights=lossWeights  # binary_crossentropy

    model.summary()
