import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,BatchNormalization,Add,ZeroPadding2D,Flatten,Dense,Input,LeakyReLU,Softmax,ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
import pickle
import numpy as np
from PIL import Image



class AttentionConv(tk.layers.Layer):
    
    def __init__(self,output_channel,kernel_size,groups):
        super().__init__()
        self.output_channel = output_channel    
        self.kernel_size = kernel_size
        self.stride = 1
        self.groups = groups

        assert output_channel % groups == 0
        
        self.rel_h = tk.backend.variable(lambda : tk.backend.truncated_normal((1,1,kernel_size,1,output_channel//2),stddev = 0.1)) 
        #output_channels//2 is the number of channels on which the relative position will be considered,1 denotes the number of those filters and the one after that too and (kernel_size,1) denotes the size of that filter
        self.rel_w = tk.backend.variable(lambda : tk.backend.truncated_normal((1,1,1,kernel_size,output_channel//2),stddev = 0.1)) 

        self.key_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)
        self.query_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)
        self.value_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)

    def call(self,x):
        
        batch,height,width,channels = x.shape
        x_padded = ZeroPadding2D(padding=(self.kernel_size//2,self.kernel_size//2))(x)
        query = self.query_weights(x)
        value = self.value_weights(x_padded)
        key = self.key_weights(x_padded)
        #key,query and value will have the shape of (batch,height,width,depth)
        keys = tf.image.extract_patches(images = key,sizes = [1,self.kernel_size,self.kernel_size,1],strides = [1,self.stride,self.stride,1],rates = [1,1,1,1], padding = "VALID")
        value = tf.image.extract_patches(images = value,sizes = [1,self.kernel_size,self.kernel_size,1],strides = [1,self.stride,self.stride,1],rates = [1,1,1,1], padding = "VALID")
        no_of_kernels = key.shape[-2] - self.kernel_size + 1
        keys = tf.reshape(keys,shape = (-1,no_of_kernels, no_of_kernels,self.kernel_size,self.kernel_size,self.output_channel))
        key_split_h,key_split_w = tf.split(keys,num_or_size_splits = 2,axis = -1)
        key_with_rel = tk.layers.concatenate([key_split_h + self.rel_h,key_split_w + self.rel_w],axis = -1) 
        
        #reshaping the query and key
        key_with_rel = tf.reshape(key_with_rel,(-1,self.groups,no_of_kernels,no_of_kernels,self.kernel_size*self.kernel_size,self.output_channel//self.groups))
        query  = tf.reshape(query,(-1,self.groups,no_of_kernels,no_of_kernels,1,self.output_channel//self.groups))        
        value = tf.reshape(value,(-1,self.groups,no_of_kernels,no_of_kernels,self.kernel_size*self.kernel_size,self.output_channel//self.groups))
        
        #multiplication  of key and query
        #assert key_with_rel.shape == query.shape        
        key_prod_query = query*key_with_rel
        
        # Now the function is passed through the softmax and is multiplied with the values
        s = Softmax(axis = -2)(key_prod_query)
        y = tf.einsum('bnchwk,bnchwk->bnchk',s,value)
        y = tf.reshape(y,(-1,height,width,self.output_channel))
        return y

class AttentionStem(tk.layers.Layer):

    def __init__(self,output_channel,kernel_size,groups):
        super().__init__()
        self.output_channel = output_channel    
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = 1
        self.m = 4
        self.value_weights = self.create_layers() #helps in execution of the funciton create_layers
               
        self.emb_a = tk.backend.variable(lambda : tk.backend.truncated_normal((output_channel//groups,kernel_size),stddev = 0.1)) 
        #output_channels//2 is the number of channels on which the relative position will be considered,1 denotes the number of those filters and the one after that too and (kernel_size,1) denotes the size of that filter
        self.emb_b = tk.backend.variable(lambda : tk.backend.truncated_normal((output_channel//groups,kernel_size),stddev = 0.1)) 
        self.emb_mix = tk.backend.variable(lambda : tk.backend.truncated_normal((self.m,output_channel//groups),stddev = 0.1))

        self.key_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)
        self.query_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)
        self.s = Softmax()

    def create_layers(self):
        layers = [Conv2D(self.output_channel,kernel_size = 1,strides = self.stride) for _ in range(self.m)]
        return layers 

    def call(self,x):

        batch,height,width,channels = x.shape
        x_padded = ZeroPadding2D(padding=(self.kernel_size//2,self.kernel_size//2))(x)
        query = self.query_weights(x)
        value = tf.stack([self.value_weights[i](x_padded) for i in range(self.m)])
        key = self.key_weights(x_padded)
        #all the weights are initialized
        no_of_kernels = key.shape[-2] - self.kernel_size + 1 #it is equivalent to the height or width of the original image  i.e. x
        #dividing the image into different patches
        #value_out = []
        a = tf.image.extract_patches(images = value[0] , sizes = [1,self.kernel_size,\
            self.kernel_size,1], strides = [1,1,1,1],rates = [1,1,1,1],padding = "VALID")
        b = tf.image.extract_patches(images = value[1] , sizes = [1,self.kernel_size,\
            self.kernel_size,1], strides = [1,1,1,1],rates = [1,1,1,1],padding = "VALID")
        c = tf.image.extract_patches(images = value[2] , sizes = [1,self.kernel_size,\
            self.kernel_size,1], strides = [1,1,1,1],rates = [1,1,1,1],padding = "VALID")
        d = tf.image.extract_patches(images = value[3] , sizes = [1,self.kernel_size,\
            self.kernel_size,1], strides = [1,1,1,1],rates = [1,1,1,1],padding = "VALID")    
        
        value_out = tf.stack([a,b,c,d]) 
        value_out = tf.reshape(value_out,shape = (self.m,-1,no_of_kernels, no_of_kernels,self.kernel_size,self.kernel_size,\
                                                  self.output_channel))
        key_out = tf.image.extract_patches(images = key, sizes = [1,self.kernel_size,self.kernel_size,1], strides = [1,1,1,1],\
                                           rates = [1,1,1,1],padding = "VALID")
        key_out = tf.reshape(key_out, shape = (-1,self.groups,no_of_kernels,no_of_kernels,self.output_channel//self.groups,\
                                               self.kernel_size*self.kernel_size)) 
        query = tf.reshape(query, shape = (-1,self.groups,no_of_kernels,no_of_kernels,self.output_channel//self.groups,1))

        #calculating Softmax(p(a,b,m))
        emb_logit_a = tk.backend.dot(self.emb_mix,self.emb_a)   #[4(self.m),7(kernel size)] [4,1,7]
        emb_logit_b = tk.backend.dot(self.emb_mix,self.emb_b)    #[4,7,1]
        emb = tf.expand_dims(emb_logit_a,axis = 2) + tf.expand_dims(emb_logit_b,axis = 1) #[4,7,7]
        emb = self.s(emb)
        emb = tf.reshape(emb,shape = (self.m,1,1,1,self.kernel_size,self.kernel_size,1)) 

        V = emb * value_out #the final_value has been calculated
        V = tk.backend.sum(V,axis = 0)
        V = tf.reshape(V, shape = (-1,self.groups,no_of_kernels,no_of_kernels,self.output_channel//self.groups,self.kernel_size*self.kernel_size)) 
        
        out = Softmax(axis = -1)(key_out*query)
        out = tf.reshape(tf.einsum('bnchwk,bnchwk->bnchw',out,V), shape = (-1,height,width,self.output_channel))
        return out



import tensorflow as tf
from keras.layers import Input, Layer
from keras.models import Model

from keras import regularizers, initializers

class SelfAttention(Layer):
    def __init__(self, hidden_dim, k_size, Nh, strides=1, padding='SAME', m_for_stem=None, **kwargs):

        self.hidden_dim = hidden_dim
        self.k_size = k_size
        self.Nh = Nh
        self.strides=strides
        self.padding=padding
        self.m_for_stem = m_for_stem
        super(SelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.hidden_dim)

    def build(self, input_shape):

        input_dim = input_shape[-1]

        self.W_K = self.add_weight(name='W_K',
                                   shape=(self.Nh, self.hidden_dim // self.Nh, input_dim),
                                   initializer='he_normal',
                                   trainable=True,
                                   regularizer=regularizers.l2(1e-4))

        if self.m_for_stem is None:
            self.W_V = self.add_weight(name='W_V',
                                       shape=(self.Nh, self.hidden_dim // self.Nh, input_dim),
                                       initializer='he_normal',
                                       trainable=True,
                                       regularizer=regularizers.l2(1e-4))
        else:
            self.W_V = self.add_weight(name='W_V',
                                       shape=(self.Nh, self.hidden_dim // self.Nh, input_dim, self.m_for_stem),
                                       initializer='he_normal',
                                       trainable=True,
                                       regularizer=regularizers.l2(1e-4))

        self.W_Q = self.add_weight(name="W_Q",
                                   shape=(self.Nh, self.hidden_dim//self.Nh, input_dim),
                                   initializer='he_normal',
                                   trainable=True,
                                   regularizer=regularizers.l2(1e-4))



        self.Rel_W = self.add_weight(name = "Rel_W",
                                     shape = (self.Nh, 1, self.k_size, (self.hidden_dim//2)//self.Nh),
                                     initializer=initializers.truncated_normal(),
                                     trainable=True,
                                     regularizer=regularizers.l2(1e-4)
                                     )

        self.Rel_H = self.add_weight(name = "Rel_H",
                                     shape = (self.Nh, self.k_size, 1, (self.hidden_dim//2)//self.Nh),
                                     initializer=initializers.truncated_normal(),
                                     trainable=True,
                                     regularizer=regularizers.l2(1e-4)
                                     )


        if self.m_for_stem is not None:
            self.emb_a = self.add_weight(name = "emb_a",
                                         shape = (self.k_size, 1, self.hidden_dim // self.Nh),
                                         initializer='he_normal',
                                         trainable=True,
                                         regularizer=regularizers.l2(1e-4)
                                         )
            self.emb_b = self.add_weight(name = "emb_b",
                                         shape = (1, self.k_size, self.hidden_dim // self.Nh),
                                         initializer='he_normal',
                                         trainable=True,
                                         regularizer=regularizers.l2(1e-4)
                                         )

            self.emb_mix= self.add_weight(name = "emb_mix",
                                          shape = (self.m_for_stem, self.hidden_dim // self.Nh),
                                          initializer='he_normal',
                                          trainable=True,
                                          regularizer=regularizers.l2(1e-4)
                                          )


        super(SelfAttention, self).build(input_shape)


    def call(self, x, **kwarges):

        input_patches = tf.image.extract_patches(x,
                                                 sizes=[1, self.k_size, self.k_size, 1],
                                                 strides=[1, self.strides, self.strides, 1],
                                                 rates=[1, 1, 1, 1],
                                                 padding=self.padding)

        batch, out_row, out_col, sizes = input_patches.get_shape().as_list()


        input_patches = tf.reshape(input_patches,
                                   [-1, out_row, out_col, self.k_size ** 2, 1,  x.get_shape()[-1]])
        input_patches = tf.tile(input_patches, [1, 1, 1, 1, self.Nh, 1])

        x = tf.reshape(x, [-1, out_row, out_col, 1, x.get_shape()[-1]])
        x = tf.tile(x, [1,1,1, self.Nh, 1])
        Q = tf.einsum('BxyHi,Hoi->BxyHo', x, self.W_Q)
        K = tf.einsum('BxyKHi,Hoi->BxyKHo', input_patches, self.W_K)

        K = tf.transpose(K, [0,1,2,4,3,5])

        if self.m_for_stem is not None:
            emb = tf.add(self.emb_a, self.emb_b)
            emb = tf.reshape(emb, [self.k_size ** 2, self.hidden_dim // self.Nh])
            emb = tf.einsum("Ko,mo->Km", emb, self.emb_mix)
            softmax_emb = tf.nn.softmax(emb, axis=-1)

            V = tf.einsum("Hoim,Km->HoiK", self.W_V, softmax_emb)
            V = tf.transpose(V, [3,0,1,2])
            V = tf.einsum('BxyKHi,KHoi->BxyKHo', input_patches, V)
        else:
            V = tf.einsum('BxyKHi,Hoi->BxyKHo', input_patches, self.W_V)


        V = tf.transpose(V, [0, 1, 2, 4, 3, 5])
        dot_QK = tf.einsum('BxyHKo,BxyHo->BxyHK', K, Q)



        height = tf.tile(self.Rel_H, [1, 1, self.k_size, 1])
        width = tf.tile(self.Rel_W, [1, self.k_size, 1, 1])

        rel_pos = tf.concat([height, width], axis=-1)
        rel_pos = tf.reshape(rel_pos, [self.Nh, self.k_size**2, self.hidden_dim//self.Nh])
        rel_pos = tf.einsum('BxyHo,HKo->BxyHK', Q, rel_pos)
        #rel_pos = tf.divide(rel_pos, tf.sqrt(tf.cast(self.hidden_dim // self.Nh, dtype='float32')))
        dot_QK = tf.add(dot_QK, rel_pos)

        dot_QK = tf.divide(dot_QK, tf.sqrt(tf.cast(self.hidden_dim // self.Nh, dtype='float32')))

        dot_QK = tf.nn.softmax(dot_QK, axis=-1)
        out = tf.einsum('BxyHK,BxyHKo->BxyHKo', dot_QK, V)

        out = tf.reduce_sum(out, axis=-2)

        out = tf.reshape(out, [-1, out_row, out_col, self.hidden_dim])
        return out



if __name__ == "__main__":
    import os

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    input_tensor = Input((320, 320, 16))
    sa = SelfAttention(128, 5, 8, m_for_stem=4)(input_tensor)
    model = Model(input_tensor, sa)
    model.summary()

