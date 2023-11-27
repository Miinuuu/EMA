import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import numpy as np
import tensorflow_addons.image as tfa_image
from vfi.src import sasa

@tf.function
def warp(image: tf.Tensor, flow: tf.Tensor) -> tf.Tensor:
  """Backward warps the image using the given flow.

  Specifically, the output pixel in batch b, at position x, y will be computed
  as follows:
    (flowed_y, flowed_x) = (y+flow[b, y, x, 1], x+flow[b, y, x, 0])
    output[b, y, x] = bilinear_lookup(image, b, flowed_y, flowed_x)

  Note that the flow vectors are expected as [x, y], e.g. x in position 0 and
  y in position 1.

  Args:
    image: An image with shape BxHxWxC.
    flow: A flow with shape BxHxWx2, with the two channels denoting the relative
      offset in order: (dx, dy).
  Returns:
    A warped image.
  """
  # tfa_image.dense_image_warp expects unconventional negated optical flow, so
  # negate the flow here. Also revert x and y for compatibility with older saved
  # models trained with custom warp op that stored (x, y) instead of (y, x) flow
  # vectors.
  flow = -flow[..., ::-1]

  # Note: we have to wrap tfa_image.dense_image_warp into a Keras Lambda,
  # because it is not compatible with Keras symbolic tensors and we want to use
  # this code as part of a Keras model.  Wrapping it into a lambda has the
  # consequence that tfa_image.dense_image_warp is only called once the tensors
  # are concrete, e.g. actually contain data. The inner lambda is a workaround
  # for passing two parameters, e.g you would really want to write:
  # tf.keras.layers.Lambda(tfa_image.dense_image_warp)(image, flow), but this is
  # not supported by the Keras Lambda.
  warped = tf.keras.layers.Lambda(
      lambda x: tfa_image.dense_image_warp(*x))((image, flow))
  return tf.reshape(warped, shape=tf.shape(image))


@tf.function
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C]) 
    # TODO contiguous memory access?
    windows = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, C])
    return windows


@tf.function
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2, 4, 5]), [B, H, W, -1])
    return x

class DropPath(layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output




class InterFrameAttention(layers.Layer):
    def __init__(
        self, 
        dim,
        motion_dim, 
        window_size,
        num_heads, 
        qkv_bias=True, 
        proj_drop=0.0,
        attn_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.motion_dim= motion_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = tf.keras.layers.Dense(dim , use_bias=qkv_bias,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.kv = layers.Dense(dim*2 , use_bias=qkv_bias,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj_drop = layers.Dropout(proj_drop)
        self.proj = layers.Dense(dim,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
        self.cor_embed = layers.Dense(motion_dim, use_bias=qkv_bias,)
        self.motion_proj = layers.Dense( motion_dim,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

    def call(self, x1,x2,cor ,mask=None):
        B_NW, WS, C = x1.shape
        
        assert self.dim ==C
        head_dim = C // self.num_heads
        motion_dim= self.motion_dim // self.num_heads
        q= self.q(x1)
        q= tf.reshape(q, shape=(B_NW,WS,self.num_heads,head_dim))
        q= tf.transpose(q, perm= ( 0 ,2,1,3))

        kv= self.kv(x2)
        kv= tf.reshape(kv, shape=(B_NW,WS,2,self.num_heads,head_dim))
        kv= tf.transpose(kv,perm=(2,0,3,1,4))
        k, v = kv[0], kv[1] 
        cor_embed_= self.cor_embed(cor)
        cor_embed = tf.reshape( cor_embed_, shape=(B_NW,WS,self.num_heads,motion_dim))
        cor_embed = tf.transpose( cor_embed,  perm= (0,2,1,3))

        attn = (q @ tf.transpose(k, perm=(0, 1, 3, 2)))* self.scale

        if mask is not None:
            nW = mask.shape[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, WS, WS))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, WS, WS))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)
        af = attn @ v 
        af = tf.transpose(af, perm=(0, 2, 1, 3))
        af = tf.reshape(af, shape=(B_NW, WS, C))
        af = self.proj(af)
        af = self.proj_drop(af)

        mf=(attn @ cor_embed)
        mf= tf.transpose( mf , perm = (0,2,1,3))
        mf = tf.reshape( mf , shape= (B_NW,WS,self.motion_dim))
        mf  = self.motion_proj(mf-cor_embed_) 

        return af,mf

class DWConv(layers.Layer):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = keras.layers.Conv2D(
                                                filters=dim,
                                                kernel_size=3,
                                                padding='same',
                                                groups=dim)
        self.dim=dim
    def call(self, x, H, W):
        B, _, C = x.shape 
        x= tf.reshape(x, shape=(B,H,W,C))
        x = self.dwconv(x)
        x= tf.reshape(x,shape= (B,H*W,self.dim))        
        return x

_truncnormal_init = lambda: tf.keras.initializers.TruncatedNormal(stddev=0.02)

class Mlp(layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer= layers.Activation(keras.activations.gelu), dropout_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features, kernel_initializer=_truncnormal_init())
        self.dwconv = DWConv(hidden_features)
        self.act = layers.Activation(keras.activations.gelu)
        self.fc2 = layers.Dense(out_features, kernel_initializer=_truncnormal_init())
        self.drop = layers.Dropout(dropout_rate)

    def call(self, x,H,W):
        x = self.fc1(x)
        x= self.dwconv(x,H,W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        


def depad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size) * window_size - h
    pad_w = math.ceil(w / window_size) * window_size - w
    if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
        return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :]
    return x

def pad_if_needed(x, window_size):
    _, h, w, _ = x.shape # x.size() 2,60,108,128
     
    pad_h = math.ceil(h / window_size) * window_size - h
    pad_w = math.ceil(w / window_size) * window_size - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        img_mask = (np.zeros([1, h+pad_h, w+pad_w, 1]))
        h_slices = (
            slice(0, pad_h//2),
            slice(pad_h//2, h+pad_h//2),
            slice(h+pad_h//2, None),
        )
        w_slices = (
            slice(0, pad_w//2),
            slice(pad_w//2, w+pad_w//2),
            slice(w+pad_w//2, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                #img_mask[:, h, w, :].assign(cnt)
                cnt += 1
        img_mask = tf.convert_to_tensor(img_mask,dtype=tf.float32)

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size*window_size, 1       
        mask_windows = tf.reshape(
                mask_windows, shape=[-1, window_size * window_size]
            )
        attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )

        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
        padding= [ [0,0], [pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2   ]  ,[0,0]    ]
        return tf.pad( x,padding), attn_mask
    

    return x, None


class MotionFormerBlock(layers.Layer):

    def __init__(
        self,
        dim, 
        motion_dim, 
        num_heads, 
        window_size=8, 
        shift_size=0, 
        mlp_ratio=4., 
        qkv_bias=False, 
        dropout_rate=0., 
        attn_drop=0.,
        drop_path=0.,
        ):

        super().__init__()
        
        self.window_size = window_size  # size of window        
        self.shift_size = shift_size  # size of window shift

        self.norm1 = layers.LayerNormalization(epsilon=1e-5,axis=-1)
        
        
        self.attn = InterFrameAttention(
                                        dim,
                                        motion_dim, 
                                        window_size=self.window_size,
                                        num_heads=num_heads, 
                                        qkv_bias=qkv_bias, 
                                        proj_drop=dropout_rate,
                                        attn_drop=attn_drop )



        #self.drop_path =  DropPath(drop_path) 
        self.drop_path =  DropPath(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = layers.LayerNormalization(epsilon=1e-5,axis=-1)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer= layers.Activation(keras.activations.gelu), dropout_rate=dropout_rate)



    def call(self, x,cor,H,W,B):
        x = tf.reshape(x, shape=(B, H, W, -1))

        x_pad, mask = pad_if_needed(x,  self.window_size)
        cor_pad, _ = pad_if_needed(cor, self.window_size)

        if self.shift_size > 0: 
            _, H_p, W_p, C = x_pad.shape

            x_pad = tf.roll(
                x_pad, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
            cor_pad = tf.roll(
                cor_pad, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
            
  
            shift_mask = np.zeros((1, H_p, W_p, 1))  # 1 H W 1
            #shift_mask =tf.Variable(tf.zeros([1, H_p, W_p, 1], tf.float32), trainable=False)
            #shift_mask =tf.experimental.numpy.zeros([1, H_p, W_p, 1], tf.float32)

            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :]=(cnt) 
                    cnt += 1
            shift_mask = tf.convert_to_tensor(shift_mask,dtype=tf.float32)

            mask_windows = window_partition(shift_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )

            shift_mask =tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )

            shift_mask = tf.where(shift_mask != 0, -100.0, shift_mask)
            shift_mask = tf.where(shift_mask == 0, 0.0, shift_mask)

            if mask is not None:
                shift_mask = tf.where(mask != 0, -100.0, shift_mask)
        else: 
            shift_mask = mask


        _, Hw, Ww, _ = x_pad.shape #2 ,60,120,128

        
        x_win = window_partition(x_pad, self.window_size)
        nwB = x_win.shape[0] 
        x_win = tf.reshape(
            x_win, shape=(nwB, self.window_size * self.window_size, -1)
        )
        cor_win = window_partition(cor_pad, self.window_size)
        cor_win = tf.reshape(
            cor_win, shape=(nwB, self.window_size * self.window_size, -1)
        )

        x_norm = self.norm1(x_win)

        x_reverse = tf.concat([x_norm[nwB//2:], x_norm[:nwB//2] ],0) # 배치순서반대 frame t-1. frame t+1 반대로  interframe attention을 위해서  {f1,f0}
        x_appearence, x_motion = self.attn(x_norm, x_reverse,cor_win, mask=shift_mask)
        x_norm = x_norm + self.drop_path(x_appearence)
        
        
        x_norm = tf.reshape(
            x_norm, shape=(nwB, self.window_size, self.window_size, -1)
        )
        x_motion = tf.reshape(
            x_motion, shape=(nwB, self.window_size, self.window_size, -1)
        )

        x_back_win = window_reverse(x_norm, self.window_size ,Hw,Ww)
        x_motion = window_reverse(x_motion, self.window_size, Hw, Ww)

        if self.shift_size > 0 :
            x_back_win = tf.roll(x_back_win, shift=[self.shift_size, self.shift_size ], axis=[1, 2])
            x_motion = tf.roll(x_motion, shift=[self.shift_size , self.shift_size ], axis=[1, 2])
     
        x = depad_if_needed(x_back_win, x.shape, self.window_size)
        x_motion = depad_if_needed(x_motion, cor.shape, self.window_size)

        x = tf.reshape(x, shape=(B, H * W, -1))
        x_motion = tf.reshape(x_motion, shape=(B, H * W, -1))

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
  
        return x,x_motion
    


class ConvBlock(layers.Layer):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=keras.layers.ReLU()):
        super().__init__()
        self.layers = keras.Sequential()

        for i in range(depths):
            if i == 0:
                self.layers.add(keras.layers.Conv2D( filters=out_dim, kernel_size=3,strides= 1,padding='same' ))
            else:
                self.layers.add(keras.layers.Conv2D( filters=out_dim, kernel_size=3,strides= 1,padding='same' ))

            self.layers.add( keras.layers.ReLU())

    def call(self, x):
        x = self.layers(x)
        return x


class OverlapPatchEmbed(layers.Layer):
    def __init__(self, patch_size=3, stride=2, in_chans=128, embed_dim=256):
        super().__init__()
        self.proj = keras.layers.Conv2D( filters=embed_dim, kernel_size=patch_size,strides= stride,padding='same' ) # in_dims= 16 * 7 , 128       
        self.norm = keras.layers.LayerNormalization(epsilon=1e-5,axis=-1)

    def call(self, x):
        x = self.proj(x)
        B, H, W, C = x.shape
        x= tf.reshape(x, shape= (B,-1,C))
        x = self.norm(x)
    
        return x, H, W


class CrossScalePatchEmbed(layers.Layer):
    def __init__(self, in_dims=[16,32,64], embed_dim=128):
        super().__init__()

        base_dim = in_dims[0] # 16
        self.layers = []
        for i in range(len(in_dims)): #i=0,1,2
            for j in range(2 ** i): #  j=0,/ 0,1 /0,1,2,3  # in_dims 64 32 32 16 16 16 16 
                self.layers.append(keras.layers.Conv2D(filters=base_dim, kernel_size= 2**(i+1), strides=2**(i+1), padding='same', dilation_rate=1))
                
        self.proj = keras.layers.Conv2D( filters=embed_dim, kernel_size=1,strides= 1,padding='same' ) # in_dims= 16 * 7 , 128       
        self.norm = keras.layers.LayerNormalization(epsilon=1e-5,axis=-1)

    def call(self, x_c):
        ys = []
        k = 0
        for i in range(len(x_c)): # 0,1,2
            for _ in range(2 ** i): # 0. 0,1 , 0,1,2,3
                ys.append(self.layers[k](x_c[-1-i]))
                k += 1
        x = self.proj(tf.concat(ys,-1)) # 1x1 conv

        B, H, W,C = x.shape
        x= tf.reshape(x, shape= (B,-1,C))
        x = self.norm(x)
        return x, H, W
    
def conv( out_planes, kernel_size=3, stride=1, padding='same', dilation=1):
    return keras.Sequential([
         keras.layers.Conv2D(
                                                filters=out_planes,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                dilation_rate=dilation,
                                                strides=stride
                                                ),

        tf.keras.layers.ReLU()]
    )



class Head(layers.Layer):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()


        self.scale = scale

        self.conv = keras.Sequential([
                                  #sasa.AttentionConv(output_channel=c,kernel_size=3,groups=1),
                                  #sasa.AttentionConv(output_channel=c,kernel_size=3,groups=1),
                                  #sasa.AttentionConv(output_channel=6,kernel_size=3,groups=1)]
                                  #sasa.SelfAttention(c,3,1),
                                  #sasa.SelfAttention(c,3,1),
                                  #sasa.SelfAttention(6,3,1)]
                                  conv(c),
                                  conv(c),
                                  conv(5)] # 0:4 flow 4:5 mask]
                                  )  

    def call (self, motion_feature, x, flow): # /16 /8 /4

        B,H,W,C= x.shape
        #motion_feature = tf.nn.depth_to_space(motion_feature,2) #/4 /2 /1 pixelshuffle 두번 
        #motion_feature = tf.nn.depth_to_space(motion_feature,2) #/4 /2 /1 pixelshuffle 두번 
        motion_feature = tf.nn.depth_to_space(motion_feature,4) #/4 /2 /1 pixelshuffle 두번 
        #motion_feature= tf.image.resize(motion_feature, [ motion_feature.shape[1] * 4  , motion_feature.shape[2]*4 ] )
        if self.scale != 4: # 이미지 줄여서  입력 

            x=tf.image.resize(x, [ x.shape[1] * 4 // self.scale , x.shape[2]*4//self.scale] )
        
        if flow != None:

            if self.scale != 4:
                flow=tf.image.resize(flow, [ flow.shape[1] * 4 // self.scale ,flow.shape[2]*4//self.scale] )* 4. / self.scale

            x = tf.concat((x, flow), -1) # 13+ 4 channel         img0 img1. warped_img0, warped_img1 , mask, flow = 17 channe; 

        '''
        motion_feature_size torch.Size([1, 512, 32, 32])
        motion_feature_size torch.Size([1, 32, 128, 128])
        x_size torch.Size([1, 6, 128, 128])
        motion_feature_size torch.Size([1, 512, 32, 32])
        motion_feature_size torch.Size([1, 32, 128, 128])
        x_size torch.Size([1, 17, 128, 128])
        '''
        x = self.conv(tf.concat([motion_feature, x], -1))    #64+6, 32+17
        
        if self.scale != 4: #줄인 아웃풋 이미지와 플로우 다시 늘리기 
            pass
            x=tf.image.resize(x, [ x.shape[1] *  self.scale//4 ,x.shape[2] * self.scale//4 ])
            flow = x[...,:4] * (self.scale // 4)
        else:
            flow = x[...,:4]
        
        mask = x[...,4:5]
        
        return flow, mask
    


class feature_extractor(layers.Layer):
    def __init__(self, in_chans=3, embed_dims=[16 ,32 ,64 ,128 ,256], motion_dims=[0,0,0,64,256], num_heads=[4, 8], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=keras.layers.LayerNormalization(epsilon=1e-5,axis=-1),
                 depths=[2, 2, 2, 2, 2], window_sizes=[8, 8], **kwarg):
        super().__init__()
        
        self.depths = depths
        self.num_stages = len(embed_dims)
        dpr = [ (x.numpy()) for x in tf.linspace(0., drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads) # 3

        for i in range(self.num_stages): #5
            if i == 0: # 처음은 컨브블럭  C
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
                #block = ConvBlock(1,embed_dims[i],depths[i])
            
            else: # C C C T T  ,  C C 
                if i < self.conv_stages: #3 , 1, 2
                    patch_embed= conv( embed_dims[i],3,2,'same')
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:  # i T T 3,4 
                    if i == self.conv_stages: #3 bottleneck
                        patch_embed = CrossScalePatchEmbed(embed_dims[:i],
                                                        embed_dim=embed_dims[i])
                    else: # i 4  
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    block =             (
                                            
                                            [ MotionFormerBlock(
                                                dim=embed_dims[i], 
                                                motion_dim=motion_dims[i], 
                                                num_heads=num_heads[i-self.conv_stages], 
                                                window_size=window_sizes[i-self.conv_stages], 
                                                #shift_size= 0 if (j % 2) == 0 else window_sizes[i-self.conv_stages] // 2,
                                                shift_size= 0 ,
                                                mlp_ratio=mlp_ratios[i-self.conv_stages], 
                                                qkv_bias=qkv_bias, 
                                                dropout_rate=drop_rate, 
                                                attn_drop=attn_drop_rate, 
                                                drop_path=dpr[cur + j])
                                            for j in range(depths[i]) ]
                                            ) # i = 3,4

                    norm =keras.layers.LayerNormalization(epsilon=1e-5,axis=-1)

                    setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]  # 32
            setattr(self, f"block{i + 1}", block)
        self.cor = {}

    @tf.function
    def get_cor(self, shape):
        
        tenHorizontal = tf.linspace(-1.0, 1.0, shape[2]) 
        tenHorizontal = tf.reshape(tenHorizontal, shape= (1,1,1,shape[2]))
        tenHorizontal =tf.broadcast_to(tenHorizontal , [ shape[0],1, shape[1],shape[2] ])
        tenHorizontal =tf.transpose(tenHorizontal , perm=(0,2,3,1))

        
        tenVertical = tf.linspace(-1.0, 1.0, shape[1]) 
        tenVertical = tf.reshape(tenVertical, shape= (1,1,shape[1],1))
        tenVertical =tf.broadcast_to(tenVertical , [ shape[0],1, shape[1],shape[2] ])
        tenVertical =tf.transpose(tenVertical , perm=(0,2,3,1))

        self.cor = tf.concat([tenHorizontal,tenVertical], -1)

        return self.cor

    def call(self, x1, x2):
        B = x1.shape[0] # batch size 
        x = tf.concat([x1, x2], 0) # 배치방향으로 콘켓 이유?
        
        motion_features = []
        appearence_features = []
        x_c = [] # conv_stage 

        for i in range(self.num_stages): # 5,  i 0,1,2,3,4
            
            motion_features.append([])
            
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            norm = getattr(self, f"norm{i + 1}",None)
            
            if i < self.conv_stages: #i < 3
                if i > 0: # 처음은 패치 임베딩 안함.  1,2
                    x = patch_embed(x) # C(x)-C(0)-C(0)
                x = block(x)  # C C C

                x_c.append(x) # covolution output 
            
            else: # i >= 3
                
                if i == self.conv_stages: # 3
                    x, H, W = patch_embed(x_c) # cross scale embedding  convolution stride 2
                    ######
                else: #4
                    x, H, W = patch_embed(x) # overlap patch
                
                cor = self.get_cor((x.shape[0], H, W))

                for blk in (block):
                    x, x_motion  = blk(x, cor, H, W,2*B) #motion former block
                    x_motion = tf.reshape(x_motion , shape =(2*B, H,W ,-1))
                    motion_features[i].append(x_motion)
                
                motion_features[i] = tf.concat(motion_features[i], -1)
                x = norm(x)
                x= tf.reshape(x, shape=(2*B,H,W,-1))
            
            appearence_features.append(x) # CCCTT
        
        return appearence_features, motion_features
    

if __name__ == "__main__":


 dim=32
 motion_dim=16
 num_heads=4
 H=32
 W=32
 B=2


 input = tf.ones((B,H,W,dim), tf.float32) 
 cor = tf.ones((B,H,W,2), tf.float32) 


 model=MotionFormerBlock(
                            dim, 
                            motion_dim, 
                            num_heads, 
                            window_size=8, 
                            shift_size=0, 
                            mlp_ratio=4., 
                            qkv_bias=True, 
                            dropout_rate=0., 
                            attn_drop=0.,
                            drop_path=0.,
                            )
 
 model(input,cor,H,W,B)