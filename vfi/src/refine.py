
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv( out_planes, kernel_size=3, stride=1, padding='same', dilation=1):
    return keras.Sequential([
         keras.layers.Conv2D(
                                                filters=out_planes,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                dilation_rate=dilation,
                                                strides=stride
                                                ),

        tf.keras.layers.LeakyReLU(0.25)]
    )

def deconv( out_planes, kernel_size=4, stride=2, padding='same'):
    return keras.Sequential([
        tf.keras.layers.Conv2DTranspose(filters=out_planes, kernel_size=4, strides=2, padding='same', use_bias=True),
         tf.keras.layers.LeakyReLU(0.25)]
        )
            
class Conv2D(layers.Layer):
    def __init__(self, out_planes, stride=2):
        super(Conv2D, self).__init__()
        self.conv1 = conv( out_planes, kernel_size=3,stride= stride, padding='same')
        self.conv2 = conv( out_planes,kernel_size= 3,stride= 1, padding='same')

    def call(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        return x

class refine(tf.Module):
    def __init__(self, c, out=3):
        super(refine, self).__init__()
        self.down0 = Conv2D( 2*c,2)
        self.down1 = Conv2D( 4*c,2)
        self.down2 = Conv2D( 8*c,2)
        self.down3 = Conv2D( 16*c,2)
        self.up0 = deconv( 8*c)
        self.up1 = deconv( 4*c)
        self.up2 = deconv( 2*c)
        self.up3 = deconv( c)

        self.conv= keras.layers.Conv2D(
                                                filters=out,
                                                kernel_size=3,
                                                padding='same',
                                                dilation_rate=1,
                                                strides=1
                                                )


    def __call__(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(tf.concat([img0, img1, warped_img0, warped_img1, mask, flow,c0[0], c1[0]], -1))
        s1 = self.down1(tf.concat([s0, c0[1], c1[1]], -1))
        s2 = self.down2(tf.concat([s1, c0[2], c1[2]], -1))
        s3 = self.down3(tf.concat([s2, c0[3], c1[3]], -1))
        x = self.up0(tf.concat([s3, c0[4], c1[4]], -1))
        x = self.up1(tf.concat([x, s2], -1)) 
        x = self.up2(tf.concat([x, s1], -1)) 
        x = self.up3(tf.concat([x, s0], -1)) 
        x = self.conv(x)
        return tf.keras.activations.sigmoid(x)
