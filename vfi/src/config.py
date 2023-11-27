from tensorflow import keras



'''==========Model config=========='''
def init_model_config(F=16, W=8, depth=[2, 2, 2, 2, 2]):
    '''This function should not be modified'''
    #backborn / multiscale
    return { #feature_extractor_cfg
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*F//32, 16*F//32],
        'mlp_ratios':[4, 4],
        'qkv_bias':True,
        'norm_layer':keras.layers.LayerNormalization(epsilon=1e-5), 
        'depths':depth,
        'window_sizes':W
    }, { #flow_estimation_cfg
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*F//32, 16*F//32],
        'window_sizes':W,
        'scales':[4, 8, 16],
        'hidden_dims':[4*F, 4*F],
        'c':F
    }

MODEL_CONFIG = {
    'LOGNAME': 'ifa',
    'MODEL_ARCH': init_model_config(
        F = 16,
        W = [8,8],
        depth = [2,2,2,2,2]
    )
}

