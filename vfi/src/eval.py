# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for models."""

import functools
import tensorflow as tf
from vfi.src import models 
from vfi.src import config 
from vfi.src import vimeo_datasets 
import argparse
from tqdm import tqdm
import wandb as wandb
from wandb.keras import WandbCallback
import numpy as np
from matplotlib import pyplot as plt

wandb.login()
run = wandb.init(project='vfi_eval')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size' , default=64, type = int, help= 'batch size')
parser.add_argument('--num_epoch' , default=300 , type = int, help= 'num_epoch')
parser.add_argument('--resume' , default='/home/jmw/backup/ema_tf/vfi/ckpt' , type = str, help= 'resume')
#parser.add_argument('--resume' , default=None , type = str, help= 'resume path')
parser.add_argument('--write_ckpt_dir' , default='/home/jmw/backup/ema_tf/vfi/ckpt' , type = str, help= 'ckpt directory path')
args= parser.parse_args()

feature_extractor_cfg, flow_estimation_cfg = config.MODEL_CONFIG['MODEL_ARCH']
model= models.Model( **flow_estimation_cfg)

class ModelsTest(tf.test.TestCase):

  def eval(self,model) :
    test_step = tf.function(model.interpolator)
    test_loader=vimeo_datasets.VimeoDataset(dataset_name='test',
               path='/data/dataset/vimeo_dataset/vimeo_triplet',
               batch_size=1,
               shuffle=False)
    loss=[]
    psnr=[]
    mse=[]
    with tqdm(test_loader , unit ='batch' ) as tepoch :
      for img0,gt,img1 in tepoch:
        
        img0 = tf.cast(img0/255.0, dtype=tf.float32)
        gt = tf.cast(gt/255.0, dtype=tf.float32)
        img1 = tf.cast(img1/255.0, dtype=tf.float32)

        pred, metrics = test_step(img0,gt,img1,0.5)

        loss.append(metrics['loss'].numpy())
        psnr.append(metrics['psnr'].numpy())
        mse.append(metrics['mse'].numpy())

        #run.Image( np.round(pred.numpy()*255))
        run.log({ "PSNR" : metrics['psnr'].numpy(), 'LOSS' : metrics['loss'].numpy() ,'MSE':metrics['mse'].numpy() })
    psnr=np.mean(psnr)
    loss=np.mean(loss)
    mse=np.mean(mse)
    print("Avg PSNR: {} Avg loss: {}  Avg MSE {}  ".format(psnr,loss ,mse ))
    
    run.finish()

  def test_train_eval(self):
    if args.resume != None :
      print('resume',args.resume)
   
      ckpt_p= args.resume
      model.load_ckpt(path=args.resume)

    self.eval(model)
   

if __name__ == "__main__":
  tf.test.main()
