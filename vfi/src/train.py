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
import os
wandb.login()
run = wandb.init(project='vfi_ema')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size' , default=32, type = int, help= 'batch size')
parser.add_argument('--num_epoch' , default=300 , type = int, help= 'num_epoch')
#parser.add_argument('--resume' , default='/home/jmw/backup/ema_tf/vfi/ckpt' , type = str, help= 'resume')
parser.add_argument('--resume' , default=None , type = str, help= 'resume path')
parser.add_argument('--write_ckpt_dir' , default='vfi/ckpt' , type = str, help= 'ckpt directory path')
args= parser.parse_args()

feature_extractor_cfg, flow_estimation_cfg = config.MODEL_CONFIG['MODEL_ARCH']
model= models.Model( **flow_estimation_cfg)

if not os.path.exists(args.write_ckpt_dir) : 
        os.mkdir(args.write_ckpt_dir)

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

        _, metrics = test_step(img0,gt,img1,0.5)

        loss.append(metrics['loss'].numpy())
        psnr.append(metrics['psnr'].numpy())
        mse.append(metrics['mse'].numpy())

      print("Avg PSNR: {} loss: {}  MSE {}  ".format(np.mean(psnr), np.mean(loss), np.mean(mse)))



  def _restore_evaluate(self, ckpt_p=None):
    """Restore and evaluate a model for the given stage."""
    print("""Restore and evaluate a model for the given stage.""")
    if ckpt_p == None:
      model.load_ckpt(args.write_ckpt_dir)
    else : 
      ckpt = tf.train.Checkpoint(model=model)
      ckpt.restore(ckpt_p).assert_existing_objects_matched()
      #ckpt.restore(ckpt_p).expect_partial()
    self.eval(model)
    

  def test_train_eval(self):
    train_loader= vimeo_datasets.VimeoDataset(dataset_name='train',
               path='/data/dataset/vimeo_dataset/vimeo_triplet',
               batch_size=args.batch_size,
               mode='full',
               shuffle=True)
    
    train_step = tf.function(model.train_step)

    if args.resume != None :
      print('resume',args.resume)
   
      ckpt_p= args.resume
      model.load_ckpt(path=args.resume)

    for epoch in (range(args.num_epoch)):
      with tqdm(train_loader , unit ='batch' ) as tepoch :
        for img0,gt,img1 in (tepoch):
          img0 = tf.cast(img0/255.0, dtype=tf.float32)
          gt = tf.cast(gt/255.0, dtype=tf.float32)
          img1 = tf.cast(img1/255.0, dtype=tf.float32)

          metrics=train_step(img0,gt,img1)
          tepoch.set_postfix(epoch=epoch,psnr=metrics['psnr'].numpy() , loss =metrics['loss'].numpy() ,mse =metrics['mse'].numpy()  )
          run.log(metrics)
            
        if tf.equal(epoch % 1, 0):
          ckpt_p = model.write_ckpt(args.write_ckpt_dir, step=epoch)
          print(ckpt_p)
          self._restore_evaluate(ckpt_p)
    run.finish()

if __name__ == "__main__":
  tf.test.main()
